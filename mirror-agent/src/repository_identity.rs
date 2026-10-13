//! Which repository a session works in, stable across clones and worktrees.

use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

/// The repository a session works in: its toplevel and a name shared by every
/// checkout of the same origin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepositoryIdentity {
    repository: String,
    toplevel: PathBuf,
}

impl RepositoryIdentity {
    /// Find the enclosing git toplevel and name the repository by its
    /// normalized origin URL, falling back to the canonical toplevel path when
    /// there is no remote.
    pub fn discover(workspace: &Path) -> Option<Self> {
        let workspace = std::fs::canonicalize(workspace).ok()?;
        let toplevel = workspace
            .ancestors()
            .find(|directory| is_repository_root(directory))?
            .to_path_buf();
        let repository = origin_url(&toplevel)
            .as_deref()
            .and_then(normalize_origin)
            .unwrap_or_else(|| toplevel.to_string_lossy().into_owned());
        Some(Self {
            repository,
            toplevel,
        })
    }

    /// The normalized origin URL, or the canonical toplevel path when there is
    /// no remote. Memory scopes compare this string for equality.
    pub fn repository(&self) -> &str {
        &self.repository
    }

    pub fn toplevel(&self) -> &Path {
        &self.toplevel
    }

    /// Express `path` relative to the toplevel with forward slashes; relative
    /// input is taken against `workspace`. Lexical, like
    /// `workspace::relative_within`: the paths a session hands over are already
    /// resolved by the tools, so nothing is read from disk.
    pub fn relative(&self, path: &Path, workspace: &Path) -> Option<String> {
        crate::relative_within(&self.toplevel, workspace, path)
    }
}

/// Whether `directory` is a checkout root: it holds a `.git` or `.jj` entry.
pub fn is_repository_root(directory: &Path) -> bool {
    directory.join(".git").exists() || directory.join(".jj").exists()
}

/// The name every clone of one repository agrees on: `host[:port]/path` read
/// from the origin URL. Accepted forms are `scheme://[user@]host[:port]/path`
/// for the ssh, https, http and git schemes, and scp-style `[user@]host:path`.
/// The host is lowercased, the user dropped, the port kept only when it is not
/// the scheme's default, and the path loses its leading and trailing `/` and a
/// `.git` suffix. `None` for a remote that names no host, such as a local or
/// relative path.
pub fn normalize_origin(url: &str) -> Option<String> {
    let trimmed = url.trim();
    let (default_port, authority, path) = match trimmed.split_once("://") {
        Some((scheme, rest)) => {
            let default_port = match scheme {
                "ssh" => 22,
                "https" => 443,
                "http" => 80,
                "git" => 9418,
                _ => return None,
            };
            let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
            (Some(default_port), authority, path)
        }
        None => {
            let (authority, path) = trimmed.split_once(':')?;
            if authority.contains('/') {
                return None;
            }
            (None, authority, path)
        }
    };
    let host_port = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    let (host, port) = match host_port.split_once(':') {
        Some((host, port)) => (host, Some(port.parse::<u16>().ok()?)),
        None => (host_port, None),
    };
    let host = host.to_lowercase();
    let path = path.trim_matches('/');
    let path = path.strip_suffix(".git").unwrap_or(path);
    if host.is_empty() || host.starts_with('.') || path.is_empty() {
        return None;
    }
    match port {
        Some(port) if Some(port) != default_port => Some(format!("{host}:{port}/{path}")),
        _ => Some(format!("{host}/{path}")),
    }
}

fn origin_url(toplevel: &Path) -> Option<String> {
    let mut command = Command::new("git");
    command.env_remove("GIT_DIR");
    command.env_remove("GIT_WORK_TREE");
    command.env_remove("GIT_COMMON_DIR");
    if toplevel.join(".git").exists() {
        command.arg("-C").arg(toplevel);
    } else {
        command.arg("--git-dir").arg(jj_git_dir(toplevel)?);
    }
    let output = command
        .args(["remote", "get-url", "origin"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let url = String::from_utf8(output.stdout).ok()?;
    let url = url.trim();
    (!url.is_empty()).then(|| url.to_string())
}

/// The config of the git store behind a Jujutsu workspace, so every workspace
/// of one repository reads the same origin. A secondary workspace's `.jj/repo`
/// is a file naming the shared repo directory relative to `.jj`; the store's
/// `git_target` names its git directory relative to the store.
fn jj_git_dir(toplevel: &Path) -> Option<PathBuf> {
    let dot_jj = toplevel.join(".jj");
    let mut repo = dot_jj.join("repo");
    if repo.is_file() {
        repo = dot_jj.join(std::fs::read_to_string(&repo).ok()?.trim());
    }
    let store = repo.join("store");
    Some(
        store.join(
            std::fs::read_to_string(store.join("git_target"))
                .ok()?
                .trim(),
        ),
    )
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use pretty_assertions::assert_ne;

    use super::*;

    fn git(path: &Path, args: &[&str]) {
        let output = Command::new("git")
            .arg("-C")
            .arg(path)
            .args(args)
            .output()
            .expect("run git");
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn init_repo(path: &Path) {
        std::fs::create_dir_all(path).expect("repo directory");
        git(path, &["init", "-q"]);
    }

    #[test]
    fn origin_forms_normalize_to_one_name() {
        let expected = Some("github.com/example/project".to_string());
        assert_eq!(
            normalize_origin("git@github.com:example/project.git"),
            expected
        );
        assert_eq!(
            normalize_origin("ssh://git@github.com/example/project.git"),
            expected
        );
        assert_eq!(
            normalize_origin("https://github.com/example/project"),
            expected
        );
        assert_eq!(
            normalize_origin("https://github.com:443/example/project/"),
            expected
        );
    }

    #[test]
    fn a_nondefault_port_is_part_of_the_name() {
        assert_eq!(
            normalize_origin("ssh://git@Code.Example:2222/team/repo.git"),
            Some("code.example:2222/team/repo".into())
        );
    }

    #[test]
    fn remotes_that_name_no_host_have_no_name() {
        assert_matches!(normalize_origin(""), None);
        assert_matches!(normalize_origin("projects/repo.git"), None);
        assert_matches!(normalize_origin("../upstream.git"), None);
        assert_matches!(normalize_origin("/tmp/local/repo.git"), None);
        assert_matches!(normalize_origin("file:///tmp/local/repo.git"), None);
        assert_matches!(normalize_origin("https://github.com/"), None);
        assert_matches!(normalize_origin("ssh://git@/example/project.git"), None);
    }

    #[test]
    fn relative_origins_do_not_merge_unrelated_checkouts() {
        let temp = tempfile::tempdir().expect("temp");
        let first = temp.path().join("first");
        let second = temp.path().join("second");
        init_repo(&first);
        init_repo(&second);
        git(&first, &["remote", "add", "origin", "projects/repo.git"]);
        git(&second, &["remote", "add", "origin", "projects/repo.git"]);

        let first = RepositoryIdentity::discover(&first).expect("first identity");
        let second = RepositoryIdentity::discover(&second).expect("second identity");
        assert_ne!(first.repository(), second.repository());
        assert_eq!(Path::new(first.repository()), first.toplevel());
        assert_eq!(Path::new(second.repository()), second.toplevel());
    }

    #[test]
    fn git_parses_inline_comments_and_included_origins() {
        let temp = tempfile::tempdir().expect("temp");
        let with_comment = temp.path().join("comment");
        init_repo(&with_comment);
        let config = with_comment.join(".git/config");
        let mut contents = std::fs::read_to_string(&config).expect("config");
        contents
            .push_str("\n[remote \"origin\"]\n url = https://github.com/team/repo.git # primary\n");
        std::fs::write(&config, contents).expect("config with comment");
        assert_eq!(
            RepositoryIdentity::discover(&with_comment)
                .expect("identity")
                .repository(),
            "github.com/team/repo"
        );

        let with_include = temp.path().join("include");
        init_repo(&with_include);
        let included_config = temp.path().join("origin.conf");
        std::fs::write(
            &included_config,
            "[remote \"origin\"]\n url = https://github.com/team/repo.git\n",
        )
        .expect("included config");
        git(
            &with_include,
            &[
                "config",
                "--local",
                "include.path",
                included_config.to_str().expect("utf8 path"),
            ],
        );
        assert_eq!(
            RepositoryIdentity::discover(&with_include)
                .expect("identity")
                .repository(),
            "github.com/team/repo"
        );
    }

    #[test]
    fn git_applies_origin_url_rewrites() {
        let temp = tempfile::tempdir().expect("temp");
        init_repo(temp.path());
        git(
            temp.path(),
            &[
                "config",
                "--local",
                "url.https://github.com/.insteadOf",
                "gh:",
            ],
        );
        git(
            temp.path(),
            &["remote", "add", "origin", "gh:team/repo.git"],
        );

        assert_eq!(
            RepositoryIdentity::discover(temp.path())
                .expect("identity")
                .repository(),
            "github.com/team/repo"
        );
    }

    #[test]
    fn worktrees_of_one_origin_share_a_name_and_clones_without_remotes_do_not() {
        let temp = tempfile::tempdir().expect("temp");
        let main = temp.path().join("main");
        init_repo(&main);
        git(
            &main,
            &[
                "remote",
                "add",
                "origin",
                "git@github.com:example/project.git",
            ],
        );
        git(
            &main,
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "--allow-empty",
                "-m",
                "base",
            ],
        );
        let worktree = temp.path().join("wt");
        git(
            &main,
            &[
                "worktree",
                "add",
                "--detach",
                worktree.to_str().expect("utf8 path"),
            ],
        );
        std::fs::create_dir_all(worktree.join("nested")).expect("worktree subdir");

        let from_main = RepositoryIdentity::discover(&main).expect("main identity");
        let from_worktree =
            RepositoryIdentity::discover(&worktree.join("nested")).expect("worktree identity");
        assert_eq!(from_main.repository(), "github.com/example/project");
        assert_eq!(from_worktree.repository(), from_main.repository());
        assert_eq!(
            from_worktree.toplevel(),
            std::fs::canonicalize(&worktree)
                .expect("canonical")
                .as_path()
        );

        git(&main, &["remote", "remove", "origin"]);
        git(&main, &["config", "extensions.worktreeConfig", "true"]);
        git(
            &worktree,
            &[
                "config",
                "--worktree",
                "remote.origin.url",
                "https://github.com/example/other.git",
            ],
        );
        assert_eq!(
            RepositoryIdentity::discover(&worktree)
                .expect("worktree override")
                .repository(),
            "github.com/example/other"
        );

        let local = temp.path().join("local");
        init_repo(&local);
        let local_identity = RepositoryIdentity::discover(&local).expect("local identity");
        assert_ne!(local_identity.repository(), from_main.repository());
        assert_eq!(
            Path::new(local_identity.repository()),
            std::fs::canonicalize(&local).expect("canonical")
        );
        assert_matches!(RepositoryIdentity::discover(temp.path()), None);
    }

    #[test]
    fn jujutsu_workspaces_of_one_origin_share_a_name() {
        let temp = tempfile::tempdir().expect("temp");
        let store = temp
            .path()
            .join("main")
            .join(".jj")
            .join("repo")
            .join("store");
        std::fs::create_dir_all(&store).expect("store");
        std::fs::create_dir_all(store.join("git")).expect("git dir");
        git(&store.join("git"), &["init", "--bare", "-q"]);
        std::fs::write(store.join("git_target"), "git").expect("target");
        git(
            &store.join("git"),
            &[
                "remote",
                "add",
                "origin",
                "git@github.com:example/project.git",
            ],
        );
        let second = temp.path().join("second").join(".jj");
        std::fs::create_dir_all(&second).expect("workspace");
        std::fs::write(second.join("repo"), "../../main/.jj/repo").expect("pointer");

        let main = RepositoryIdentity::discover(&temp.path().join("main")).expect("main");
        let other = RepositoryIdentity::discover(&temp.path().join("second")).expect("second");
        assert_eq!(main.repository(), "github.com/example/project");
        assert_eq!(other.repository(), main.repository());
    }

    #[test]
    fn relative_paths_are_toplevel_relative_and_bounded() {
        let temp = tempfile::tempdir().expect("temp");
        let root = temp.path().join("repo");
        init_repo(&root);
        std::fs::create_dir_all(root.join("src")).expect("src");
        let identity = RepositoryIdentity::discover(&root.join("src")).expect("identity");
        let workspace = identity.toplevel().join("src");

        assert_eq!(
            identity.relative(Path::new("memory/mod.rs"), &workspace),
            Some("src/memory/mod.rs".to_string()),
            "relative input resolves against the workspace, whether or not it exists"
        );
        assert_eq!(
            identity.relative(Path::new("./memory/../lib.rs"), &workspace),
            Some("src/lib.rs".to_string())
        );
        assert_eq!(
            identity.relative(&identity.toplevel().join("README.md"), &workspace),
            Some("README.md".to_string())
        );
        assert_matches!(
            identity.relative(Path::new("../../outside.rs"), &workspace),
            None
        );
        assert_matches!(identity.relative(Path::new("/etc/hosts"), &workspace), None);
        assert_matches!(identity.relative(identity.toplevel(), &workspace), None);
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_relative_paths_do_not_collapse_to_one_glob_scope() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let temp = tempfile::tempdir().expect("temp");
        init_repo(temp.path());
        let identity = RepositoryIdentity::discover(temp.path()).expect("identity");
        for filename in [vec![b'a', 0xff], vec![b'a', 0xfe]] {
            let path = identity.toplevel().join(OsString::from_vec(filename));
            assert_matches!(identity.relative(&path, identity.toplevel()), None);
        }
    }
}
