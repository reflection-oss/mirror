//! Hermetic ripgrep subprocess execution.

use std::env;
use std::io;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use super::process_group::return_code;
use super::process_group::ProcessGroup;

const TIMEOUT: Duration = Duration::from_secs(30);
const EXCLUSIONS: &[&str] = &[
    "!.git/",
    "!.svn/",
    "!.hg/",
    "!.bzr/",
    "!.jj/",
    "!.sl/",
    "!.fossil/",
    "!_darcs/",
    "!__pycache__/",
    "!node_modules/.cache/",
    "!.cache/",
    "!.mypy_cache/",
    "!.ruff_cache/",
    "!.pytest_cache/",
];

pub(crate) struct RgResult {
    pub return_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
}

pub(crate) fn exclusion_arguments() -> impl Iterator<Item = String> {
    EXCLUSIONS
        .iter()
        .flat_map(|pattern| ["--glob".to_string(), (*pattern).to_string()])
}

pub(crate) async fn run(arguments: &[String], cwd: &Path) -> io::Result<RgResult> {
    run_with_optional_input(arguments, cwd, None).await
}

pub(crate) async fn run_with_input(arguments: &[String], input: &str) -> io::Result<RgResult> {
    run_with_optional_input(arguments, Path::new("."), Some(input)).await
}

async fn run_with_optional_input(
    arguments: &[String],
    cwd: &Path,
    input: Option<&str>,
) -> io::Result<RgResult> {
    let result = run_once(arguments, cwd, input).await?;
    let output = [result.stdout.as_str(), result.stderr.as_str()].concat();
    if !matches!(result.return_code, 0 | 1)
        && !result.timed_out
        && ["EAGAIN", "Resource temporarily unavailable"]
            .iter()
            .any(|pattern| output.contains(pattern))
    {
        let retry = [vec!["-j".to_string(), "1".to_string()], arguments.to_vec()].concat();
        return run_once(&retry, cwd, input).await;
    }
    Ok(result)
}

async fn executable() -> io::Result<(PathBuf, Option<tempfile::TempDir>)> {
    if let Some(path) = env::var_os("MIRROR_RG_PATH") {
        return Ok((tokio::fs::canonicalize(path).await?, None));
    }
    #[cfg(feature = "bundled-ripgrep")]
    {
        let directory = tokio::task::spawn_blocking(|| {
            use std::os::unix::fs::PermissionsExt;

            let directory = tempfile::Builder::new().prefix("mirror-rg-").tempdir()?;
            let binary = directory.path().join("rg");
            std::fs::write(&binary, include_bytes!(env!("MIRROR_BUNDLED_RG")))?;
            std::fs::write(
                directory.path().join("LICENSE-MIT"),
                include_str!("../../RIPGREP-LICENSE-MIT"),
            )?;
            std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o500))?;
            Ok::<_, io::Error>(directory)
        })
        .await??;
        Ok((directory.path().join("rg"), Some(directory)))
    }
    #[cfg(not(feature = "bundled-ripgrep"))]
    Ok(("rg".into(), None))
}

async fn run_once(arguments: &[String], cwd: &Path, input: Option<&str>) -> io::Result<RgResult> {
    // Keep the private extraction directory alive until the subprocess finishes.
    let (binary, _directory) = executable().await?;
    let mut std_command = std::process::Command::new(binary);
    std_command.env_remove("REFLECTION_API_KEY");
    std_command
        .arg("--no-config")
        .args(arguments)
        .current_dir(cwd)
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    std_command.process_group(0);
    let mut command = Command::from(std_command);
    command.kill_on_drop(true);
    let mut process_group = ProcessGroup::new(command.spawn()?);
    let child = process_group.child();
    let mut child_stdin = child.stdin.take();
    let stdout = child.stdout.take().expect("piped stdout is present");
    let stderr = child.stderr.take().expect("piped stderr is present");
    let mut stdout_output = Vec::new();
    let mut stderr_output = Vec::new();
    let execution = futures_util::future::try_join4(
        async {
            let status = child.wait().await?;
            Ok::<_, io::Error>(return_code(status))
        },
        read_all(stdout, &mut stdout_output),
        read_all(stderr, &mut stderr_output),
        async {
            if let Some(input) = input {
                let mut stdin = child_stdin
                    .take()
                    .expect("piped stdin is present when input is provided");
                if let Err(error) = stdin.write_all(input.as_bytes()).await {
                    if error.kind() != io::ErrorKind::BrokenPipe {
                        return Err(error);
                    }
                }
                if let Err(error) = stdin.shutdown().await {
                    if error.kind() != io::ErrorKind::BrokenPipe {
                        return Err(error);
                    }
                }
            }
            Ok::<_, io::Error>(())
        },
    );
    let (return_code, timed_out) = match tokio::time::timeout(TIMEOUT, execution).await {
        Ok(result) => {
            let (return_code, (), (), ()) = result?;
            process_group.disarm();
            (return_code, false)
        }
        Err(_) => {
            let _ = process_group.terminate().await;
            (124, true)
        }
    };
    Ok(RgResult {
        return_code,
        stdout: String::from_utf8_lossy(&stdout_output).into_owned(),
        stderr: String::from_utf8_lossy(&stderr_output).into_owned(),
        timed_out,
    })
}

async fn read_all(
    mut reader: impl tokio::io::AsyncRead + Unpin,
    output: &mut Vec<u8>,
) -> io::Result<()> {
    reader.read_to_end(output).await?;
    Ok(())
}
