//! `PINNED_MEMORIES.md`: the memories that are always in the prompt, one
//! bullet each, written by the memory tool in chat and editable by hand.

use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use super::invalid;
use super::reject_credentials;
use super::MemoryResult;
use crate::system_prompt::MAX_PINNED_MEMORIES_BYTES;

/// The bullets currently pinned, without their `- ` markers.
pub fn pinned_bullets(path: &Path) -> MemoryResult<Vec<String>> {
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    Ok(content
        .lines()
        .filter_map(|line| line.trim_start().strip_prefix("- "))
        .map(|bullet| bullet.trim().to_string())
        .filter(|bullet| !bullet.is_empty())
        .collect())
}

/// Add `text` as a bullet. Refused when it looks like a credential, and,
/// with the current bullets in the error, when the file would exceed its
/// cap: pinning forces tidying, never truncation.
pub fn pin(path: &Path, text: &str) -> MemoryResult<()> {
    reject_credentials(&[text])?;
    let text = single_line(text);
    let mut bullets = pinned_bullets(path)?;
    if bullets.contains(&text) {
        return Ok(());
    }
    bullets.push(text);
    let content = render(&bullets);
    if content.len() > MAX_PINNED_MEMORIES_BYTES {
        return Err(invalid(format!(
            "the pinned memories are full ({MAX_PINNED_MEMORIES_BYTES} bytes); unpin or shorten one first. Currently pinned:\n{}",
            render(&bullets[..bullets.len() - 1])
        )));
    }
    write(path, &content)
}

/// Remove the bullet that reads `text`.
pub fn unpin(path: &Path, text: &str) -> MemoryResult<()> {
    let text = single_line(text);
    let bullets = pinned_bullets(path)?;
    let kept: Vec<String> = bullets
        .iter()
        .filter(|bullet| **bullet != text)
        .cloned()
        .collect();
    if kept.len() == bullets.len() {
        return Err(invalid(format!(
            "no pinned memory reads {text:?}. Currently pinned:\n{}",
            render(&bullets)
        )));
    }
    write(path, &render(&kept))
}

fn single_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn render(bullets: &[String]) -> String {
    bullets
        .iter()
        .map(|bullet| format!("- {bullet}\n"))
        .collect()
}

/// Replace the file atomically through an owner-only temporary file, so the
/// bullets are never world-readable, not even while being written.
fn write(path: &Path, content: &str) -> MemoryResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("md.tmp");
    match std::fs::remove_file(&temporary) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error.into()),
        _ => {}
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temporary)?;
    file.write_all(content.as_bytes())?;
    std::fs::rename(&temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::memory::MemoryError;

    #[test]
    fn pins_and_unpins_bullets_and_refuses_past_the_cap() {
        let directory = tempfile::tempdir().expect("directory");
        let path = directory.path().join("PINNED_MEMORIES.md");
        assert_eq!(pinned_bullets(&path).expect("empty"), Vec::<String>::new());

        pin(&path, "Prefer bazelisk\nover bazel").expect("pin");
        pin(&path, "Reply tersely").expect("pin");
        pin(&path, "Reply tersely").expect("pinning twice is a no-op");
        assert_eq!(
            std::fs::read_to_string(&path).expect("file"),
            "- Prefer bazelisk over bazel\n- Reply tersely\n"
        );

        let oversized = "x".repeat(MAX_PINNED_MEMORIES_BYTES);
        assert_matches!(
            pin(&path, &oversized),
            Err(MemoryError::Invalid(message)) if message.contains("Reply tersely")
        );
        assert_eq!(pinned_bullets(&path).expect("unchanged").len(), 2);

        unpin(&path, "Reply tersely").expect("unpin");
        assert_eq!(
            pinned_bullets(&path).expect("one left"),
            vec!["Prefer bazelisk over bazel"]
        );
        assert_matches!(unpin(&path, "Reply tersely"), Err(MemoryError::Invalid(_)));
    }

    #[test]
    fn refuses_to_pin_a_credential() {
        let directory = tempfile::tempdir().expect("directory");
        let path = directory.path().join("PINNED_MEMORIES.md");
        pin(&path, "Reply tersely").expect("pin");

        let leaked = format!("token {}_{}", "ghp", "a".repeat(36));
        assert_matches!(pin(&path, &leaked), Err(MemoryError::Credential));
        assert_eq!(
            std::fs::read_to_string(&path).expect("file"),
            "- Reply tersely\n"
        );
    }

    #[cfg(unix)]
    #[test]
    fn the_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("directory");
        let path = directory.path().join("PINNED_MEMORIES.md");
        pin(&path, "Reply tersely").expect("pin");
        let mode = std::fs::metadata(&path)
            .expect("metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600);
    }
}
