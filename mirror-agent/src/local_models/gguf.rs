//! Just enough of the GGUF header to tell what a cached file is.

use std::path::Path;

use tokio::io::AsyncReadExt;

/// Whether a file name marks a projector or drafter, for files whose header is out of reach.
pub fn is_sidecar_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    SIDECAR_WORDS.iter().any(|word| name.contains(word))
}

/// The header's `general.architecture`, when it sits within the first 256 KiB.
///
/// Writers put it first, before the large tokenizer tables, so reading further is not worth it.
pub async fn architecture(path: &Path) -> Option<String> {
    let mut header = Vec::new();
    tokio::fs::File::open(path)
        .await
        .ok()?
        .take(256 * 1024)
        .read_to_end(&mut header)
        .await
        .ok()?;
    parse_architecture(&header)
}

/// Whether the file at `path` is a drafter rather than a model.
///
/// An unreadable header falls back to the file name, which drafter repositories also mark.
pub async fn is_drafter(path: Option<&Path>, name: &str) -> bool {
    if let Some(architecture) = match path {
        Some(path) => architecture(path).await,
        None => None,
    } {
        return DRAFTER_ARCHITECTURES.contains(&architecture.as_str());
    }
    is_sidecar_name(name)
}

/// Architectures of speculative-decoding drafters, which are not models to chat with.
const DRAFTER_ARCHITECTURES: [&str; 3] = ["dflash", "dspark", "eagle3"];

/// File-name parts llama.cpp treats as auxiliary to a model (projector, speculative-decoding
/// drafters) rather than a model of their own.
const SIDECAR_WORDS: [&str; 6] = ["mmproj", "mtp", "draft", "eagle", "dflash", "dspark"];

fn parse_architecture(bytes: &[u8]) -> Option<String> {
    let mut reader = Reader { bytes, offset: 0 };
    if reader.take(4)? != b"GGUF" {
        return None;
    }
    reader.take(4)?; // version
    reader.take(8)?; // tensor count
    let entries = u64::from_le_bytes(reader.take(8)?.try_into().ok()?);
    for _ in 0..entries.min(4096) {
        let key = reader.string()?;
        let kind = u32::from_le_bytes(reader.take(4)?.try_into().ok()?);
        if key == "general.architecture" {
            return (kind == 8).then(|| reader.string()).flatten();
        }
        reader.skip_value(kind)?;
    }
    None
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        let end = self.offset.checked_add(count)?;
        let slice = self.bytes.get(self.offset..end)?;
        self.offset = end;
        Some(slice)
    }

    fn string(&mut self) -> Option<String> {
        let length = u64::from_le_bytes(self.take(8)?.try_into().ok()?);
        let bytes = self.take(usize::try_from(length).ok()?)?;
        String::from_utf8(bytes.to_vec()).ok()
    }

    fn skip_value(&mut self, kind: u32) -> Option<()> {
        match kind {
            0 | 1 | 7 => self.take(1).map(drop),
            2 | 3 => self.take(2).map(drop),
            4..=6 => self.take(4).map(drop),
            10..=12 => self.take(8).map(drop),
            8 => self.string().map(drop),
            9 => {
                let element = u32::from_le_bytes(self.take(4)?.try_into().ok()?);
                let length = u64::from_le_bytes(self.take(8)?.try_into().ok()?);
                for _ in 0..length {
                    self.skip_value(element)?;
                }
                Some(())
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;

    use super::*;

    fn text(value: &str) -> Vec<u8> {
        let mut bytes = (value.len() as u64).to_le_bytes().to_vec();
        bytes.extend(value.as_bytes());
        bytes
    }

    /// A header with a number and a string array before the architecture.
    fn header(architecture: &str) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(3u64.to_le_bytes());
        bytes.extend(text("general.file_type"));
        bytes.extend(4u32.to_le_bytes());
        bytes.extend(7u32.to_le_bytes());
        bytes.extend(text("tokenizer.tokens"));
        bytes.extend(9u32.to_le_bytes());
        bytes.extend(8u32.to_le_bytes());
        bytes.extend(2u64.to_le_bytes());
        bytes.extend(text("a"));
        bytes.extend(text("b"));
        bytes.extend(text("general.architecture"));
        bytes.extend(8u32.to_le_bytes());
        bytes.extend(text(architecture));
        bytes
    }

    #[test]
    fn the_architecture_is_found_after_other_entries() {
        assert_matches!(parse_architecture(&header("qwen3")), Some(architecture) if architecture == "qwen3");
    }

    #[test]
    fn anything_that_is_not_a_complete_gguf_header_yields_nothing() {
        assert_matches!(parse_architecture(b"not a model"), None);
        let truncated = header("dflash");
        assert_matches!(parse_architecture(&truncated[..truncated.len() - 4]), None);
    }

    #[tokio::test]
    async fn drafters_are_told_from_models_by_architecture_then_by_name() {
        let directory = tempfile::tempdir().expect("directory");
        let drafter = directory.path().join("a.gguf");
        let model = directory.path().join("b.gguf");
        std::fs::write(&drafter, header("eagle3")).expect("drafter");
        std::fs::write(&model, header("qwen3")).expect("model");

        assert!(is_drafter(Some(&drafter), "org/anything:Q4").await);
        assert!(!is_drafter(Some(&model), "org/model-dflash-lookalike:Q4").await);
        assert!(is_drafter(None, "org/Model-DFlash-GGUF:Q8_0").await);
        assert!(!is_drafter(None, "org/Model-GGUF:Q4_K_M").await);
    }
}
