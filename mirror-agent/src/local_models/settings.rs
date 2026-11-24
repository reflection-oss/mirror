//! User-editable local model settings kept beside the cached models.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The contents of `settings.json` in the local model directory.
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Settings {
    pub context_tokens: Option<u64>,
    /// Seconds of idleness before llama.cpp unloads the model; unset or zero means never.
    pub sleep_idle_seconds: Option<u64>,
    /// Where llama.cpp keeps downloaded models; unset means llama.cpp's own default.
    pub models_dir: Option<PathBuf>,
    /// A `--models-preset` file the user already keeps; unset means Mirror's own `presets.ini`.
    pub models_preset: Option<PathBuf>,
}

#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    #[error("Could not read {}: {source}", .path.display())]
    Unreadable {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{} is not valid settings: {source}", .path.display())]
    Malformed {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("Could not write {}: {source}", .path.display())]
    Unwritable {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{setting} in settings.json must be an absolute path, not {}", .path.display())]
    RelativePath {
        setting: &'static str,
        path: PathBuf,
    },
    #[error("Cannot find the model cache; set models_dir in settings.json")]
    NoModelCache,
    #[error("{} must be UTF-8 text", .0.display())]
    PresetNotText(PathBuf),
    #[error("{} is larger than 64 KiB", .0.display())]
    PresetTooLarge(PathBuf),
}

/// Per-model llama-server settings in the format of llama.cpp's `--models-preset`.
///
/// One `[repo:QUANT]` section per model, or `[*]` for all, with flag names minus the dashes.
const PRESET_FILE: &str = "presets.ini";

/// The largest preset file llama.cpp is given.
const PRESET_LIMIT: usize = 64 * 1024;

impl Settings {
    /// Read the settings; before the first save they are all unset.
    pub fn load(root: &Path) -> Result<Self, SettingsError> {
        let path = root.join("settings.json");
        match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|source| SettingsError::Malformed { path, source }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(source) => Err(SettingsError::Unreadable { path, source }),
        }
    }

    /// The folder holding downloaded models: the user's setting, else the cache llama.cpp
    /// itself uses, so models fetched by llama.cpp or other tools are found here too.
    pub fn models_dir(&self) -> Result<PathBuf, SettingsError> {
        match &self.models_dir {
            Some(directory) if directory.is_absolute() => Ok(directory.clone()),
            Some(directory) => Err(SettingsError::RelativePath {
                setting: "models_dir",
                path: directory.clone(),
            }),
            None => default_models_dir(|name| std::env::var_os(name), passwd_home),
        }
    }

    /// The preset file passed to llama.cpp: the user's setting, else Mirror's own.
    pub fn preset_path(&self, root: &Path) -> Result<PathBuf, SettingsError> {
        match &self.models_preset {
            Some(path) if path.is_absolute() => Ok(path.clone()),
            Some(path) => Err(SettingsError::RelativePath {
                setting: "models_preset",
                path: path.clone(),
            }),
            None => Ok(root.join(PRESET_FILE)),
        }
    }

    /// Replace the context while keeping the user's other settings.
    pub fn save_context(root: &Path, context: u64) -> Result<(), SettingsError> {
        Self::update(root, |settings| settings.context_tokens = Some(context))
    }

    /// Replace the idle-unload delay while keeping the user's other settings.
    pub fn save_sleep_idle(root: &Path, seconds: u64) -> Result<(), SettingsError> {
        Self::update(root, |settings| settings.sleep_idle_seconds = Some(seconds))
    }

    /// A malformed file is replaced, which is how it is repaired; one that cannot be read is not.
    fn update(root: &Path, change: impl FnOnce(&mut Self)) -> Result<(), SettingsError> {
        let mut settings = match Self::load(root) {
            Ok(settings) => settings,
            Err(SettingsError::Malformed { .. }) => Self::default(),
            Err(error) => return Err(error),
        };
        change(&mut settings);
        let path = root.join("settings.json");
        let unwritable = |source| SettingsError::Unwritable {
            path: path.clone(),
            source,
        };
        let mut temporary = tempfile::NamedTempFile::new_in(root).map_err(unwritable)?;
        serde_json::to_writer(&mut temporary, &settings)
            .map_err(|error| unwritable(error.into()))?;
        temporary
            .persist(&path)
            .map_err(|error| unwritable(error.error))?;
        Ok(())
    }
}

/// The preset file to pass to llama.cpp, or `None` when there is none or it only has comments.
///
/// Whether the file is well formed, and which keys it may set, is llama.cpp's call: Mirror's
/// own command-line flags take precedence over presets, and the router names what it rejects.
pub fn load_preset(root: &Path) -> Result<Option<PathBuf>, SettingsError> {
    let path = Settings::load(root)?.preset_path(root)?;
    let text = match std::fs::read(&path) {
        Ok(bytes) if bytes.len() <= PRESET_LIMIT => {
            String::from_utf8(bytes).map_err(|_| SettingsError::PresetNotText(path.clone()))?
        }
        Ok(_) => return Err(SettingsError::PresetTooLarge(path)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(SettingsError::Unreadable { path, source }),
    };
    let defines_a_section = text.lines().any(|line| line.trim_start().starts_with('['));
    Ok(defines_a_section.then_some(path))
}

/// The user's home from the account database, for a daemon started without `HOME`.
fn passwd_home() -> Option<PathBuf> {
    nix::unistd::User::from_uid(nix::unistd::getuid())
        .ok()
        .flatten()
        .map(|user| user.dir)
}

/// llama.cpp's default model cache: `LLAMA_CACHE`, else the Hugging Face hub cache.
fn default_models_dir(
    environment: impl Fn(&str) -> Option<std::ffi::OsString>,
    account_home: impl Fn() -> Option<PathBuf>,
) -> Result<PathBuf, SettingsError> {
    let variable = |name| environment(name).filter(|value| !value.is_empty());
    if let Some(directory) = variable("LLAMA_CACHE").or_else(|| variable("HF_HUB_CACHE")) {
        return Ok(directory.into());
    }
    if let Some(home) = variable("HF_HOME") {
        return Ok(PathBuf::from(home).join("hub"));
    }
    if let Some(cache) = variable("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(cache).join("huggingface/hub"));
    }
    variable("HOME")
        .map(PathBuf::from)
        .or_else(account_home)
        .map(|home| home.join(".cache/huggingface/hub"))
        .ok_or(SettingsError::NoModelCache)
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    use super::*;

    fn environment(
        pairs: &'static [(&'static str, &'static str)],
    ) -> impl Fn(&str) -> Option<std::ffi::OsString> {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).into())
        }
    }

    #[test]
    fn models_default_to_the_cache_llama_cpp_itself_uses() {
        assert_matches!(
            default_models_dir(environment(&[("HOME", "/home/a")]), || None),
            Ok(path) if path == Path::new("/home/a/.cache/huggingface/hub")
        );
        assert_matches!(
            default_models_dir(
                environment(&[("HOME", "/home/a"), ("HF_HOME", "/data/hf")]),
                || None
            ),
            Ok(path) if path == Path::new("/data/hf/hub")
        );
        assert_matches!(
            default_models_dir(
                environment(&[("HOME", "/home/a"), ("LLAMA_CACHE", "/models")]),
                || None
            ),
            Ok(path) if path == Path::new("/models")
        );
    }

    #[test]
    fn a_daemon_without_home_falls_back_to_the_account_home() {
        assert_matches!(
            default_models_dir(environment(&[]), || Some("/Users/me".into())),
            Ok(path) if path == Path::new("/Users/me/.cache/huggingface/hub")
        );
    }

    fn settings_file(contents: &str) -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("settings directory");
        std::fs::write(root.path().join("settings.json"), contents).expect("settings");
        root
    }

    #[test]
    fn preset_and_models_settings_replace_the_defaults_and_must_be_absolute() {
        let set =
            settings_file(r#"{"models_preset": "/me/models.ini", "models_dir": "/data/gguf"}"#);
        let settings = Settings::load(set.path()).expect("settings");
        assert_matches!(
            settings.preset_path(set.path()),
            Ok(path) if path == Path::new("/me/models.ini")
        );
        assert_matches!(settings.models_dir(), Ok(path) if path == Path::new("/data/gguf"));

        let relative = settings_file(r#"{"models_preset": "models.ini", "models_dir": "gguf"}"#);
        let settings = Settings::load(relative.path()).expect("settings");
        assert_matches!(
            settings.preset_path(relative.path()),
            Err(SettingsError::RelativePath {
                setting: "models_preset",
                ..
            })
        );
        assert_matches!(
            settings.models_dir(),
            Err(SettingsError::RelativePath {
                setting: "models_dir",
                ..
            })
        );
    }

    #[test]
    fn saving_keeps_the_users_other_settings_and_repairs_a_malformed_file() {
        let root = settings_file(r#"{"context_tokens": 8192, "models_dir": "/data/gguf"}"#);
        Settings::save_context(root.path(), 32768).expect("save");
        let saved = Settings::load(root.path()).expect("load");
        assert_eq!(saved.context_tokens, Some(32768));
        assert_eq!(saved.models_dir, Some("/data/gguf".into()));

        std::fs::write(root.path().join("settings.json"), b"invalid").expect("corrupt");
        assert_matches!(
            Settings::load(root.path()),
            Err(SettingsError::Malformed { .. })
        );
        Settings::save_sleep_idle(root.path(), 900).expect("repair");
        assert_eq!(
            Settings::load(root.path())
                .expect("load")
                .sleep_idle_seconds,
            Some(900)
        );
    }

    #[test]
    fn a_settings_file_that_cannot_be_read_is_not_overwritten() {
        let root = tempfile::tempdir().expect("settings directory");
        std::fs::create_dir(root.path().join("settings.json")).expect("a directory in its place");

        assert_matches!(
            Settings::save_context(root.path(), 8192),
            Err(SettingsError::Unreadable { .. })
        );
    }

    #[test]
    fn a_preset_file_is_passed_only_when_it_defines_a_section() {
        let root = tempfile::tempdir().expect("settings directory");
        assert_matches!(load_preset(root.path()), Ok(None));

        std::fs::write(
            root.path().join("presets.ini"),
            "; examples only\n# [org/m:Q4]\n",
        )
        .expect("comments");
        assert_matches!(load_preset(root.path()), Ok(None));

        std::fs::write(root.path().join("presets.ini"), "[*]\nctx-size = 8192\n").expect("preset");
        assert_matches!(
            load_preset(root.path()),
            Ok(Some(path)) if path == root.path().join("presets.ini")
        );
    }
}
