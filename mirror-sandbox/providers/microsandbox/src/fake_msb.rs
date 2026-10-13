//! A fake `msb` for tests, backed by `testdata/fake_msb.sh`.

use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;

use serde_json::Value;
use tempfile::TempDir;

use crate::Msb;

/// A directory holding the fake `msb` and the sandboxes it pretends to run.
pub(crate) struct FakeMsb {
    state: TempDir,
}

const SCRIPT: &str = include_str!("../testdata/fake_msb.sh");

/// A file that exists on every test host, standing in for `sandboxd` where a
/// mount's host path must be a file.
pub fn sandboxd_file() -> String {
    std::env::current_exe()
        .expect("the test binary's path")
        .display()
        .to_string()
}

impl FakeMsb {
    pub fn new() -> Self {
        let state = tempfile::tempdir().expect("fake msb state directory");
        let program = state.path().join("msb");
        // Writing the script from a child keeps its write handle out of this
        // process, whose concurrent forks could otherwise make executing it
        // fail with "text file busy".
        let mut installer = std::process::Command::new("/bin/sh")
            .args(["-c", r#"cat >"$1" && chmod +x "$1""#, "install"])
            .arg(&program)
            .stdin(Stdio::piped())
            .spawn()
            .expect("start the fake msb installer");
        installer
            .stdin
            .take()
            .expect("the installer's stdin is piped")
            .write_all(SCRIPT.as_bytes())
            .expect("write the fake msb");
        let installed = installer.wait().expect("the installer exits");
        assert!(installed.success(), "installing the fake msb failed");
        Self { state }
    }

    pub fn program(&self) -> PathBuf {
        self.state.path().join("msb")
    }

    pub async fn msb(&self) -> Msb {
        Msb::at(self.program())
            .await
            .expect("the fake msb reports a supported version")
    }

    pub fn write(&self, file: &str, contents: &str) {
        std::fs::write(self.state.path().join(file), contents).expect("write fake msb state");
    }

    /// The inspection document `create` gives each new sandbox, with
    /// `"__STATUS__"` as its status.
    pub fn set_template(&self, inspection: &Value) {
        self.write("template.json", &inspection.to_string());
    }

    pub fn add_sandbox(&self, name: &str, status: &str, inspection: &Value) {
        self.write(&format!("{name}.json"), &inspection.to_string());
        self.set_status(name, status);
    }

    pub fn set_status(&self, name: &str, status: &str) {
        self.write(&format!("{name}.status"), status);
    }

    /// The sandbox's status, or `None` once it is removed.
    pub fn status(&self, name: &str) -> Option<String> {
        let status = std::fs::read_to_string(self.state.path().join(format!("{name}.status")));
        status.ok().map(|status| status.trim().to_string())
    }

    /// Each invocation's arguments, joined by spaces.
    pub fn calls(&self) -> Vec<String> {
        std::fs::read_to_string(self.state.path().join("calls"))
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }
}
