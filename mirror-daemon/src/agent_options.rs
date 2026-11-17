//! Fixed daemon-wide paths; live provider inputs arrive only with attachments.

use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct AgentOptions {
    pub executable: PathBuf,
    pub bundle: PathBuf,
    pub database: PathBuf,
    pub home: PathBuf,
    pub runtime_directory: PathBuf,
    pub state_directory: PathBuf,
    pub path: String,
}
