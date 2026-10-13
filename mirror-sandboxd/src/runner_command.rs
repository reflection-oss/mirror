//! The isolated worker executable and its fixed arguments.
use std::path::PathBuf;
#[derive(Clone, Debug)]
pub struct RunnerCommand {
    pub program: PathBuf,
    pub args: Vec<String>,
}
