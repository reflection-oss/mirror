#![forbid(unsafe_code)]

use std::error::Error;
use std::path::PathBuf;

use mirror_sandbox::SANDBOXD_WORKSPACE_VARIABLE;
use mirror_sandboxd::prepare_repository_stdio;
use mirror_sandboxd::run_once_stdio;
use mirror_sandboxd::serve;
use mirror_sandboxd::RunnerCommand;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let workspace = match std::env::var_os(SANDBOXD_WORKSPACE_VARIABLE) {
        Some(workspace) => PathBuf::from(workspace),
        None => std::env::current_dir()?,
    };
    match std::env::args().nth(1).as_deref() {
        None | Some("serve") => {
            if let Some(workspace) = std::env::args_os().nth(2) {
                std::env::set_var(SANDBOXD_WORKSPACE_VARIABLE, workspace);
            }
            let runner = RunnerCommand {
                program: std::env::current_exe()?,
                args: vec!["run-once".to_string()],
            };
            serve(tokio::io::stdin(), tokio::io::stdout(), runner).await?;
        }
        Some("run-once") => run_once_stdio(&workspace).await?,
        Some("context") => mirror_sandboxd::print_workspace_context(&workspace)?,
        Some("prepare-repository") => {
            let skills_only = match std::env::args().nth(2).as_deref() {
                None => false,
                Some("--skills-only") => true,
                Some(argument) => {
                    return Err(format!("unsupported preparation argument {argument:?}").into())
                }
            };
            prepare_repository_stdio(&workspace, skills_only).await?;
        }
        Some(command) => {
            return Err(format!("unsupported mirror-sandboxd command {command:?}").into());
        }
    }
    Ok(())
}
