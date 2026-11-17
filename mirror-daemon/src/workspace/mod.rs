//! Workspace implementations and native execution; WorkspaceRegistry owns their lifetime.

mod checkout;
mod directory;
mod discovery;
mod fork;
mod local;
mod operations;
mod process;
mod remote;
mod repository;
mod sandbox;
mod sandbox_runtime;
mod shell;
mod update;
#[allow(clippy::module_inception)] // The trait lives in its entity-named file, not mod.rs.
mod workspace;
mod workspace_record;
mod workspace_store;

pub use checkout::*;
pub use directory::*;
pub use discovery::*;
pub use fork::*;
pub use local::*;
pub use operations::*;
pub use process::*;
pub use remote::*;
pub use repository::*;
pub use sandbox::*;
pub use sandbox_runtime::*;
pub use shell::*;
pub use update::*;
pub use workspace::*;
pub use workspace_record::*;
pub use workspace_store::*;
