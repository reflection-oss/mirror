//! Host-local model management, isolated from conversation workers.

mod capabilities;
mod catalog;
mod downloads;
mod endpoint;
mod engine;
mod executable;
mod gguf;
mod guard;
mod host;
mod host_state;
mod local_model_error;
mod reads;
mod restart_policy;
mod runtime;
mod service;
mod settings;

use catalog::{Catalog, CatalogError};
use downloads::{DownloadError, Downloads};
pub(crate) use endpoint::resolve_endpoint;
pub use executable::executable;
pub(crate) use gguf::is_drafter;
pub use guard::run_guard;
pub use host::{Host, CONTROL_PORT};
pub use local_model_error::LocalModelError;
pub use service::Service;
