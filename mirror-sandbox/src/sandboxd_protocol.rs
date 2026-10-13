//! The gRPC protocol between a host and the `sandboxd` inside a sandbox.

use std::time::Duration;

include!(concat!(env!("OUT_DIR"), "/mirror.sandbox.v1.rs"));
pub use execute_response::Outcome as CallOutcome;
pub const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_DISPATCH_CALL_ID_BYTES: usize = 4 * 1024;

pub const MAX_EGRESS_IDENTITY_BYTES: usize = 2048;

/// How long `sandboxd` waits for a cancelled call's process group to exit
/// before killing it; hosts size their wait for a cancel's answer from it.
pub const STOP_GRACE: Duration = Duration::from_secs(5);
