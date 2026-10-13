#![forbid(unsafe_code)]

//! Microsandbox adapter for the Mirror sandbox contracts.
//!
//! `MicrosandboxProvider` runs each sandbox as a microVM through the `msb`
//! its user installs; Mirror does not ship microsandbox. The host workspace is
//! mounted at the same path in the guest, `sandboxd` is mounted read-only, and
//! the network is denied except HTTPS to an allowlist. Every use checks the
//! mounts and the network policy.

mod egress_host;
#[cfg(test)]
mod fake_msb;
mod microsandbox_config;
mod microsandbox_provider;
mod microsandbox_transport;
mod msb;
mod msb_exec_stream;
mod sandbox_inspection;

pub use egress_host::*;
#[cfg(test)]
pub(crate) use fake_msb::*;
pub use microsandbox_config::*;
pub use microsandbox_provider::*;
pub use msb::*;
pub(crate) use msb_exec_stream::*;
pub(crate) use sandbox_inspection::*;
