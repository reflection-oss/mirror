#![forbid(unsafe_code)]

//! Provider-neutral sandbox contracts and the host side of `sandboxd`.
//!
//! `SandboxProvider` owns a sandbox's lifecycle and `SandboxTransport` opens
//! the stream to its `sandboxd` and streams bulk transfers through the
//! sandbox. Adapters implement both; callers such as the hosted server compose
//! them and never depend on a provider's own API. `SandboxdSession` runs tool
//! calls over that stream with the `sandboxd` gRPC protocol.

mod grpc_connection;
mod sandbox_provider;
mod sandbox_transport;
mod sandboxd_protocol;
mod sandboxd_session;
mod sandboxd_stream;

pub use grpc_connection::*;
pub use sandbox_provider::*;
pub use sandbox_transport::*;
pub use sandboxd_protocol::*;
pub use sandboxd_session::*;
pub use sandboxd_stream::*;
