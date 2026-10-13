# mirror-sandboxd

`sandboxd` is the tool worker inside a Mirror sandbox. The hosted server calls
its generated gRPC service through the provider's connection. The Kubernetes
adapter starts `mirror-sandboxd serve` with `pods/exec`; HTTP/2 runs over its
stdin/stdout streams. The provider needs no guest listening port. The connector
consumes one connection and never reconnects or replays an execution.

`sandboxd.proto`, in `mirror-sandbox` with the host-side session, imports the
existing `ToolCall` and `ChatRecord` contracts; that crate generates the tonic
clients and services. The RPC message limit is 64 MiB.
There is no application version counter or custom frame codec.

- `Execute` streams admission followed by the final outcome. Admission confirms
  that the call ID is registered, so `Cancel` cannot overtake an uploading call.
  Rejected calls return their outcome without starting a process.
- `Cancel` acknowledges the cancellation request. The execution's final outcome
  reports whether the runner was reaped and its process group observed gone.
  A process that escaped the group is not claimed stopped.
- Dropping an execution RPC cancels that run. Losing the connection stops all
  remaining runs. Cleanup belongs to the execution runtime and survives the
  RPC handler disappearing.

The package separates the gRPC adapter (`sandboxd_service.rs`) from process
ownership (`execution_runtime.rs`). It serves over the provider's byte stream
with `mirror-sandbox`, which hosts also use to connect.
The runtime runs each call in its own process group, limits concurrent work,
retains call IDs until cleanup completes, and bounds shutdown. Each child runs
`run-once`, which consumes one protobuf tool call on stdin and emits one record
on stdout. When the workspace is shared from a host, as with microsandbox's
virtiofs mounts, the provider sets `MIRROR_REFRESH_FILE_CACHE=1` and `run-once`
first drops the guest's dentry and inode caches, so the call sees host edits
made just before it. Cancellation sends SIGTERM, then SIGKILL if needed, reaps
the worker, and checks the process group.

`mirror-sandboxd prepare-repository` is the server's session-start entrypoint.
The server runs it over the provider's exec stream with a protobuf
`RepositoryDefinition` on stdin, sized by `MIRROR_TOOL_REQUEST_BYTES` like a
`run-once` request, and reads a JSON `ResolvedInstructions` from stdout. It
materializes the session checkout under the workspace's `repository/` child and
loads bounded instruction and skill context through the shared harness loader.
Hosted Mirror's server documents how it uses this entrypoint in its repository
preparation notes.

`sandboxd` holds no durable state or provider credentials. It holds a short-lived
Mirror execution bearer in memory when egress is configured. It provides
process isolation under the sandbox's Unix identity; whole-sandbox deletion is
the stronger boundary. The sandbox image installs the binary at
`/usr/local/bin/mirror-sandboxd`. `MIRROR_WORKSPACE` selects the tool workspace.

The optional `Egress` RPC configures a stream-owned local CONNECT proxy and
renews Mirror identity through the trusted server. `egress_runtime.rs` owns
renewal sequencing and expiry; `egress_connector.rs` owns the bounded local
listener and verified outer TLS relay. Child environments contain a fixed local
proxy URL and public CA path, never the execution token. Refresh needs no child
restart. Existing tunnels reject new requests after their original identity expires;
already admitted transfers may finish, and interrupted operations are not replayed. Renewal failure closes the connector, and a new
provider control stream is required to configure it again. Hosted Mirror's
egress contract covers authority and deployment requirements.
