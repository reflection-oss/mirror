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

The crate is split by job. `connection/` serves one provider stream: the gRPC
adapter (`sandboxd_service.rs`), process ownership (`execution_runtime.rs`) and
egress setup (`egress_setup.rs`). `egress/` owns the local proxy and identity
renewal, `guest/` applies guest profiles, `tool_runner/` is the `run-once`
worker and `repository/` is `prepare-repository`. sandboxd serves over the
provider's byte stream with `mirror-sandbox`, which hosts also use to connect.
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
A new checkout is published by renaming its staging directory into place.
Where the filesystem has no rename that refuses to replace, as under gVisor,
preparation checks that the destination is absent and then renames. A new
checkout also sets `core.trustctime=false`: earlier guests published by hard
link, which moved every file's change time, and Git would otherwise re-read
the whole tree on each status. The environment snapshot of a checkout that
preparation just made reports a clean status without running `git status`,
since nothing else is in it: a forced checkout and a clean leave nothing, and
a merge over a named seed starts from a tree its caller vouches for. A caller
may name, in `MIRROR_DISPOSABLE_PATHS`, workspace-relative
paths that hold nothing but what the sandbox image recreates, one per line. When
a repository seed sits at one of them, preparation consumes it: the seed leaves
the workspace for the preparation's staging directory in one rename, the
entries its commit tracks become the checkout's, and Git merges the requested
revision over them, writing only the paths that differ between the two
commits. The rest of the seed is deleted with the staging directory. However a
preparation ends, the workspace holds a complete seed or none. The guest takes
the caller's word that the seed is as its image made it and does not read the
files the two commits share. Naming a seed also says its image put nothing
untracked below the seed's top level: such a file would move into the checkout
with its directory. When Git refuses the merge, a forced checkout and
a clean give the tree that a checkout into an empty directory would. A seed
that is not named is only read.
Every checkout fetches the remote's default branch with the requested revision
and reads `.mirror/setup` and `.mirror/resume` from the default branch, never
from the requested revision, keeping copies in the Git directory. A remote
whose `HEAD` does not resolve is checked out at the requested revision with no
scripts. When the
default branch has a setup script, the guest checks out the default branch at
the final path, runs setup there, then forces a checkout of the requested
revision over what setup left; a preparation that stops in between leaves a
marker, and the next one discards that checkout and starts again. Resume runs
after the checkout and again on the first preparation after a wake, which the
guest detects by a marker in the temporary directory, since a restarted sandbox
loses it and a workspace archive does not hold it. Both scripts run as the
guest user from the repository root with `MIRROR_SANDBOX=1`, in their own
process group, with output in `~/.cache/mirror/setup.log` and `resume.log`.
Setup has 20 minutes and resume one; a failure or timeout never fails the
preparation, and a new session's environment names the script and its log.
A preparation that receives SIGTERM kills the running script's process group
at once, so the script cannot outlive a cancelled preparation.
A checkout that either script ran in reports its status with `git status`,
since the scripts can leave files behind.
With `--skills-only` the entrypoint reuses the existing checkout,
loads skills without instructions, and reports the environment without a
repository status, so it never scans the tree. Hosted Mirror's server documents
how it uses this entrypoint in its repository preparation notes.

`sandboxd` holds no durable state or provider credentials. It holds a short-lived
Mirror execution bearer in memory when egress is configured. The serving
process is not dumpable, so other processes under the sandbox's Unix identity
cannot open its control stream or read its memory. An image that installs the
binary without read permission for that identity gets the same protection from
the moment the process starts, on a runtime that can start such a binary: Linux
can, and so can gVisor builds from October 2026. Builds from July 2026 and
earlier refuse to load it. It provides
process isolation under the sandbox's Unix identity; whole-sandbox deletion is
the stronger boundary. The sandbox image installs the binary at
`/usr/local/bin/mirror-sandboxd`. `MIRROR_WORKSPACE` selects the tool workspace;
`serve` also takes it as an argument, so a provider can start sandboxd without a
shell.

The optional `Egress` RPC configures a stream-owned local CONNECT proxy and
renews Mirror identity through the trusted server. `egress/egress_runtime.rs`
owns renewal sequencing and expiry; `egress/egress_connector.rs` owns the bounded local
listener and verified outer TLS relay. The local proxy listens on the loopback
port the setup names, retrying for two seconds while a replaced process lets
go of it, so processes that outlive a reconnect keep a working proxy. Child
environments contain that proxy URL for every proxy variable and Java's
`JAVA_TOOL_OPTIONS`, never the execution token, and `NO_PROXY` keeps loopback
off the proxy. Refresh needs no
child restart. Existing tunnels reject new requests after their original identity expires;
already admitted transfers may finish, and interrupted operations are not replayed. Renewal failure closes the connector, and a new
provider control stream is required to configure it again. Hosted Mirror's
egress contract covers authority and deployment requirements.

## Guest profiles

The `Egress` setup may carry guest profiles: named, data-only sets of
environment variables, files and static HTTP endpoints built by the host.
sandboxd knows no providers or clouds. A profile applies in two phases:

- `ready` runs once per connection, after setup and before any `Execute`. It
  writes the files (absolute paths, atomic replace, failing on any symbolic link
  in the path) and starts each endpoint on `127.0.0.1` at the port given. The
  first `Egress` event always reports the outcome: ready, or failed with the
  reason.
- `before_command` sets the environment variables on every `Execute` call.

A static endpoint matches routes by exact path, ignoring the query string. A
request missing a route's required header gets 403 and an unknown path 404;
endpoint response headers go on every response. Requests are served with
bounded concurrency, and the listener closes with the connection.

Until `ready` succeeds, `Execute` is rejected with the failure reason. Profiles
are refused if two profiles set one variable, two files name one path (compared
after normalizing), a path is relative, an endpoint listens anywhere but
`127.0.0.1`, or a variable is one sandboxd sets itself. A new connection is a
new process, so a reconnect applies the profiles again.

`mirror_sandbox::egress_profile` builds the profile every egress setup needs:
the proxy CA at `/tmp/mirror-egress-ca.pem`, the CA variables tools read, and
the host's placeholder variables set to `mirror-managed`.
