# Security Policy

We welcome security reports for Mirror, including its CLI, agent, and bundled
components. Security reports are welcome even though we are not currently
accepting external code or documentation contributions.

## Supported versions

Security fixes target the latest released version of Mirror. Older releases
are not guaranteed to receive backports. Please keep Mirror updated.

## Reporting a vulnerability

Report suspected vulnerabilities privately by emailing
[developers@reflection.ai](mailto:developers@reflection.ai).

Do not use `/feedback` or public GitHub issues to report vulnerabilities.

Include:

- Mirror version (`mirror --version`), commit if built from source, and operating
  system.
- A description of the vulnerability and its potential impact.
- Steps to reproduce it, with a minimal proof of concept if possible.
- Relevant configuration and error messages.
- Any known mitigations.

For dependency reports, identify the shipped version and explain how the
vulnerable behavior can be reached through Mirror.

Remove API keys, access tokens, private source code, and personal information
from any logs or attachments.

Please give us an opportunity to investigate and address the issue before
publishing vulnerability details. We will coordinate disclosure with you
by email.

## Security boundaries

Mirror runs tools locally with your account's permissions. It is not a sandbox.

The working directory does not limit which files or network services tools can
access. Manual approval controls tool execution; it does not provide filesystem
or network isolation.

Extensions and local MCP servers execute code with your account's permissions.
Only enable code and services you trust. Review project-provided extensions
before granting consent; `--trust-extensions` enables new extensions without
asking for that consent.

Repository instructions, file contents, tool output, and model responses can
contain prompt injections that influence the agent's actions. Review commands
and changes, and use version control or backups for recovery. For stronger
isolation, run Mirror and its child processes inside a container or virtual
machine, exposing only the files, credentials, and network access the task needs.

Model requests send conversation context, including relevant file contents and
tool results, to the selected provider. Connected MCP servers and web services
receive their tool requests. Conversation history and logs may contain
sensitive information. See the [README](README.md#local-data-and-safety) for
details.

## Report scope

Reports of vulnerabilities in Mirror's distributed code and bundled components
are in scope. Examples include bypasses of manual approval or extension consent,
unintended credential disclosure caused by Mirror, and access beyond the
permissions of the account running it.

The following do not, by themselves, demonstrate a vulnerability in Mirror:

- Tools reading, writing, or executing files with the running user's permissions.
- Malicious behavior in extensions or MCP servers the user has chosen to trust.
- Malicious model output or prompt injection without a demonstrated bypass of
  a security control.
- Attacks that require prior write access to the user's configuration, workspace,
  or shell environment, unless the report shows how Mirror grants that access or
  crosses a privilege boundary.

If a report involves one of these conditions, explain the security control that
was bypassed and the resulting impact. If you are unsure whether an issue is in
scope, contact us privately.
