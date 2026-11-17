"""Sandbox choices rendered by the shared pending-workspace view."""

from mirror.cli import selection, workspaces
from mirror.cli.sandbox import setup as sandbox_setup


def choices(
  workspace: workspaces.SandboxWorkspace,
) -> tuple[selection.Choice, ...]:
  """Offer approvals or an explicit retry; running operations only allow cancellation."""
  if workspace.launching:
    return ()
  if workspace.error:
    choice = selection.Choice(workspace.setup_action + '-sandbox', 'Retry')
  elif not workspace.options.runtime_ready():
    size = sum(
      artifact.size for artifact in sandbox_setup.runtime_artifacts()
    ) // (1024 * 1024)
    choice = selection.Choice(
      'install-sandbox',
      'Approve and install Microsandbox 0.7.7',
      f'{size} MiB · Private to Mirror; no sudo or PATH changes',
    )
  elif not workspace.options.image_ready():
    image, size = sandbox_setup.default_image()
    detail = (
      f'Python, Git and search tools · about {size // (1024 * 1024) + 3} MiB'
      if workspace.options.image == image
      else workspace.options.image
    )
    choice = selection.Choice(
      'download-sandbox', 'Approve and download the sandbox environment', detail
    )
  else:
    options = [selection.Choice('start-sandbox', 'Empty workspace')]
    if workspace.source is not None:
      options.append(
        selection.Choice(
          'copy-sandbox',
          'Copy current repository',
          f'{workspace.source.label} · committed revision only; no uncommitted changes',
        )
      )
    return (*options, selection.Choice('cancel-sandbox', 'Cancel'))
  return choice, selection.Choice('cancel-sandbox', 'Cancel')
