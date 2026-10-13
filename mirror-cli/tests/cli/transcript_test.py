import pytest
from mirror.protocol import agent_pb2
from rich import text

from mirror.cli import actions, rendering, transcript


@pytest.fixture
def history() -> transcript.Transcript:
  return transcript.Transcript(
    view_mode=lambda: actions.ViewMode.TRANSCRIPT,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )


@pytest.mark.parametrize(
  ('entry', 'conversation'),
  [
    (agent_pb2.AgentMessage(), False),
    (agent_pb2.AgentMessage(log=agent_pb2.LogRecord(text='warning')), False),
    (
      agent_pb2.AgentMessage(
        context_window=agent_pb2.ContextWindowRecord(system_prompt='setup')
      ),
      False,
    ),
    (
      agent_pb2.AgentMessage(
        chat=agent_pb2.ChatRecord(
          system=agent_pb2.SystemMessage(content='setup')
        )
      ),
      False,
    ),
    (
      agent_pb2.AgentMessage(
        chat=agent_pb2.ChatRecord(user=agent_pb2.UserMessage(content='hello'))
      ),
      True,
    ),
    (
      agent_pb2.AgentMessage(
        chat=agent_pb2.ChatRecord(
          assistant=agent_pb2.AssistantMessage(content='hello')
        )
      ),
      True,
    ),
    (
      agent_pb2.AgentMessage(
        chat=agent_pb2.ChatRecord(tool=agent_pb2.ToolMessage(content='result'))
      ),
      True,
    ),
    (
      agent_pb2.AgentMessage(
        task_result=agent_pb2.TaskResultRecord(content='result')
      ),
      True,
    ),
    (
      agent_pb2.AgentMessage(
        context_window=agent_pb2.ContextWindowRecord(
          compaction=agent_pb2.Compaction(summary='earlier conversation')
        )
      ),
      True,
    ),
  ],
)
def test_conversation_presence_tracks_live_and_replayed_entries(
  history: transcript.Transcript,
  entry: agent_pb2.AgentMessage,
  conversation: bool,
) -> None:
  assert not history.has_conversation
  history.append_replayed(entry)
  assert history.has_conversation is conversation
  history.adopt_replayed()
  assert history.has_conversation is conversation
  history.clear_history()
  assert not history.has_conversation
  history.append(entry)
  assert history.has_conversation is conversation
  history.truncate(0)
  assert not history.has_conversation


@pytest.mark.parametrize('view_mode', list(actions.ViewMode))
def test_user_message_bar_matches_live_and_replayed_history(
  view_mode: actions.ViewMode,
) -> None:
  history = transcript.Transcript(
    view_mode=lambda: view_mode,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )
  entry = agent_pb2.AgentMessage(
    chat=agent_pb2.ChatRecord(user=agent_pb2.UserMessage(content='hello'))
  )
  live = history.project(entry, {})
  assert live is not None
  history.append_replayed(entry)
  history.adopt_replayed()

  assert (
    history.render_history([], [], console_width=40, scrollback_limit=100)
    == rendering.render_blocks([live], 40) + '\n\n'
  )


def test_ui_notices_do_not_start_a_conversation(
  history: transcript.Transcript,
) -> None:
  history.add_decoration(text.Text('Beta mode is enabled.'))
  assert not history.has_conversation


def test_welcome_notices_have_one_retained_copy(
  history: transcript.Transcript,
) -> None:
  notice = text.Text('Beta mode is enabled.')
  history.add_decoration(notice, welcome=True)
  history.add_decoration(text.Text('Other notice'))

  assert history.welcome_notices() == (notice,)
  live_history = history.render_history(
    [],
    [],
    console_width=80,
    scrollback_limit=100,
    include_welcome_notices=False,
  )
  assert 'Beta mode' not in live_history
  assert 'Other notice' in live_history
  scrollback = history.render_history(
    [],
    [],
    console_width=80,
    scrollback_limit=100,
  )
  assert scrollback.count('Beta mode is enabled.') == 1
  history.clear_history()
  assert not history.welcome_notices()
