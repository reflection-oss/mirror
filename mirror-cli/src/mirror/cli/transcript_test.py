import pytest
from mirror.protocol import agent_pb2
from rich import text

from mirror.cli import actions, rendering, transcript


@pytest.fixture
def history() -> transcript.Transcript:
  return transcript.Transcript(
    view_mode=lambda: actions.ViewMode.DETAIL,
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


@pytest.mark.parametrize(
  ('mode', 'visible'),
  [
    (actions.ViewMode.DEFAULT, False),
    (actions.ViewMode.DETAIL, True),
    (actions.ViewMode.FOCUS, False),
  ],
)
@pytest.mark.parametrize('content', ['', 'the answer'])
def test_reasoning_visibility_in_settled_and_replayed_history(
  mode: actions.ViewMode, visible: bool, content: str
) -> None:
  history = transcript.Transcript(
    view_mode=lambda: mode,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )
  entry = agent_pb2.AgentMessage(
    chat=agent_pb2.ChatRecord(
      assistant=agent_pb2.AssistantMessage(
        reasoning_content='available reasoning', content=content
      )
    )
  )
  history.append(entry)
  settled = history.render_history(
    [], [], console_width=80, scrollback_limit=100
  )
  assert ('available reasoning' in settled) is visible
  assert 'Reasoning' not in settled
  assert ('the answer' in settled) is bool(content)

  history.append_replayed(entry)
  history.adopt_replayed()

  assert (
    history.render_history([], [], console_width=80, scrollback_limit=100)
    == settled
  )
  assert history.entries() == (entry,)


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


@pytest.mark.parametrize('identity', ['', 'answer'])
def test_record_snapshots_detach_ingress_and_exports(
  history: transcript.Transcript, identity: str
) -> None:
  entry = agent_pb2.AgentMessage(
    id=identity,
    chat=agent_pb2.ChatRecord(
      assistant=agent_pb2.AssistantMessage(content='Original')
    ),
  )
  history.append(entry)
  history.append_replayed(entry)
  original = history.entries()[0]
  entry.chat.assistant.content = 'Changed input'
  history.entries()[0].chat.assistant.content = 'Changed export'
  history.replayed()[0].chat.assistant.content = 'Changed replay export'
  assert history.entries() == (original,)
  assert history.contains(original) == bool(identity)
  assert 'Original' in history.render_history(
    [], [], console_width=80, scrollback_limit=100
  )
  history.adopt_replayed()
  assert history.entries() == (original,)


def test_turn_end_records_are_not_drawn(
  history: transcript.Transcript,
) -> None:
  history.append(
    agent_pb2.AgentMessage(
      id='end',
      turn_end=agent_pb2.TurnEndRecord(
        outcome=agent_pb2.TURN_OUTCOME_FAILED, error='model offline'
      ),
    )
  )

  assert history.blocks() == ()
  assert not history.has_conversation
  assert 'model offline' not in history.render_history(
    [], [], console_width=80, scrollback_limit=100
  )


def test_lazy_blocks_keep_their_original_presentation() -> None:
  mode = actions.ViewMode.DETAIL
  history = transcript.Transcript(
    view_mode=lambda: mode,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )
  history.append(
    agent_pb2.AgentMessage(
      chat=agent_pb2.ChatRecord(
        assistant=agent_pb2.AssistantMessage(
          reasoning_content='Reasoning', content='Answer'
        )
      )
    )
  )
  original = history.blocks()[0]
  mode = actions.ViewMode.FOCUS
  focused = history.blocks()[0]
  assert 'Reasoning' in rendering.render_blocks([original.renderable], 80)
  assert 'Reasoning' not in rendering.render_blocks([focused.renderable], 80)
