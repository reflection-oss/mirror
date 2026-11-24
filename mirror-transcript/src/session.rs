//! Narrow transcript session: documents, committed geometry, and input ownership.

use std::collections::HashMap;
use std::io::{self, Write};
use std::time::{Duration, Instant};

use crate::{invalid, protocol, Compositor, InputRouter, RoutedInput, Viewport};

#[derive(Debug, Default)]
pub struct Session {
    views: HashMap<String, Viewport>,
    compositor: Compositor,
    router: InputRouter,
    sequence: u64,
    input_sequence: u64,
    paused: bool,
    closing: bool,
    reported_reading: Option<(String, bool)>,
    terminal_size: Option<protocol::Size>,
    scrolled: Option<Instant>,
}

/// How long the scroll thumb shows in full after native scrolling stops.
const THUMB_HOLD: Duration = Duration::from_millis(325);
/// How long the thumb shows in each `scrollbar-fade-N` role as it hides.
const THUMB_FADE_STEP: Duration = Duration::from_millis(50);

/// How many fade steps the scroll thumb has taken `elapsed` after scrolling;
/// 0 while it shows in full.
fn thumb_step(elapsed: Duration) -> u32 {
    elapsed.checked_sub(THUMB_HOLD).map_or(0, |fading| {
        u32::try_from(fading.as_nanos() / THUMB_FADE_STEP.as_nanos()).unwrap_or(u32::MAX - 1) + 1
    })
}

fn thumb_role(step: u32) -> String {
    if step == 0 {
        "scrollbar".into()
    } else {
        format!("scrollbar-fade-{step}")
    }
}

impl Session {
    pub fn resize(&mut self, size: protocol::Size) {
        self.terminal_size = Some(size);
        self.compositor.invalidate();
    }

    pub fn paused(&self) -> bool {
        self.paused
    }
    pub fn closing(&self) -> bool {
        self.closing
    }
    pub fn pending_input(&self) -> bool {
        self.router.pending_escape()
    }

    /// When the scroll thumb next changes as it lingers and fades.
    pub fn thumb_deadline(&self) -> Option<Instant> {
        let scrolled = self.scrolled?;
        Some(scrolled + THUMB_HOLD + THUMB_FADE_STEP * thumb_step(scrolled.elapsed()))
    }

    pub fn command(
        &mut self,
        command: protocol::TranscriptCommand,
        terminal: &mut impl Write,
    ) -> io::Result<Vec<protocol::TranscriptEvent>> {
        if command.sequence <= self.sequence {
            return Err(invalid("transcript command sequence must increase"));
        }
        self.sequence = command.sequence;
        let paint = !matches!(
            command.payload,
            Some(
                protocol::transcript_command::Payload::Terminal(_)
                    | protocol::transcript_command::Payload::Pause(_)
                    | protocol::transcript_command::Payload::Quit(_)
            )
        );
        let mut events = if matches!(
            command.payload,
            Some(protocol::transcript_command::Payload::Pause(true))
        ) {
            self.flush_input()
        } else {
            Vec::new()
        };
        if let Err(error) = self.apply(command, terminal) {
            if error.kind() != io::ErrorKind::InvalidData {
                return Err(error);
            }
            return Ok(vec![protocol::TranscriptEvent {
                payload: Some(protocol::transcript_event::Payload::Rejected(
                    protocol::Rejected {
                        command: self.sequence,
                        reason: error.to_string(),
                    },
                )),
            }]);
        }
        let active = self
            .compositor
            .frame()
            .filter(|frame| frame.transcript.is_some())
            .map(|frame| frame.view.as_str());
        for (id, view) in &mut self.views {
            if Some(id.as_str()) != active {
                view.evict_layout();
            }
        }
        if paint {
            self.present(terminal)?;
        }
        events.extend(self.position_event());
        events.push(protocol::TranscriptEvent {
            payload: Some(protocol::transcript_event::Payload::Presented(
                self.sequence,
            )),
        });
        Ok(events)
    }

    fn apply(
        &mut self,
        command: protocol::TranscriptCommand,
        terminal: &mut impl Write,
    ) -> io::Result<()> {
        use protocol::transcript_command::Payload;
        match command
            .payload
            .ok_or_else(|| invalid("transcript command has no payload"))?
        {
            Payload::Update(update) => self.update(update)?,
            Payload::Frame(frame) => self.frame(frame)?,
            Payload::Move(movement) => {
                if let Some(view) = self.active_view() {
                    view.navigate(&movement);
                }
            }
            Payload::Navigate(navigate) => {
                if let (Some(view), Some(movement)) =
                    (self.views.get_mut(&navigate.view), navigate.r#move)
                {
                    view.navigate(&movement);
                }
            }
            Payload::Terminal(bytes) => {
                if self.paused {
                    return Err(invalid("terminal writer is handed off"));
                }
                terminal.write_all(&bytes)?;
                terminal.flush()?;
                self.compositor.invalidate();
            }
            Payload::Pause(paused) => {
                if !self.router.pending() {
                    self.paused = paused;
                    self.compositor.invalidate();
                } else {
                    return Err(invalid("terminal handoff requires drained input"));
                }
            }
            Payload::Quit(_) => self.closing = true,
            Payload::DropView(view) => {
                if self
                    .compositor
                    .frame()
                    .is_some_and(|frame| frame.view == view && frame.transcript.is_some())
                {
                    return Err(invalid("cannot drop the visible transcript"));
                }
                self.views.remove(&view);
            }
            Payload::Settle(settlement) => {
                if let Some(view) = self.views.get_mut(&settlement.view) {
                    view.settle(settlement.before, settlement.after);
                }
            }
        }
        Ok(())
    }

    fn update(&mut self, mut update: protocol::TranscriptUpdate) -> io::Result<()> {
        if update.view.is_empty()
            || update.view.len() > 256
            || (!self.views.contains_key(&update.view) && self.views.len() >= 128)
        {
            return Err(invalid(
                "invalid transcript view or more than 128 retained views",
            ));
        }
        let id = update.view.clone();
        let budget = crate::MAX_RETAINED_SOURCE.saturating_sub(
            self.views
                .iter()
                .filter(|(view, _)| *view != &id)
                .map(|(_, view)| view.document.source_bytes())
                .sum::<usize>(),
        );
        let revision = update.revision;
        let settlements = std::mem::take(&mut update.settlements);
        if update.reset || !self.views.contains_key(&id) {
            let mut view = Viewport::default();
            view.document.update_bounded(update, budget)?;
            self.views.insert(id.clone(), view);
        } else if let Some(view) = self.views.get_mut(&id) {
            view.document.update_bounded(update, budget)?;
        }
        if let Some(view) = self.views.get_mut(&id) {
            view.revision = revision;
            view.clear_selection();
            for settlement in settlements {
                view.settle(settlement.before, settlement.after);
            }
        }
        Ok(())
    }

    fn frame(&mut self, mut frame: protocol::ChromeFrame) -> io::Result<()> {
        crate::validate_frame(&frame)?;
        if frame.input_sequence > self.input_sequence
            || (frame.transcript.is_some()
                && frame.update.is_none()
                && !self.views.contains_key(&frame.view))
        {
            return Err(invalid(
                "chrome frame references an unknown view or future input",
            ));
        }
        if let Some(update) = frame.update.take() {
            self.update(*update)?;
        }
        let changed = self.compositor.frame().is_none_or(|previous| {
            previous.view != frame.view
                || previous.transcript != frame.transcript
                || previous.occlusions != frame.occlusions
                || previous.wheel != frame.wheel
        });
        if !frame.moves.is_empty() {
            if let (Some(rect), Some(view)) = (&frame.transcript, self.views.get_mut(&frame.view)) {
                view.rows(rect.width as u16, rect.height as u16);
                for movement in frame.moves.drain(..) {
                    view.navigate(&movement);
                }
            }
        }
        self.compositor.set_frame(frame)?;
        if changed {
            self.router.cancel_capture();
            for view in self.views.values_mut() {
                view.clear_selection();
            }
        }
        Ok(())
    }

    fn active_view(&mut self) -> Option<&mut Viewport> {
        let frame = self.compositor.frame()?;
        frame.transcript.as_ref()?;
        self.views.get_mut(&frame.view)
    }

    pub fn input(
        &mut self,
        bytes: &[u8],
        terminal: &mut impl Write,
    ) -> io::Result<Vec<protocol::TranscriptEvent>> {
        let frame = self
            .compositor
            .frame()
            .filter(|frame| self.terminal_size.is_none() || frame.size == self.terminal_size);
        let blocked = frame.is_none_or(|frame| frame.input_sequence != self.input_sequence);
        let mut events = Vec::new();
        let mut moved = false;
        for input in self.router.feed(bytes, frame, blocked)? {
            match input {
                RoutedInput::Move(movement) => {
                    moved = true;
                    self.scrolled = Some(Instant::now());
                    if let Some(view) = self.active_view() {
                        view.navigate(&movement);
                    }
                }
                RoutedInput::Pointer(pointer) => {
                    moved = true;
                    if let Some(event) = self.pointer(pointer) {
                        events.push(event);
                    }
                }
                RoutedInput::Python(bytes) => {
                    moved = true;
                    events.push(self.python_input(bytes));
                }
            }
        }
        if moved {
            self.present(terminal)?;
        }
        events.extend(self.position_event());
        Ok(events)
    }

    fn pointer(&mut self, pointer: crate::Pointer) -> Option<protocol::TranscriptEvent> {
        let frame = self.compositor.frame()?;
        let rect = frame.transcript.as_ref()?;
        let x = pointer.x.saturating_sub(rect.x).min(rect.width) as usize;
        let y = pointer
            .y
            .saturating_sub(rect.y)
            .min(rect.height.saturating_sub(1)) as usize;
        let action =
            self.views
                .get_mut(&frame.view)?
                .pointer(x, y, pointer.button, pointer.released)?;
        let action = match action {
            crate::Action::Copy(text) => protocol::transcript_action::Action::Copy(text),
            crate::Action::Link(url) => protocol::transcript_action::Action::Link(url),
        };
        Some(protocol::TranscriptEvent {
            payload: Some(protocol::transcript_event::Payload::Action(
                protocol::TranscriptAction {
                    view: frame.view.clone(),
                    revision: self.views.get(&frame.view)?.revision,
                    action: Some(action),
                },
            )),
        })
    }

    pub fn flush_input(&mut self) -> Vec<protocol::TranscriptEvent> {
        let bytes = self.router.flush();
        if bytes.is_empty() {
            vec![]
        } else {
            vec![self.python_input(bytes)]
        }
    }

    fn python_input(&mut self, data: Vec<u8>) -> protocol::TranscriptEvent {
        self.input_sequence += 1;
        let selection = self
            .active_view()
            .and_then(Viewport::take_selection)
            .unwrap_or_default();
        protocol::TranscriptEvent {
            payload: Some(protocol::transcript_event::Payload::Input(
                protocol::TranscriptInput {
                    sequence: self.input_sequence,
                    data,
                    selection,
                },
            )),
        }
    }

    fn position_event(&mut self) -> Vec<protocol::TranscriptEvent> {
        let Some(frame) = self.compositor.frame() else {
            return vec![];
        };
        let Some(view) = self.views.get(&frame.view) else {
            return vec![];
        };
        let position = (frame.view.clone(), view.reading());
        if self.reported_reading.as_ref() == Some(&position) {
            return vec![];
        }
        self.reported_reading = Some(position.clone());
        vec![protocol::TranscriptEvent {
            payload: Some(protocol::transcript_event::Payload::Position(
                protocol::TranscriptPosition {
                    view: position.0,
                    reading: position.1,
                },
            )),
        }]
    }

    /// The scroll thumb's current style, forgetting the scroll once it has
    /// faded or nothing can show it.
    fn thumb(&mut self) -> Option<String> {
        let thumb = self.scrolled.filter(|_| !self.paused).and_then(|scrolled| {
            let frame = self.compositor.frame()?;
            self.views
                .get(&frame.view)?
                .document
                .theme()
                .roles
                .get(&thumb_role(thumb_step(scrolled.elapsed())))
                .cloned()
        });
        if thumb.is_none() {
            self.scrolled = None;
        }
        thumb
    }

    /// Paint the active view, including the scroll thumb's current fade step.
    pub fn present(&mut self, terminal: &mut impl Write) -> io::Result<()> {
        let thumb = self.thumb();
        if self.paused
            || self.closing
            || self.compositor.frame().is_some_and(|frame| {
                self.terminal_size.is_some() && frame.size != self.terminal_size
            })
        {
            return Ok(());
        }
        let view = self
            .compositor
            .frame()
            .and_then(|frame| self.views.get_mut(&frame.view));
        self.compositor.present(terminal, view, thumb.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use protocol::transcript_command::Payload;

    fn update(view: &str, revision: u64) -> Payload {
        Payload::Update(protocol::TranscriptUpdate {
            view: view.into(),
            revision,
            set: vec![protocol::Element {
                id: 1,
                blocks: vec![protocol::Block {
                    text: "[link](https://example.test)".into(),
                    kind: protocol::block::Kind::Markdown as i32,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            order: Some(protocol::TranscriptOrder { ids: vec![1] }),
            ..Default::default()
        })
    }

    fn frame(view: &str) -> protocol::ChromeFrame {
        protocol::ChromeFrame {
            size: Some(protocol::Size {
                columns: 20,
                rows: 2,
            }),
            rows: vec![vec![b' '; 20]; 2],
            view: view.into(),
            transcript: Some(protocol::TranscriptRect {
                width: 20,
                height: 2,
                ..Default::default()
            }),
            wheel: true,
            ..Default::default()
        }
    }

    fn command(
        session: &mut Session,
        sequence: u64,
        payload: Payload,
    ) -> Vec<protocol::TranscriptEvent> {
        session
            .command(
                protocol::TranscriptCommand {
                    sequence,
                    payload: Some(payload),
                },
                &mut Vec::new(),
            )
            .unwrap()
    }

    fn large_update(view: &str) -> Payload {
        let block = protocol::Block {
            text: ("x".repeat(64 * 1024) + "\n").repeat(104),
            ..Default::default()
        };
        Payload::Update(protocol::TranscriptUpdate {
            view: view.into(),
            set: (1..=4)
                .map(|id| protocol::Element {
                    id,
                    blocks: vec![block.clone()],
                    ..Default::default()
                })
                .collect(),
            order: Some(protocol::TranscriptOrder {
                ids: vec![1, 2, 3, 4],
            }),
            ..Default::default()
        })
    }

    #[test]
    fn thumb_fades_through_the_theme_roles_after_scrolling() {
        assert_eq!(thumb_role(thumb_step(Duration::ZERO)), "scrollbar");
        assert_eq!(thumb_role(thumb_step(THUMB_HOLD)), "scrollbar-fade-1");
        assert_eq!(
            thumb_role(thumb_step(THUMB_HOLD + THUMB_FADE_STEP * 2)),
            "scrollbar-fade-3"
        );
    }

    #[test]
    fn wheel_scrolling_shows_the_thumb_only_when_the_theme_has_one() {
        let mut session = Session::default();
        let lines = Payload::Update(protocol::TranscriptUpdate {
            view: "one".into(),
            set: vec![protocol::Element {
                id: 1,
                blocks: vec![protocol::Block {
                    text: "1\n2\n3\n4\n5\n6".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            order: Some(protocol::TranscriptOrder { ids: vec![1] }),
            ..Default::default()
        });
        command(&mut session, 1, lines);
        command(&mut session, 2, Payload::Frame(frame("one")));
        let mut terminal = Vec::new();
        session.input(b"\x1b[<64;2;1M", &mut terminal).unwrap();
        assert!(!String::from_utf8(terminal).unwrap().contains("\x1b[0;"));
        assert_matches!(session.thumb_deadline(), None);
        command(
            &mut session,
            3,
            Payload::Update(protocol::TranscriptUpdate {
                view: "one".into(),
                theme: Some(protocol::Theme {
                    roles: [("scrollbar".to_owned(), "90".to_owned())].into(),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        );
        let mut terminal = Vec::new();
        session.input(b"\x1b[<64;2;1M", &mut terminal).unwrap();
        assert!(String::from_utf8(terminal).unwrap().contains("\x1b[0;90"));
        assert!(session.thumb_deadline().is_some());
    }

    #[test]
    fn frame_commits_source_and_smaller_viewport_without_an_intermediate_paint() {
        let mut session = Session::default();
        command(&mut session, 1, update("one", 1));
        command(&mut session, 2, Payload::Frame(frame("one")));
        let mut next = frame("one");
        next.rows[1] = b"FOOTER".to_vec();
        next.transcript.as_mut().unwrap().height = 1;
        next.update = Some(Box::new(protocol::TranscriptUpdate {
            view: "one".into(),
            set: vec![protocol::Element {
                id: 1,
                blocks: vec![protocol::Block {
                    text: "new first\nnew last".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }));
        let mut terminal = Vec::new();
        session
            .command(
                protocol::TranscriptCommand {
                    sequence: 3,
                    payload: Some(Payload::Frame(next)),
                },
                &mut terminal,
            )
            .unwrap();
        let painted = String::from_utf8(terminal).unwrap();
        assert_eq!(painted.matches("\x1b[?2026h").count(), 1);
        assert!(!painted.contains("new first"));
        assert!(painted.contains("\x1b[1;1Hnew last"));
        assert!(painted.contains("\x1b[2;1HFOOTER"));
    }

    #[test]
    fn invalid_frame_does_not_commit_its_source_update() {
        let mut session = Session::default();
        command(&mut session, 1, update("one", 1));
        command(&mut session, 2, Payload::Frame(frame("one")));
        let mut next = frame("one");
        next.cursor_y = 2;
        next.update = Some(Box::new(protocol::TranscriptUpdate {
            view: "one".into(),
            order: Some(protocol::TranscriptOrder { ids: vec![] }),
            ..Default::default()
        }));
        let rejected = command(&mut session, 3, Payload::Frame(next));
        assert_matches!(
            rejected[0].payload,
            Some(protocol::transcript_event::Payload::Rejected(_))
        );
        session.input(b"\x1b[<0;2;1M", &mut Vec::new()).unwrap();
        let events = session.input(b"\x1b[<0;2;1m", &mut Vec::new()).unwrap();
        assert_matches!(events[0].payload.as_ref(), Some(protocol::transcript_event::Payload::Action(action)) if action.revision == 1);
    }

    #[test]
    fn aggregate_source_admission_rejects_a_view_until_another_is_dropped() {
        let mut session = Session::default();
        command(&mut session, 1, large_update("one"));
        command(&mut session, 2, large_update("two"));
        command(&mut session, 3, large_update("three"));
        let accepted = command(&mut session, 4, large_update("four"));
        assert_matches!(
            accepted[0].payload,
            Some(protocol::transcript_event::Payload::Presented(4))
        );
        let rejected = command(&mut session, 5, large_update("five"));
        assert_matches!(rejected[0].payload.as_ref(), Some(protocol::transcript_event::Payload::Rejected(rejected)) if rejected.reason.contains("retained source budget"));
        let absent = command(&mut session, 6, Payload::Frame(frame("five")));
        assert_matches!(
            absent[0].payload,
            Some(protocol::transcript_event::Payload::Rejected(_))
        );
        command(&mut session, 7, Payload::DropView("one".into()));
        let accepted = command(&mut session, 8, large_update("five"));
        assert_matches!(
            accepted[0].payload,
            Some(protocol::transcript_event::Payload::Presented(8))
        );
    }

    #[test]
    fn source_replacement_cancels_click_and_fresh_action_carries_revision() {
        let mut session = Session::default();
        command(&mut session, 1, update("one", 1));
        command(&mut session, 2, Payload::Frame(frame("one")));
        session.input(b"\x1b[<0;2;1M", &mut Vec::new()).unwrap();
        command(&mut session, 3, update("one", 2));
        assert_eq!(
            session.input(b"\x1b[<0;2;1m", &mut Vec::new()).unwrap(),
            vec![]
        );
        session.input(b"\x1b[<0;2;1M", &mut Vec::new()).unwrap();
        let events = session.input(b"\x1b[<0;2;1m", &mut Vec::new()).unwrap();
        assert_matches!(events[0].payload.as_ref(), Some(protocol::transcript_event::Payload::Action(action)) if action.revision == 2 && action.view == "one");
    }

    #[test]
    fn occlusion_cancels_capture_and_routes_the_release_to_python() {
        let mut session = Session::default();
        command(&mut session, 1, update("one", 1));
        command(&mut session, 2, Payload::Frame(frame("one")));
        session.input(b"\x1b[<0;2;1M", &mut Vec::new()).unwrap();
        let mut covered = frame("one");
        covered.occlusions.push(protocol::TranscriptRect {
            width: 20,
            height: 1,
            ..Default::default()
        });
        command(&mut session, 3, Payload::Frame(covered));
        let events = session.input(b"\x1b[<0;2;1m", &mut Vec::new()).unwrap();
        assert_matches!(events[0].payload.as_ref(), Some(protocol::transcript_event::Payload::Input(input)) if input.data == b"\x1b[<0;2;1m");
    }

    #[test]
    fn forwarded_key_carries_and_dismisses_the_transcript_selection() {
        let mut session = Session::default();
        command(&mut session, 1, update("one", 1));
        command(&mut session, 2, Payload::Frame(frame("one")));
        session
            .input(b"\x1b[<0;1;1M\x1b[<32;5;1M\x1b[<0;5;1m", &mut Vec::new())
            .unwrap();
        let events = session.input(b"\x1bl", &mut Vec::new()).unwrap();
        assert_matches!(events[0].payload.as_ref(), Some(protocol::transcript_event::Payload::Input(input)) if input.data == b"\x1bl" && input.selection == "link");
        let events = session.input(b"\x1bl", &mut Vec::new()).unwrap();
        assert_matches!(events[0].payload.as_ref(), Some(protocol::transcript_event::Payload::Input(input)) if input.selection.is_empty());
    }

    #[test]
    fn dropping_inactive_view_releases_it_without_disturbing_the_active_view() {
        let mut session = Session::default();
        command(&mut session, 1, update("one", 1));
        command(&mut session, 2, update("two", 1));
        command(&mut session, 3, Payload::Frame(frame("two")));
        command(&mut session, 4, Payload::DropView("one".into()));
        let rejected = command(&mut session, 5, Payload::Frame(frame("one")));
        assert_matches!(
            rejected[0].payload,
            Some(protocol::transcript_event::Payload::Rejected(_))
        );
        let rejected = command(&mut session, 6, Payload::DropView("two".into()));
        assert_matches!(
            rejected[0].payload,
            Some(protocol::transcript_event::Payload::Rejected(_))
        );
        session.input(b"\x1b[<0;2;1M", &mut Vec::new()).unwrap();
        let events = session.input(b"\x1b[<0;2;1m", &mut Vec::new()).unwrap();
        assert_matches!(events[0].payload.as_ref(), Some(protocol::transcript_event::Payload::Action(action)) if action.view == "two");
    }
}
