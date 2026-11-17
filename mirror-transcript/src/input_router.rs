//! Recognize delegated transcript input while preserving all other terminal bytes.

use std::io;

use crate::{invalid, protocol};

#[derive(Debug, Default)]
pub struct InputRouter {
    pending: Vec<u8>,
    paste: bool,
    capture: bool,
}

#[derive(Debug, PartialEq)]
pub enum RoutedInput {
    Python(Vec<u8>),
    Move(protocol::TranscriptMove),
    Pointer(crate::Pointer),
}

impl InputRouter {
    pub fn cancel_capture(&mut self) {
        self.capture = false;
    }

    pub fn pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn pending_escape(&self) -> bool {
        !self.paste && self.pending()
    }

    pub fn flush(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.pending)
    }

    pub fn feed(
        &mut self,
        bytes: &[u8],
        frame: Option<&protocol::ChromeFrame>,
        mut blocked: bool,
    ) -> io::Result<Vec<RoutedInput>> {
        if self.pending.len() + bytes.len() > 1024 * 1024 {
            return Err(invalid("terminal input backlog exceeds 1 MiB"));
        }
        self.pending.extend_from_slice(bytes);
        let mut result = Vec::new();
        let mut offset = 0;
        while offset < self.pending.len() {
            let remaining = &self.pending[offset..];
            if self.paste {
                if b"\x1b[201~".starts_with(remaining) && remaining.len() < 6 {
                    break;
                }
                if remaining.starts_with(b"\x1b[201~") {
                    self.paste = false;
                }
                let count = if remaining.starts_with(b"\x1b[201~") {
                    6
                } else {
                    1
                };
                forward(&mut result, &remaining[..count]);
                offset += count;
                blocked = true;
                continue;
            }
            let routes = frame
                .filter(|_| !blocked)
                .map_or(&[][..], |frame| frame.routes.as_slice());
            if routes.iter().any(|route| {
                route.input.starts_with(remaining) && route.input.len() > remaining.len()
            }) {
                break;
            }
            if let Some(route) = routes
                .iter()
                .filter(|route| remaining.starts_with(&route.input))
                .max_by_key(|route| route.input.len())
            {
                if let Some(movement) = route.r#move {
                    result.push(RoutedInput::Move(movement));
                    offset += route.input.len();
                    continue;
                }
            }
            let Some(count) = token_length(remaining) else {
                break;
            };
            let token = &remaining[..count];
            if token == b"\x1b[200~" {
                self.paste = true;
            }
            if let Some(movement) = frame
                .filter(|frame| frame.wheel && !blocked)
                .and_then(|frame| wheel(token, frame))
            {
                result.push(RoutedInput::Move(movement));
            } else if let Some(pointer) =
                frame
                    .filter(|frame| frame.wheel && !blocked)
                    .and_then(|frame| {
                        crate::Pointer::parse(token).filter(|pointer| {
                            matches!(pointer.button, 0 | 32 | 35)
                                && (self.capture || pointer.inside(frame))
                        })
                    })
            {
                if pointer.button == 0 {
                    self.capture = !pointer.released;
                }
                result.push(RoutedInput::Pointer(pointer));
            } else {
                self.capture = false;
                forward(&mut result, token);
                blocked = true;
            }
            offset += count;
        }
        self.pending.drain(..offset);
        if self.pending.len() > 128 && !self.paste {
            return Err(invalid("terminal escape sequence exceeds 128 bytes"));
        }
        Ok(result)
    }
}

fn forward(result: &mut Vec<RoutedInput>, bytes: &[u8]) {
    match result.last_mut() {
        Some(RoutedInput::Python(previous)) => previous.extend_from_slice(bytes),
        _ => result.push(RoutedInput::Python(bytes.to_vec())),
    }
}

fn token_length(bytes: &[u8]) -> Option<usize> {
    if bytes[0] != 0x1b {
        return Some(1);
    }
    match *bytes.get(1)? {
        b'[' => bytes[2..]
            .iter()
            .position(|byte| (0x40..=0x7e).contains(byte))
            .map(|offset| offset + 3),
        b'O' => (bytes.len() >= 3).then_some(3),
        _ => Some(2),
    }
}

fn wheel(token: &[u8], frame: &protocol::ChromeFrame) -> Option<protocol::TranscriptMove> {
    let rect = frame.transcript.as_ref()?;
    let values = std::str::from_utf8(token)
        .ok()?
        .strip_prefix("\x1b[<")?
        .strip_suffix('M')?;
    let mut values = values.split(';');
    let button: u32 = values.next()?.parse().ok()?;
    let x = values.next()?.parse::<u32>().ok()?.checked_sub(1)?;
    let y = values.next()?.parse::<u32>().ok()?.checked_sub(1)?;
    if values.next().is_some()
        || x < rect.x
        || x >= rect.x + rect.width
        || y < rect.y
        || y >= rect.y + rect.height
        || frame.occlusions.iter().any(|covered| {
            x >= covered.x
                && x < covered.x + covered.width
                && y >= covered.y
                && y < covered.y + covered.height
        })
    {
        return None;
    }
    let amount = match button {
        64 => -1,
        65 => 1,
        _ => return None,
    };
    Some(protocol::TranscriptMove {
        amount,
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    #[test]
    fn fragmented_wheel_is_local_but_pasted_wheel_bytes_are_not() {
        let frame = protocol::ChromeFrame {
            wheel: true,
            transcript: Some(protocol::TranscriptRect {
                x: 0,
                y: 0,
                width: 80,
                height: 20,
            }),
            ..Default::default()
        };
        let mut router = InputRouter::default();
        assert_eq!(
            router.feed(b"\x1b[<64;4", Some(&frame), false).unwrap(),
            vec![]
        );
        assert_matches!(router.feed(b";3M", Some(&frame), false).unwrap().as_slice(), [RoutedInput::Move(movement)] if movement.amount == -1);
        assert_eq!(
            router
                .feed(b"\x1b[200~\x1b[<64;4;3M\x1b[20", Some(&frame), false)
                .unwrap(),
            vec![RoutedInput::Python(b"\x1b[200~\x1b[<64;4;3M".to_vec())]
        );
        assert_eq!(
            router.feed(b"1~", Some(&frame), false).unwrap(),
            vec![RoutedInput::Python(b"\x1b[201~".to_vec())]
        );
    }

    #[test]
    fn earlier_python_input_prevents_routing_against_stale_geometry() {
        let frame = protocol::ChromeFrame {
            routes: vec![protocol::TranscriptRoute {
                input: b"\x1b[5~".to_vec(),
                r#move: Some(protocol::TranscriptMove {
                    kind: protocol::transcript_move::Kind::Page as i32,
                    amount: -1,
                }),
            }],
            ..Default::default()
        };
        let mut router = InputRouter::default();
        assert_eq!(
            router.feed(b"/menu\x1b[5~", Some(&frame), false).unwrap(),
            vec![RoutedInput::Python(b"/menu\x1b[5~".to_vec())]
        );
    }

    #[test]
    fn extended_editor_keys_pass_through_byte_for_byte() {
        let mut router = InputRouter::default();
        assert_eq!(
            router
                .feed(b"\x1b[122;6u\xf0\x9f\x98\x80", None, false)
                .unwrap(),
            vec![RoutedInput::Python(b"\x1b[122;6u\xf0\x9f\x98\x80".to_vec())]
        );
    }
}
