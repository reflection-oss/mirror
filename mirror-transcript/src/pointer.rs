//! SGR pointer coordinates, kept independent of Python mouse dispatch.

use crate::protocol;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pointer {
    pub x: u32,
    pub y: u32,
    pub button: u32,
    pub released: bool,
}

impl Pointer {
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        let text = std::str::from_utf8(bytes).ok()?.strip_prefix("\x1b[<")?;
        let released = text.ends_with('m');
        let mut values = text
            .strip_suffix(if released { 'm' } else { 'M' })?
            .split(';');
        let button = values.next()?.parse().ok()?;
        let x = values.next()?.parse::<u32>().ok()?.checked_sub(1)?;
        let y = values.next()?.parse::<u32>().ok()?.checked_sub(1)?;
        values.next().is_none().then_some(Self {
            x,
            y,
            button,
            released,
        })
    }

    pub fn inside(&self, frame: &protocol::ChromeFrame) -> bool {
        frame.transcript.as_ref().is_some_and(|rect| {
            self.x >= rect.x
                && self.x < rect.x + rect.width
                && self.y >= rect.y
                && self.y < rect.y + rect.height
                && !frame.occlusions.iter().any(|cover| {
                    self.x >= cover.x
                        && self.x < cover.x + cover.width
                        && self.y >= cover.y
                        && self.y < cover.y + cover.height
                })
        })
    }
}
