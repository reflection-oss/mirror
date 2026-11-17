//! Nonblocking event delivery never puts Python on the scrolling paint path.

use std::collections::VecDeque;
use std::io;
use std::os::fd::AsFd;

use prost::Message as _;

use crate::{invalid, protocol};

#[derive(Debug, Default)]
pub struct Outbox {
    queue: VecDeque<protocol::TranscriptEvent>,
    packet: Vec<u8>,
    offset: usize,
    bytes: usize,
}

impl Outbox {
    pub fn pending(&self) -> bool {
        self.offset < self.packet.len() || !self.queue.is_empty()
    }

    pub fn push(&mut self, event: protocol::TranscriptEvent) -> io::Result<()> {
        if let Some(protocol::transcript_event::Payload::Position(position)) = &event.payload {
            if let Some(previous) = self.queue.iter_mut().find(|event| matches!(&event.payload,
                Some(protocol::transcript_event::Payload::Position(before)) if before.view == position.view)) {
                self.bytes -= previous.encoded_len() + 4;
                self.bytes += event.encoded_len() + 4;
                *previous = event;
                return Ok(());
            }
        }
        let bytes = event.encoded_len() + 4;
        if self.bytes + bytes > 1024 * 1024 {
            return Err(invalid("Python input/event backlog exceeds 1 MiB"));
        }
        self.bytes += bytes;
        self.queue.push_back(event);
        Ok(())
    }

    pub fn write(&mut self, output: impl AsFd) -> io::Result<()> {
        if self.offset == self.packet.len() {
            let Some(event) = self.queue.pop_front() else {
                return Ok(());
            };
            self.packet.clear();
            self.offset = 0;
            self.packet
                .extend_from_slice(&(event.encoded_len() as u32).to_le_bytes());
            event.encode(&mut self.packet)?;
        }
        let written = rustix::io::write(output, &self.packet[self.offset..])?;
        if written == 0 {
            return Err(io::ErrorKind::WriteZero.into());
        }
        self.offset += written;
        self.bytes -= written;
        Ok(())
    }
}
