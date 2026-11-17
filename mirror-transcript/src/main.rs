//! Sole terminal writer for Python chrome and a native transcript viewport.
#![forbid(unsafe_code)]

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::OpenOptionsExt as _;
use std::time::Duration;

use mirror_transcript::{invalid, protocol, Outbox, Session, MAX_FRAME};
use prost::Message as _;
use tokio::io::unix::AsyncFd;
use tokio::signal::unix::{signal, SignalKind};

#[tokio::main(flavor = "current_thread")]
async fn main() -> io::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.as_slice() == ["--version"] {
        println!("mirror-transcript {} protocol 1", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    let path = match args.as_slice() {
        [] => "/dev/tty",
        [flag, path] if flag == "--terminal" => path,
        _ => return Err(invalid("usage: mirror-transcript [--terminal PATH]")),
    };
    let input = AsyncFd::new(
        OpenOptions::new()
            .read(true)
            .custom_flags(rustix::fs::OFlags::NONBLOCK.bits() as i32)
            .open(path)?,
    )?;
    let mut terminal = OpenOptions::new().write(true).open(path)?;
    let commands = nonblocking(File::from(rustix::io::dup(io::stdin())?))?;
    let events = nonblocking(File::from(rustix::io::dup(io::stdout())?))?;
    let mut interrupt = signal(SignalKind::interrupt())?;
    let mut terminate = signal(SignalKind::terminate())?;
    let mut resize = signal(SignalKind::window_change())?;
    let mut session = Session::default();
    let mut outbox = Outbox::default();
    let mut buffer = Vec::new();
    let mut escape = tokio::time::Instant::now();
    outbox.push(protocol::TranscriptEvent {
        payload: Some(protocol::transcript_event::Payload::Ready(
            "mirror-transcript/1".into(),
        )),
    })?;
    outbox.push(size_event(&terminal, &mut session)?)?;
    loop {
        if session.closing() && !outbox.pending() {
            break;
        }
        tokio::select! {
            // Python retains its application-level interrupt and foreground policy.
            _ = interrupt.recv() => {},
            _ = terminate.recv() => break,
            _ = resize.recv() => outbox.push(size_event(&terminal, &mut session)?)?,
            readiness = commands.readable(), if !session.closing() => {
                let mut readiness = readiness?;
                let mut bytes = [0; 8192];
                match readiness.try_io(|fd| Ok(rustix::io::read(fd.get_ref(), &mut bytes)?)) {
                    Ok(Ok(0)) => break,
                    Ok(Ok(count)) => {
                        buffer.extend_from_slice(&bytes[..count]);
                        process_commands(&mut buffer, &mut session, &mut terminal, &mut outbox)?;
                    }
                    Ok(Err(error)) => return Err(error),
                    Err(_) => {},
                }
            },
            readiness = input.readable(), if !session.paused() && !session.closing() => {
                let mut readiness = readiness?;
                let mut bytes = [0; 4096];
                match readiness.try_io(|fd| Ok(rustix::io::read(fd.get_ref(), &mut bytes)?)) {
                    Ok(Ok(0)) => break,
                    Ok(Ok(count)) => {
                        escape = tokio::time::Instant::now() + Duration::from_millis(50);
                        for event in session.input(&bytes[..count], &mut terminal)? { outbox.push(event)?; }
                    }
                    Ok(Err(error)) => return Err(error),
                    Err(_) => {},
                }
            },
            readiness = events.writable(), if outbox.pending() => {
                let mut readiness = readiness?;
                if let Ok(result) = readiness.try_io(|fd| outbox.write(fd.get_ref())) { result?; }
            },
            _ = tokio::time::sleep_until(escape), if session.pending_input() => {
                for event in session.flush_input() { outbox.push(event)?; }
            },
        }
    }
    Ok(())
}

fn nonblocking(file: File) -> io::Result<AsyncFd<File>> {
    let flags = rustix::fs::fcntl_getfl(&file)?;
    rustix::fs::fcntl_setfl(&file, flags | rustix::fs::OFlags::NONBLOCK)?;
    AsyncFd::new(file)
}

fn size_event(terminal: &File, session: &mut Session) -> io::Result<protocol::TranscriptEvent> {
    let size = rustix::termios::tcgetwinsize(terminal)?;
    let size = protocol::Size {
        columns: size.ws_col.into(),
        rows: size.ws_row.into(),
    };
    session.resize(size);
    Ok(protocol::TranscriptEvent {
        payload: Some(protocol::transcript_event::Payload::Resize(size)),
    })
}

fn process_commands(
    buffer: &mut Vec<u8>,
    session: &mut Session,
    terminal: &mut File,
    outbox: &mut Outbox,
) -> io::Result<()> {
    let mut offset = 0;
    while let Some(header) = buffer.get(offset..offset + 4) {
        let length = u32::from_le_bytes(
            header
                .try_into()
                .map_err(|_| invalid("invalid transcript frame header"))?,
        ) as usize;
        if length == 0 || length > MAX_FRAME {
            return Err(invalid("transcript frame must contain 1 byte..8 MiB"));
        }
        let Some(payload) = buffer.get(offset + 4..offset + 4 + length) else {
            break;
        };
        let command = protocol::TranscriptCommand::decode(payload)?;
        for event in session.command(command, terminal)? {
            outbox.push(event)?;
        }
        offset += 4 + length;
        if session.closing() {
            break;
        }
    }
    buffer.drain(..offset);
    Ok(())
}
