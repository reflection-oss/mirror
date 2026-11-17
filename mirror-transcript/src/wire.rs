//! Bounds and validation shared by the transcript transport and document.

use std::io;

use prost::Message as _;

use crate::protocol::{Block, Element, Span, Theme};

/// Maximum size of one length-prefixed command or event.
pub const MAX_FRAME: usize = 8 * 1024 * 1024;
/// Maximum serialized source retained by one transcript.
pub const MAX_SOURCE: usize = 32 * 1024 * 1024;
pub const MAX_ELEMENTS: usize = 100_000;
/// Aggregate retained source storage across all views, including source structure.
pub const MAX_RETAINED_SOURCE: usize = 128 * 1024 * 1024;
/// Only the active view retains layout and wrapped rows.
pub const MAX_CACHE: usize = 16 * 1024 * 1024;

pub fn retained_source_bytes(element: &Element) -> usize {
    // Reserve index/map/tree storage as well as the owned element itself.
    std::mem::size_of::<crate::Element>()
        + 256
        + element.lines.capacity() * std::mem::size_of::<crate::Line>()
        + element
            .lines
            .iter()
            .map(crate::source_line_heap_bytes)
            .sum::<usize>()
        + element.blocks.capacity() * std::mem::size_of::<Block>()
        + element
            .blocks
            .iter()
            .map(crate::block_heap_bytes)
            .sum::<usize>()
        + element
            .sticky
            .as_ref()
            .map_or(0, |span| span.text.capacity() + span.sgr.capacity())
}

pub fn retained_theme_bytes(theme: &Theme) -> usize {
    std::mem::size_of::<Theme>()
        + theme.text.capacity()
        + theme.muted.capacity()
        + theme.accent.capacity()
        + theme.selected.capacity()
        + theme.input.capacity()
        + theme.roles.capacity() * 2 * std::mem::size_of::<(String, String)>()
        + theme
            .roles
            .iter()
            .map(|(key, value)| key.capacity() + value.capacity())
            .sum::<usize>()
}

pub fn invalid(reason: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, reason)
}

pub fn validate_style(style: &str) -> io::Result<()> {
    if style.len() > 256
        || !style
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte == b';')
    {
        return Err(invalid("invalid transcript SGR parameters"));
    }
    Ok(())
}

pub fn validate_span(span: &Span) -> io::Result<()> {
    validate_style(&span.sgr)?;
    validate_text(&span.text)
}

fn validate_text(text: &str) -> io::Result<()> {
    if text.split('\n').any(|line| line.len() > 64 * 1024) {
        return Err(invalid("transcript source line exceeds 64 KiB"));
    }
    Ok(())
}

pub fn validate_theme(theme: &Theme) -> io::Result<()> {
    if theme.encoded_len() > 64 * 1024 {
        return Err(invalid("transcript theme exceeds 64 KiB"));
    }
    for style in [
        &theme.text,
        &theme.muted,
        &theme.accent,
        &theme.selected,
        &theme.input,
    ]
    .into_iter()
    .chain(theme.roles.values())
    {
        validate_style(style)?;
    }
    Ok(())
}

pub fn validate_element(element: &Element) -> io::Result<()> {
    if element.id == 0 || element.region != 0 || element.spinner || element.timer.is_some() {
        return Err(invalid(
            "transcript elements require a nonzero ID and source content only",
        ));
    }
    if element.encoded_len() > MAX_FRAME {
        return Err(invalid("transcript element exceeds 8 MiB"));
    }
    if let Some(sticky) = &element.sticky {
        validate_span(sticky)?;
    }
    let mut remaining = 8192usize;
    validate_lines(&element.lines, &mut remaining)?;
    validate_blocks(&element.blocks, 0, &mut remaining)
}

fn validate_lines(lines: &[crate::protocol::Line], remaining: &mut usize) -> io::Result<()> {
    *remaining = remaining
        .checked_sub(lines.len())
        .ok_or_else(|| invalid("too many transcript source lines"))?;
    for line in lines {
        for span in &line.spans {
            validate_span(span)?;
        }
    }
    Ok(())
}

fn validate_blocks(blocks: &[Block], depth: usize, remaining: &mut usize) -> io::Result<()> {
    if depth > 16 {
        return Err(invalid("transcript source nesting exceeds 16 levels"));
    }
    for block in blocks {
        *remaining = remaining
            .checked_sub(1 + block.text.bytes().filter(|&byte| byte == b'\n').count())
            .ok_or_else(|| invalid("too many transcript source lines"))?;
        validate_text(&block.text)?;
        validate_text(&block.title)?;
        for style in [&block.sgr, &block.title_sgr, &block.border_sgr] {
            validate_style(style)?;
        }
        if let Some(marker) = &block.marker {
            validate_span(marker)?;
        }
        validate_lines(&block.lines, remaining)?;
        validate_blocks(&block.children, depth + 1, remaining)?;
        validate_blocks(&block.collapsed, depth + 1, remaining)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    #[test]
    fn style_cannot_escape_into_a_terminal_command() {
        assert_matches!(validate_style("31m\x1b[2J"), Err(_));
        assert_eq!(validate_style("1;38;2;20;30;40").is_ok(), true);
    }
}
