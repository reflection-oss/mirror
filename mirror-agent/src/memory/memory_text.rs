//! Index-side text normalization so identifiers match the way a model writes them.

/// Insert a space at every lower→upper and letter↔digit boundary so FTS5's
/// `unicode61` tokenizer sees `camelCase` and `v2beta` as separate terms. The
/// original spelling is preserved by keeping the joined form as well.
pub fn split_compounds(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 2);
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String| {
        if word.is_empty() {
            return;
        }
        let parts = compound_parts(word);
        if parts.len() > 1 {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&parts.join(" "));
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word);
        word.clear();
    };
    for character in text.chars() {
        if character.is_alphanumeric() {
            word.push(character);
        } else {
            flush(&mut word, &mut out);
        }
    }
    flush(&mut word, &mut out);
    out
}

fn compound_parts(word: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut previous: Option<char> = None;
    for character in word.chars() {
        let boundary = match previous {
            Some(last) => {
                (last.is_lowercase() && character.is_uppercase())
                    || (last.is_alphabetic() && character.is_numeric())
                    || (last.is_numeric() && character.is_alphabetic())
            }
            None => false,
        };
        if boundary && !current.is_empty() {
            parts.push(std::mem::take(&mut current));
        }
        current.push(character);
        previous = Some(character);
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

/// A kebab-case key of at most 64 bytes derived from free text.
pub fn slug_key(text: &str) -> String {
    let mut key = String::new();
    let mut last_dash = true;
    for character in text.chars().flat_map(char::to_lowercase) {
        if character.is_ascii_alphanumeric() {
            key.push(character);
            last_dash = false;
        } else if !last_dash {
            key.push('-');
            last_dash = true;
        }
        if key.len() >= 60 {
            break;
        }
    }
    let key = key.trim_matches('-').to_string();
    if key.is_empty() {
        "note".to_string()
    } else {
        key
    }
}

/// Whether a user message is worth a lexical lookup: not empty, not a slash
/// command, not a bare acknowledgement, and at least two words.
pub fn should_query(text: &str) -> bool {
    static TRIVIAL: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(concat!(
            r"(?i)^(yes|no|ok|okay|sure|thanks|thank you|y|n|yep|nope|yeah|nah|",
            r"hi|hey|hello|yo|sup|continue|go ahead|do it|proceed|got it|cool|nice|great|done|next|lgtm|k)",
            "[\\s!?.:;,\"'~()\\[\\]{}<>*&^%$#@!+=`\u{a0}\u{2018}\u{2019}\u{201c}\u{201d}\u{2014}\u{2013}\u{2026}]*$"
        ))
        .expect("trivial prompt pattern compiles")
    });
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.starts_with('/') || TRIVIAL.is_match(trimmed) {
        return false;
    }
    trimmed.split_whitespace().nth(1).is_some()
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn splits_camel_case_and_digit_boundaries_keeping_the_original() {
        assert_eq!(
            split_compounds("ReactAgentConfig v2beta"),
            "React Agent Config ReactAgentConfig v 2 beta v2beta"
        );
    }

    #[test]
    fn leaves_plain_words_and_snake_case_alone() {
        assert_eq!(split_compounds("bazel build_file"), "bazel build file");
    }
}

#[cfg(test)]
mod query_tests {
    use super::*;

    #[test]
    fn gate_skips_acknowledgements_commands_and_one_word_prompts() {
        assert!(!should_query("ok"));
        assert!(!should_query("Thanks!!"));
        assert!(!should_query("go ahead"));
        assert!(!should_query("/memory"));
        assert!(!should_query("compaction"));
        assert!(should_query("fix the compaction loop"));
    }
}
