use pretty_assertions::assert_eq;

use super::*;

fn literal(text: &str) -> Word {
    Word::Literal(text.to_string())
}

fn words(script: &str) -> Vec<Vec<Word>> {
    commands(script)
        .into_iter()
        .map(|command| command.words)
        .collect()
}

#[test]
fn splits_lists_and_leaves_redirections_and_assignments_out_of_the_words() {
    assert_eq!(
        words("cd /testbed && LC_ALL=C grep -rn needle src 2>/dev/null; echo done"),
        vec![
            vec![literal("cd"), literal("/testbed")],
            vec![
                literal("grep"),
                literal("-rn"),
                literal("needle"),
                literal("src")
            ],
            vec![literal("echo"), literal("done")],
        ]
    );
}

#[test]
fn keeps_words_after_a_redirection_as_arguments() {
    assert_eq!(
        words("grep -rl x / 2>/dev/null --exclude-dir=proc | head; grep -rl x </dev/null /"),
        vec![
            vec![
                literal("grep"),
                literal("-rl"),
                literal("x"),
                literal("/"),
                literal("--exclude-dir=proc"),
            ],
            vec![literal("head")],
            vec![literal("grep"), literal("-rl"), literal("x"), literal("/")],
        ]
    );
}

#[test]
fn removes_quotes_and_escapes() {
    assert_eq!(
        words(r#"grep -e 'a b' "c \"d\"" e\ f -g'!proc'"#),
        vec![vec![
            literal("grep"),
            literal("-e"),
            literal("a b"),
            literal("c \"d\""),
            literal("e f"),
            literal("-g!proc"),
        ]]
    );
}

#[test]
fn keeps_a_substitution_as_one_expanded_word_and_lists_the_command_inside_it() {
    assert_eq!(
        words("grep -rn pat $(ruby -e 'puts 1') / | head"),
        vec![
            vec![
                literal("grep"),
                literal("-rn"),
                literal("pat"),
                Word::Expanded,
                literal("/"),
            ],
            vec![literal("ruby"), literal("-e"), literal("puts 1")],
            vec![literal("head")],
        ]
    );
}

#[test]
fn treats_quoted_expansions_as_expanded() {
    assert_eq!(
        words(r#"grep -r "$PATTERN" "$(pwd)/src" "${HOME}""#),
        vec![
            vec![
                literal("grep"),
                literal("-r"),
                Word::Expanded,
                Word::Expanded,
                Word::Expanded,
            ],
            vec![literal("pwd")],
        ]
    );
}

#[test]
fn links_each_pipeline_stage_to_the_command_feeding_it() {
    let parsed = commands("find / -type f 2>/dev/null | xargs grep -l x |& sort; ls | wc -l");
    let upstreams: Vec<Option<usize>> = parsed.iter().map(|command| command.upstream).collect();
    assert_eq!(upstreams, vec![None, Some(0), Some(1), None, Some(3)]);
}

#[test]
fn links_a_grouped_pipeline_stage_to_the_command_feeding_it() {
    let parsed = commands("find / -type f | (xargs grep -l x) | { sort; uniq; }");
    let upstreams: Vec<Option<usize>> = parsed.iter().map(|command| command.upstream).collect();
    assert_eq!(upstreams, vec![None, Some(0), Some(1), None]);
}

#[test]
fn applies_a_redirection_after_a_list_to_its_last_command() {
    let parsed = commands("cd /app && grep -rl x / 2>/dev/null --exclude-dir=proc | head");
    assert_eq!(
        parsed,
        vec![
            ShellCommand {
                words: vec![literal("cd"), literal("/app")],
                upstream: None,
                subshells: vec![],
            },
            ShellCommand {
                words: vec![
                    literal("grep"),
                    literal("-rl"),
                    literal("x"),
                    literal("/"),
                    literal("--exclude-dir=proc"),
                ],
                upstream: None,
                subshells: vec![],
            },
            ShellCommand {
                words: vec![literal("head")],
                upstream: Some(1),
                subshells: vec![],
            },
        ]
    );
}

#[test]
fn does_not_link_commands_joined_by_and_or_or() {
    let parsed = commands("find / -type f || xargs grep -l x && sort");
    assert!(parsed.iter().all(|command| command.upstream.is_none()));
}

#[test]
fn ignores_heredoc_bodies_and_comments() {
    assert_eq!(
        words("cat > /tmp/x.sh <<'EOF'\ngrep -r needle /\nEOF\n# grep -r needle /\nbash /tmp/x.sh"),
        vec![
            vec![literal("cat")],
            vec![literal("bash"), literal("/tmp/x.sh")],
        ]
    );
}

#[test]
fn lists_substitutions_in_unquoted_heredoc_bodies() {
    assert_eq!(
        words("cat <<EOF\nfound: $(rg -l needle /)\nEOF"),
        vec![
            vec![literal("cat")],
            vec![
                literal("rg"),
                literal("-l"),
                literal("needle"),
                literal("/")
            ],
        ]
    );
}

#[test]
fn records_the_subshells_each_command_runs_in() {
    let subshells: Vec<Vec<usize>> = commands("cd / && echo $(cd /tmp) && (cd /opt; ls) | wc")
        .into_iter()
        .map(|command| command.subshells)
        .collect();
    assert_eq!(
        subshells,
        vec![vec![], vec![], vec![0], vec![1], vec![1], vec![]]
    );
}

#[test]
fn still_lists_commands_around_a_syntax_error() {
    assert_eq!(
        words("grep -r needle / | head; echo 'unterminated"),
        vec![
            vec![
                literal("grep"),
                literal("-r"),
                literal("needle"),
                literal("/"),
            ],
            vec![literal("head")],
            vec![literal("echo")],
        ]
    );
}
