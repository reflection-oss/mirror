use pretty_assertions::assert_eq;

use super::*;

#[test]
fn native_tool_describes_every_argument() {
    let workspace = tempfile::tempdir().expect("workspace");

    let undescribed = ["bash", "read", "write", "edit", "grep", "glob", "ls"]
        .into_iter()
        .flat_map(|name| {
            let definition = native_tool(name, workspace.path())
                .expect("native tool")
                .definition();
            let properties = definition.parameters["properties"]
                .as_object()
                .expect("object properties")
                .clone();
            properties
                .into_iter()
                .filter(|(_, schema)| schema["description"].as_str().is_none_or(str::is_empty))
                .map(move |(argument, _)| format!("{name}.{argument}"))
        })
        .collect::<Vec<_>>();

    assert_eq!(undescribed, Vec::<String>::new());
}
