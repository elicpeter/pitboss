//! Prompt templates shipped in the binary via `include_str!`.
//!
//! `pitboss prompts new <name>` writes [`NEW_PROMPT_RAW`] with the `__NAME__`
//! placeholder replaced. Keeping the raw text in a module-level constant lets
//! tests assert against the verbatim file without re-reading from disk.

/// Raw template body for `pitboss prompts new`. Contains the literal token
/// `__NAME__` which callers replace with the validated prompt name before
/// writing to disk.
pub const NEW_PROMPT_RAW: &str = include_str!("prompts/templates/new_prompt.md");

/// Render the `New` template for a prompt named `name`. The caller is
/// responsible for validating `name` against the prompt-name regex; this
/// helper only performs literal substitution.
pub fn render_new_prompt(name: &str) -> String {
    NEW_PROMPT_RAW.replace("__NAME__", name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_is_substituted() {
        let rendered = render_new_prompt("fp-hunter");
        assert!(rendered.contains("name: fp-hunter"));
        assert!(!rendered.contains("__NAME__"));
    }

    #[test]
    fn rendered_template_round_trips_through_parser() {
        let rendered = render_new_prompt("triage");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("triage.md");
        std::fs::write(&path, &rendered).unwrap();
        let doc = crate::grind::parse_prompt_file(&path).expect("template should parse");
        assert_eq!(doc.meta.name, "triage");
        assert_eq!(doc.meta.weight, 1);
        assert_eq!(doc.meta.every, 1);
        assert!(!doc.meta.verify);
        assert!(!doc.meta.parallel_safe);
    }
}
