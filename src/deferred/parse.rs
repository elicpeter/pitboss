//! Strict parser and serializer for `deferred.md`.
//!
//! `deferred.md` is the only artifact agents are permitted to write. After every
//! agent run the runner re-parses it; a parse failure triggers a snapshot
//! restore and halts the run. The parser is therefore intentionally narrow:
//! every kind of malformed input maps to a typed [`DeferredParseError`] that
//! carries a 1-based line number so error messages are actionable.
//!
//! The accepted shape is two H2 sections, in either order:
//!
//! ```markdown
//! ## Deferred items
//!
//! - [ ] open item
//! - [x] completed item
//!
//! ## Deferred phases
//!
//! ### From phase 07: rework agent trait
//!
//! Body of the replanned phase, preserved verbatim until the next H3 or H2.
//! ```
//!
//! Either section may be absent or empty. An empty file (or one containing only
//! whitespace) parses as [`super::DeferredDoc::empty`].
//!
//! [`serialize`] emits a canonical form: items first (if any), then a blank
//! line, then phases (if any). Phase bodies are reproduced verbatim. For inputs
//! already in canonical form, `serialize(parse(s)?) == s`.

use thiserror::Error;

use super::{DeferredDoc, DeferredItem, DeferredPhase};
use crate::plan::{PhaseId, PhaseIdParseError};

/// Errors produced by [`parse`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DeferredParseError {
    /// An H2 heading other than `## Deferred items` or `## Deferred phases`.
    #[error("deferred.md line {line}: unknown section header {raw:?}")]
    UnknownSection {
        /// 1-based line number of the offending heading.
        line: usize,
        /// The raw heading line (without trailing newline / whitespace).
        raw: String,
    },
    /// The same H2 section appeared twice.
    #[error("deferred.md line {line}: duplicate section {section:?}")]
    DuplicateSection {
        /// 1-based line number of the second occurrence.
        line: usize,
        /// Canonical name of the section (e.g., `## Deferred items`).
        section: String,
    },
    /// Non-blank content appeared before any section header.
    #[error("deferred.md line {line}: content before any section header: {raw:?}")]
    ContentBeforeSection {
        /// 1-based line number.
        line: usize,
        /// The offending line (without trailing newline).
        raw: String,
    },
    /// A line in `## Deferred items` was neither blank nor a `- [ ]` / `- [x]`
    /// checklist line.
    #[error("deferred.md line {line}: malformed checklist line: {raw:?}")]
    BadChecklistLine {
        /// 1-based line number.
        line: usize,
        /// The offending line (without trailing newline).
        raw: String,
    },
    /// An `### …` heading appeared outside `## Deferred phases`.
    #[error("deferred.md line {line}: H3 heading outside ## Deferred phases: {raw:?}")]
    H3OutsidePhases {
        /// 1-based line number.
        line: usize,
        /// The offending heading (without trailing newline).
        raw: String,
    },
    /// An H3 inside `## Deferred phases` did not match
    /// `### From phase <id>: <title>`.
    #[error("deferred.md line {line}: malformed phase heading {raw:?} (expected `### From phase <id>: <title>`)")]
    BadPhaseHeading {
        /// 1-based line number.
        line: usize,
        /// The offending heading (without trailing newline).
        raw: String,
    },
    /// The H3 was well-formed but the phase id did not parse.
    #[error("deferred.md line {line}: invalid phase id in heading: {source}")]
    BadPhaseHeadingId {
        /// 1-based line number.
        line: usize,
        /// Underlying parse error.
        #[source]
        source: PhaseIdParseError,
    },
    /// Non-blank content appeared in `## Deferred phases` before the first
    /// `### From phase …` heading.
    #[error("deferred.md line {line}: content in ## Deferred phases before first H3: {raw:?}")]
    ContentBeforeFirstPhase {
        /// 1-based line number.
        line: usize,
        /// The offending line (without trailing newline).
        raw: String,
    },
}

const ITEMS_HEADING: &str = "## Deferred items";
const PHASES_HEADING: &str = "## Deferred phases";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    None,
    Items,
    Phases,
}

/// Parse `deferred.md` content into a [`DeferredDoc`].
///
/// CRLF line endings are normalized to LF before parsing. An empty file (or one
/// containing only whitespace) returns [`DeferredDoc::empty`]. Trailing
/// whitespace on any line is tolerated. Section ordering is not enforced, but
/// each section may appear at most once.
pub fn parse(input: &str) -> Result<DeferredDoc, DeferredParseError> {
    let normalized = if input.contains('\r') {
        input.replace("\r\n", "\n")
    } else {
        input.to_string()
    };

    if normalized.trim().is_empty() {
        return Ok(DeferredDoc::empty());
    }

    let mut items: Vec<DeferredItem> = Vec::new();
    let mut phases: Vec<DeferredPhase> = Vec::new();

    let mut section = Section::None;
    let mut seen_items = false;
    let mut seen_phases = false;
    let mut current_phase: Option<(PhaseId, String, String)> = None;

    for (idx, line) in normalized.split_inclusive('\n').enumerate() {
        let line_no = idx + 1;
        let no_eol = line.strip_suffix('\n').unwrap_or(line);
        let trimmed = no_eol.trim_end();

        if let Some(rest) = trimmed.strip_prefix("## ") {
            if let Some((id, title, body)) = current_phase.take() {
                phases.push(DeferredPhase {
                    source_phase: id,
                    title,
                    body,
                });
            }

            match rest.trim() {
                "Deferred items" => {
                    if seen_items {
                        return Err(DeferredParseError::DuplicateSection {
                            line: line_no,
                            section: ITEMS_HEADING.to_string(),
                        });
                    }
                    seen_items = true;
                    section = Section::Items;
                }
                "Deferred phases" => {
                    if seen_phases {
                        return Err(DeferredParseError::DuplicateSection {
                            line: line_no,
                            section: PHASES_HEADING.to_string(),
                        });
                    }
                    seen_phases = true;
                    section = Section::Phases;
                }
                _ => {
                    return Err(DeferredParseError::UnknownSection {
                        line: line_no,
                        raw: trimmed.to_string(),
                    });
                }
            }
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("### ") {
            if section != Section::Phases {
                return Err(DeferredParseError::H3OutsidePhases {
                    line: line_no,
                    raw: trimmed.to_string(),
                });
            }

            if let Some((id, title, body)) = current_phase.take() {
                phases.push(DeferredPhase {
                    source_phase: id,
                    title,
                    body,
                });
            }

            let after = rest.strip_prefix("From phase ").ok_or_else(|| {
                DeferredParseError::BadPhaseHeading {
                    line: line_no,
                    raw: trimmed.to_string(),
                }
            })?;
            let (id_str, title) =
                after
                    .split_once(": ")
                    .ok_or_else(|| DeferredParseError::BadPhaseHeading {
                        line: line_no,
                        raw: trimmed.to_string(),
                    })?;
            let id =
                PhaseId::parse(id_str).map_err(|source| DeferredParseError::BadPhaseHeadingId {
                    line: line_no,
                    source,
                })?;

            current_phase = Some((id, title.to_string(), String::new()));
            continue;
        }

        match section {
            Section::None => {
                if !trimmed.is_empty() {
                    return Err(DeferredParseError::ContentBeforeSection {
                        line: line_no,
                        raw: trimmed.to_string(),
                    });
                }
            }
            Section::Items => {
                if trimmed.is_empty() {
                    continue;
                }
                items.push(parse_checklist_line(line_no, trimmed)?);
            }
            Section::Phases => {
                if let Some((_, _, body)) = current_phase.as_mut() {
                    body.push_str(line);
                } else if !trimmed.is_empty() {
                    return Err(DeferredParseError::ContentBeforeFirstPhase {
                        line: line_no,
                        raw: trimmed.to_string(),
                    });
                }
            }
        }
    }

    if let Some((id, title, body)) = current_phase.take() {
        phases.push(DeferredPhase {
            source_phase: id,
            title,
            body,
        });
    }

    Ok(DeferredDoc { items, phases })
}

fn parse_checklist_line(line_no: usize, raw: &str) -> Result<DeferredItem, DeferredParseError> {
    let bad = || DeferredParseError::BadChecklistLine {
        line: line_no,
        raw: raw.to_string(),
    };

    let rest = raw.strip_prefix("- [").ok_or_else(bad)?;
    let mut chars = rest.chars();
    let mark = chars.next().ok_or_else(bad)?;
    if chars.next() != Some(']') {
        return Err(bad());
    }
    let done = match mark {
        ' ' => false,
        'x' | 'X' => true,
        _ => return Err(bad()),
    };

    // Position of the byte immediately after `]`. `- [` is 3 bytes; mark is 1
    // byte (always ASCII per the match above); `]` is 1 byte.
    let after_close = "- [".len() + mark.len_utf8() + 1;
    let tail = &raw[after_close..];
    let text = if tail.is_empty() {
        String::new()
    } else {
        // Require at least one space between `]` and the text. An entirely
        // empty trailing region was handled above; a non-space first character
        // means the line is malformed.
        if !tail.starts_with(' ') {
            return Err(bad());
        }
        tail.trim_start().to_string()
    };

    Ok(DeferredItem { text, done })
}

/// Serialize a [`DeferredDoc`] back to `deferred.md` text.
///
/// Canonical form: items section (if non-empty), a blank-line separator, phases
/// section (if non-empty). An empty doc serializes to the empty string. Always
/// emits LF line endings.
pub fn serialize(doc: &DeferredDoc) -> String {
    if doc.items.is_empty() && doc.phases.is_empty() {
        return String::new();
    }

    let mut out = String::new();

    if !doc.items.is_empty() {
        out.push_str(ITEMS_HEADING);
        out.push_str("\n\n");
        for item in &doc.items {
            let mark = if item.done { 'x' } else { ' ' };
            if item.text.is_empty() {
                out.push_str(&format!("- [{}]\n", mark));
            } else {
                out.push_str(&format!("- [{}] {}\n", mark, item.text));
            }
        }
    }

    if !doc.phases.is_empty() {
        if !doc.items.is_empty() {
            out.push('\n');
        }
        out.push_str(PHASES_HEADING);
        out.push_str("\n\n");
        for phase in &doc.phases {
            out.push_str(&format!(
                "### From phase {}: {}\n",
                phase.source_phase, phase.title
            ));
            out.push_str(&phase.body);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pid(s: &str) -> PhaseId {
        PhaseId::parse(s).expect("test id must parse")
    }

    #[test]
    fn empty_input_parses_as_empty_doc() {
        assert_eq!(parse("").unwrap(), DeferredDoc::empty());
        assert_eq!(parse("   \n\n  \n").unwrap(), DeferredDoc::empty());
    }

    #[test]
    fn empty_doc_serializes_to_empty_string() {
        assert_eq!(serialize(&DeferredDoc::empty()), "");
    }

    #[test]
    fn items_only_round_trip() {
        let s = "## Deferred items\n\n- [ ] first\n- [x] second\n";
        let doc = parse(s).unwrap();
        assert_eq!(doc.items.len(), 2);
        assert_eq!(doc.items[0].text, "first");
        assert!(!doc.items[0].done);
        assert_eq!(doc.items[1].text, "second");
        assert!(doc.items[1].done);
        assert!(doc.phases.is_empty());
        assert_eq!(serialize(&doc), s);
    }

    #[test]
    fn phases_only_round_trip() {
        let s =
            "## Deferred phases\n\n### From phase 07: rework agent trait\n\nbody line\n- bullet\n";
        let doc = parse(s).unwrap();
        assert!(doc.items.is_empty());
        assert_eq!(doc.phases.len(), 1);
        assert_eq!(doc.phases[0].source_phase, pid("07"));
        assert_eq!(doc.phases[0].title, "rework agent trait");
        assert_eq!(doc.phases[0].body, "\nbody line\n- bullet\n");
        assert_eq!(serialize(&doc), s);
    }

    #[test]
    fn both_sections_round_trip() {
        let s = "## Deferred items\n\
                 \n\
                 - [ ] open\n\
                 - [x] done\n\
                 \n\
                 ## Deferred phases\n\
                 \n\
                 ### From phase 03: parser hardening\n\
                 \n\
                 More detail.\n\
                 \n\
                 ### From phase 12: runner cleanup\n\
                 \n\
                 Tail body.\n";
        let doc = parse(s).unwrap();
        assert_eq!(doc.items.len(), 2);
        assert_eq!(doc.phases.len(), 2);
        assert_eq!(doc.phases[0].source_phase, pid("03"));
        assert_eq!(doc.phases[0].title, "parser hardening");
        assert_eq!(doc.phases[0].body, "\nMore detail.\n\n");
        assert_eq!(doc.phases[1].source_phase, pid("12"));
        assert_eq!(doc.phases[1].title, "runner cleanup");
        assert_eq!(doc.phases[1].body, "\nTail body.\n");
        assert_eq!(serialize(&doc), s);
    }

    #[test]
    fn empty_sections_round_trip() {
        // Items section header alone serializes back to empty (canonical form
        // omits empty sections).
        let doc = DeferredDoc::empty();
        assert_eq!(serialize(&doc), "");
        // A populated items section with empty phases section input parses
        // through canonical form.
        let s = "## Deferred items\n\n- [ ] only\n";
        assert_eq!(serialize(&parse(s).unwrap()), s);
    }

    #[test]
    fn tolerates_blank_lines_and_trailing_whitespace() {
        let s = "## Deferred items   \n\
                 \n\
                 - [ ] first   \n\
                 \n\
                 - [x] second\n\
                 \n";
        let doc = parse(s).unwrap();
        assert_eq!(doc.items.len(), 2);
        assert_eq!(doc.items[0].text, "first");
        assert_eq!(doc.items[1].text, "second");
    }

    #[test]
    fn capital_x_accepted_normalizes_to_lowercase_on_serialize() {
        let s = "## Deferred items\n\n- [X] done\n";
        let doc = parse(s).unwrap();
        assert!(doc.items[0].done);
        assert_eq!(serialize(&doc), "## Deferred items\n\n- [x] done\n");
    }

    #[test]
    fn empty_checkbox_text_round_trips() {
        let s = "## Deferred items\n\n- [ ]\n- [x]\n";
        let doc = parse(s).unwrap();
        assert_eq!(doc.items.len(), 2);
        assert_eq!(doc.items[0].text, "");
        assert_eq!(doc.items[1].text, "");
        assert_eq!(serialize(&doc), s);
    }

    #[test]
    fn rejects_unknown_h2_section() {
        let err = parse("## Random\n\nstuff\n").unwrap_err();
        assert!(matches!(
            err,
            DeferredParseError::UnknownSection { line: 1, .. }
        ));
    }

    #[test]
    fn rejects_duplicate_items_section() {
        let s = "## Deferred items\n\n- [ ] a\n\n## Deferred items\n\n- [ ] b\n";
        let err = parse(s).unwrap_err();
        assert!(matches!(
            err,
            DeferredParseError::DuplicateSection { line: 5, .. }
        ));
    }

    #[test]
    fn rejects_duplicate_phases_section() {
        let s = "## Deferred phases\n\n### From phase 01: a\n\n## Deferred phases\n";
        let err = parse(s).unwrap_err();
        assert!(matches!(
            err,
            DeferredParseError::DuplicateSection { line: 5, .. }
        ));
    }

    #[test]
    fn rejects_content_before_section() {
        let err = parse("intro paragraph\n\n## Deferred items\n").unwrap_err();
        assert!(matches!(
            err,
            DeferredParseError::ContentBeforeSection { line: 1, .. }
        ));
    }

    #[test]
    fn rejects_h3_outside_phases() {
        let s = "## Deferred items\n\n### From phase 01: x\n";
        let err = parse(s).unwrap_err();
        assert!(matches!(
            err,
            DeferredParseError::H3OutsidePhases { line: 3, .. }
        ));
    }

    #[test]
    fn rejects_h3_before_any_section() {
        let err = parse("### From phase 01: x\n").unwrap_err();
        assert!(matches!(
            err,
            DeferredParseError::H3OutsidePhases { line: 1, .. }
        ));
    }

    #[test]
    fn rejects_malformed_phase_heading() {
        let s = "## Deferred phases\n\n### bogus heading\n";
        let err = parse(s).unwrap_err();
        assert!(matches!(
            err,
            DeferredParseError::BadPhaseHeading { line: 3, .. }
        ));
    }

    #[test]
    fn rejects_phase_heading_without_colon() {
        let s = "## Deferred phases\n\n### From phase 07 missing-colon\n";
        let err = parse(s).unwrap_err();
        assert!(matches!(
            err,
            DeferredParseError::BadPhaseHeading { line: 3, .. }
        ));
    }

    #[test]
    fn rejects_invalid_phase_id_in_heading() {
        let s = "## Deferred phases\n\n### From phase abc: oops\n";
        let err = parse(s).unwrap_err();
        assert!(matches!(
            err,
            DeferredParseError::BadPhaseHeadingId { line: 3, .. }
        ));
    }

    #[test]
    fn rejects_content_in_phases_before_first_h3() {
        let s = "## Deferred phases\n\nstray text\n\n### From phase 01: x\n";
        let err = parse(s).unwrap_err();
        assert!(matches!(
            err,
            DeferredParseError::ContentBeforeFirstPhase { line: 3, .. }
        ));
    }

    #[test]
    fn rejects_malformed_checklist_line() {
        let cases = [
            ("## Deferred items\n\nrandom prose\n", 3),
            ("## Deferred items\n\n- [?] bad mark\n", 3),
            ("## Deferred items\n\n- [ x] extra space inside\n", 3),
            ("## Deferred items\n\n- [x]no-space-before-text\n", 3),
            ("## Deferred items\n\n* [ ] wrong bullet\n", 3),
        ];
        for (input, expected_line) in cases {
            let err = parse(input).unwrap_err();
            match err {
                DeferredParseError::BadChecklistLine { line, .. } => {
                    assert_eq!(line, expected_line, "input: {:?}", input);
                }
                other => panic!("expected BadChecklistLine for {:?}, got {:?}", input, other),
            }
        }
    }

    #[test]
    fn normalizes_crlf_to_lf() {
        let crlf =
            "## Deferred items\r\n\r\n- [ ] hi\r\n\r\n## Deferred phases\r\n\r\n### From phase 01: x\r\n\r\nbody\r\n";
        let doc = parse(crlf).unwrap();
        assert_eq!(doc.items.len(), 1);
        assert_eq!(doc.phases.len(), 1);
        let out = serialize(&doc);
        assert!(!out.contains('\r'));
    }

    #[test]
    fn parse_serialize_idempotent_on_canonical_output() {
        // serialize(parse(serialize(d))) == serialize(d) for any d.
        let doc = DeferredDoc {
            items: vec![
                DeferredItem {
                    text: "polish error message".into(),
                    done: false,
                },
                DeferredItem {
                    text: "remove unused stub".into(),
                    done: true,
                },
            ],
            phases: vec![
                DeferredPhase {
                    source_phase: pid("07"),
                    title: "rework agent trait".into(),
                    body: "\nSome body\n- bullet\n\n".into(),
                },
                DeferredPhase {
                    source_phase: pid("10b"),
                    title: "follow-up".into(),
                    body: "\ntail.\n".into(),
                },
            ],
        };
        let once = serialize(&doc);
        let twice = serialize(&parse(&once).unwrap());
        assert_eq!(once, twice);
    }

    #[test]
    fn phases_in_reverse_order_in_input_round_trip_canonicalizes() {
        // Phases section appearing before items is accepted; canonical output
        // moves items first.
        let s =
            "## Deferred phases\n\n### From phase 02: a\nbody\n\n## Deferred items\n\n- [ ] x\n";
        let doc = parse(s).unwrap();
        let canonical = serialize(&doc);
        assert!(canonical.starts_with("## Deferred items\n"));
        // Re-parse is stable.
        let doc2 = parse(&canonical).unwrap();
        assert_eq!(doc, doc2);
    }
}
