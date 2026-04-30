//! `deferred.md` domain types and (de)serialization.
//!
//! Phase 2 introduced the type vocabulary; phase 4 adds parsing, serialization,
//! sweep, and snapshot/verify utilities. Snapshot/verify are re-exported from
//! [`crate::plan`] because the underlying SHA-256 file hash is generic — both
//! `plan.md` and `deferred.md` use the same bytes-on-disk integrity check.

use serde::{Deserialize, Serialize};

use crate::plan::PhaseId;

mod parse;

pub use crate::plan::{snapshot, verify_unchanged, Snapshot, SnapshotError};
pub use parse::{parse, serialize, DeferredParseError};

/// A single checkbox item under `## Deferred items`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeferredItem {
    /// The text following the checkbox marker.
    pub text: String,
    /// `true` if the box is checked. Swept by [`DeferredDoc::sweep`] in phase 4.
    pub done: bool,
}

/// A `### From phase <id>: <title>` block under `## Deferred phases`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeferredPhase {
    /// Phase id this replan was emitted from.
    pub source_phase: PhaseId,
    /// Title text after the colon in the H3 heading.
    pub title: String,
    /// Raw markdown body following the heading line, preserved verbatim.
    pub body: String,
}

/// Parsed `deferred.md`: pending checklist items plus replanned phase blocks.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeferredDoc {
    /// Checklist items from `## Deferred items`.
    pub items: Vec<DeferredItem>,
    /// Replan blocks from `## Deferred phases`.
    pub phases: Vec<DeferredPhase>,
}

impl DeferredDoc {
    /// An empty document — equivalent to a missing `deferred.md`.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Drop every checklist item with `done: true`. Phase blocks are left
    /// untouched. Called by the runner between phases so completed items don't
    /// pile up across runs.
    pub fn sweep(&mut self) {
        self.items.retain(|item| !item.done);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pid(s: &str) -> PhaseId {
        PhaseId::parse(s).unwrap()
    }

    #[test]
    fn empty_is_default() {
        assert_eq!(DeferredDoc::empty(), DeferredDoc::default());
        let doc = DeferredDoc::empty();
        assert!(doc.items.is_empty());
        assert!(doc.phases.is_empty());
    }

    #[test]
    fn sweep_removes_done_items_only() {
        let mut doc = DeferredDoc {
            items: vec![
                DeferredItem {
                    text: "keep".into(),
                    done: false,
                },
                DeferredItem {
                    text: "drop".into(),
                    done: true,
                },
                DeferredItem {
                    text: "also keep".into(),
                    done: false,
                },
            ],
            phases: vec![DeferredPhase {
                source_phase: pid("07"),
                title: "untouched".into(),
                body: String::new(),
            }],
        };
        doc.sweep();
        assert_eq!(
            doc.items,
            vec![
                DeferredItem {
                    text: "keep".into(),
                    done: false,
                },
                DeferredItem {
                    text: "also keep".into(),
                    done: false,
                },
            ]
        );
        assert_eq!(doc.phases.len(), 1);
    }

    #[test]
    fn snapshot_re_exports_detect_drift_on_deferred_md() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deferred.md");
        std::fs::write(&path, "## Deferred items\n\n- [ ] x\n").unwrap();
        let snap = snapshot(&path).unwrap();
        verify_unchanged(&path, &snap).unwrap();
        std::fs::write(&path, "## Deferred items\n\n- [x] x\n").unwrap();
        let err = verify_unchanged(&path, &snap).unwrap_err();
        assert!(matches!(err, SnapshotError::Mismatch { .. }));
    }

    #[test]
    fn serde_round_trip() {
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
            phases: vec![DeferredPhase {
                source_phase: pid("07"),
                title: "rework agent trait".into(),
                body: "Some body text\n- bullet\n".into(),
            }],
        };
        let json = serde_json::to_string(&doc).unwrap();
        let back: DeferredDoc = serde_json::from_str(&json).unwrap();
        assert_eq!(doc, back);
    }
}
