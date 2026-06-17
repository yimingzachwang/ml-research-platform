# Agent Draft Governance Compatibility Audit

**Phase:** 5F.3  
**Date:** 2026-06-11  
**Status:** Complete — no compatibility gaps found, no source changes required.

---

## 1. Purpose

This audit verifies that drafts created by the new agent workflow chain
are fully compatible with the existing governance pipeline. No new agent
features were added; this is a checkpoint-only phase.

---

## 2. Compatibility Chain Tested

```
Agent chain:
  ResearchSpec (LassoRegression, canonical_multi_asset_etf, 2015–2024)
  → resolve_research_spec()
  → create_experiment_plan()
  → propose_spec_from_next_action() [model_comparison, confirm_proposal=False first]
  → run_draft_from_spec_proposal(confirm_proposal=True)
  → save_agent_draft_result()  [persists to tmp_path as JSON]

Existing governance layer:
  load_draft()          [from src.orchestration.config_generation.draft_generator]
  validate_draft()      [uses configs/experiments/canonical_ml_multi_asset.yaml]
  approve_draft()       [pure state transition, no I/O]
  render_to_yaml(dry_run=True)  [YAML produced, not written to disk]
```

---

## 3. Results by Assertion Category

### A. Saved Draft Compatibility

| Check | Result |
|-------|--------|
| `experiment_draft_object` is present on `WorkflowResult` | ✓ Pass |
| `base_experiment == "canonical_ml_multi_asset"` | ✓ Pass |
| `approved == False` before any approval call | ✓ Pass |
| `changes` includes `(model, type, LassoRegression)` | ✓ Pass |
| `save_agent_draft_result()` returns `status="saved"` | ✓ Pass |
| JSON file written to `<llm_base>/<base_experiment>/draft_<id>.json` | ✓ Pass |
| JSON `approved == False` | ✓ Pass |
| `approval_required == True` on save result | ✓ Pass |
| `execution_performed == False` on save result | ✓ Pass |

### B. Load Compatibility

| Check | Result |
|-------|--------|
| `load_draft(base_experiment, draft_id, llm_base)` returns non-None | ✓ Pass |
| Loaded draft has correct `base_experiment` | ✓ Pass |
| Loaded draft has correct `draft_id` | ✓ Pass |
| Loaded draft is unapproved | ✓ Pass |
| Loaded draft model change preserved (LassoRegression) | ✓ Pass |
| `load_draft` returns None for bad draft_id | ✓ Pass |

### C. Validation Compatibility

| Check | Result |
|-------|--------|
| `validate_draft(loaded, base=tmp, configs_base=real)` returns `is_valid=True` | ✓ Pass |
| No validation errors | ✓ Pass |
| `(model, type)` is in `_VALID_CHANGE_PATHS` | ✓ Pass |
| Validation does not change `approved` state | ✓ Pass |

### D. Approval Compatibility

| Check | Result |
|-------|--------|
| `approve_draft()` returns `approved=True` | ✓ Pass |
| `approved_at` is set after approval | ✓ Pass |
| Original draft not mutated | ✓ Pass |
| Model change preserved after approval | ✓ Pass |
| `base_experiment` preserved after approval | ✓ Pass |
| Hash recomputed on approval | ✓ Pass |

### E. Render Compatibility

| Check | Result |
|-------|--------|
| `render_to_yaml(approved, dry_run=True)` produces non-empty YAML | ✓ Pass |
| Rendered YAML contains `LassoRegression` | ✓ Pass |
| Rendered YAML contains provenance header | ✓ Pass |
| Rendered YAML contains `draft_hash` | ✓ Pass |
| Rendered YAML contains `base_experiment` | ✓ Pass |
| Rendered YAML `model.type == LassoRegression` (parsed) | ✓ Pass |
| Rendered YAML `version == "2"` | ✓ Pass |
| Render raises `ValueError` on unapproved draft | ✓ Pass |
| `dry_run=True` does not write any config file | ✓ Pass |

### F. Execution Boundary

| Check | Result |
|-------|--------|
| `execution_performed == False` on `WorkflowResult` | ✓ Pass |
| `execution_performed == False` on `SavedAgentDraftResult` | ✓ Pass |
| `approve_draft()` does not trigger execution | ✓ Pass |
| `render_to_yaml(dry_run=True)` does not trigger execution | ✓ Pass |

### G. Governance Flags Throughout Chain

| Stage | `approval_required` | `draft_approved` | `execution_performed` | `requires_user_confirmation` |
|-------|---------------------|------------------|-----------------------|------------------------------|
| ResearchSpecProposal | True | — | False | True |
| Confirmation gate (confirm=False) | True | False | False | True |
| WorkflowResult (draft_created) | True | False | False | True |
| SavedAgentDraftResult | True | False | False | — |
| After approve_draft() | — | True | (no field on ExperimentDraft) | — |

---

## 4. No Compatibility Gaps Found

The existing governance pipeline accepts agent-created drafts without modification:

- The `ExperimentDraft` dataclass structure produced by the agent chain is
  identical to that produced by the traditional LLM iteration path.
- The `base_experiment = "canonical_ml_multi_asset"` correctly maps from the
  `canonical_multi_asset_etf` preset via `_PRESET_TO_CONFIG_NAME`.
- The `(model, type)` change path is in `_VALID_CHANGE_PATHS`.
- `validate_draft()` accepts the agent-created changes without errors.
- `approve_draft()` is a pure dataclass state transition that works on any
  `ExperimentDraft` regardless of origin.
- `render_to_yaml()` produces correct YAML with provenance metadata.

**No source code changes were required for this phase.**

---

## 5. Files Created

| File | Purpose |
|------|---------|
| `tests/agent/test_agent_draft_governance_compatibility.py` | 69 compatibility tests |
| `docs/agent_governance_compatibility_audit.md` | This document |

---

## 6. Test Results

| Suite | Result |
|-------|--------|
| `tests/agent/test_agent_draft_governance_compatibility.py` | **69 passed** |
| `tests/agent/` (full) | **1505 passed** |
| `tests/mcp_server/` | **356 passed** |
| `tests/orchestration/test_config_generation.py` | **80 passed** |
| `tests/experiments/test_ml_config.py` + `test_config_introspection.py` | **98 passed** |
| `tests/orchestration/` (full) | 699 passed, **59 pre-existing failures** (artefact-dependent, unchanged) |

The 59 pre-existing orchestration failures are in `test_retrieval.py` and one
`test_research_api.py` test. They depend on real experiment artefacts in
`results/experiments/` that do not exist in the test environment. These are
unchanged from prior phases.

---

## 7. Execution Gap Confirmed

Execution is a separate, explicitly gated step that is not reached at any
point in the compatibility chain. The execution path is:

```
render_to_yaml() → configs/experiments/<name>.yaml
→ scripts/run_from_config.py <path>  [explicit human invocation]
→ OR: execute_experiment() via MCP tool [explicit approved-config path]
```

This gate is separate from:
- Agent draft creation
- Draft saving
- Draft approval
- Draft rendering

The test `test_render_dry_run_does_not_write_file` confirms `dry_run=True`
never writes to disk, and none of the agent workflow functions call
`execute_experiment()` or any experiment runner.

---

## 8. Known Limitation: render_to_yaml requires configs/experiments base config

`render_to_yaml()` and `validate_draft()` require the base experiment YAML
(`configs/experiments/canonical_ml_multi_asset.yaml`) to exist on disk. This
means:

- The compatibility tests use `pytest.skip` if the base config is absent.
- For environments that don't have the real config (e.g., stripped CI), the
  tests are skipped rather than failing.
- This is the same constraint that applies to the traditional draft pipeline.

---

## 9. Recommendation for LM Studio / Qwen Live Testing

Phase 5F.3 is complete. The full governance chain is proven:

```
Qwen (local LLM) → intent extraction → ResearchSpec → resolve → plan
→ propose_spec → confirm_proposal=True → draft_created → save → load
→ validate → approve → render_to_yaml (dry_run=True)
```

The live Qwen path can be tested at any time using:

```bash
python scripts/demo_lmstudio_research_agent_loop.py \
    --base-url http://127.0.0.1:1234/v1 \
    --model qwen2.5-7b-instruct \
    --save-draft
```

The `--save-draft` flag will exercise the full save path. After saving, the
resulting draft JSON can be passed through the governance layer using the
functions verified in this audit.
