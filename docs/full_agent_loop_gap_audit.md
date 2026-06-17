# Full Agent Loop — End-to-End Integration Gap Audit

**Date:** 2026-06-10  
**Auditor:** Claude Code (automated + stub-mode live execution)  
**Scope:** Phase 1–5F agent chain, stub provider only  
**Tests at audit time:** 1752 passing (1396 agent + 356 mcp_server)

---

## 0. Executive Summary

The full agent loop was executed end-to-end in stub mode against the canonical scenario:

> *"Use the canonical multi-asset ETF preset from 2015 to 2024. What should we try next?"*

**One critical blocking gap was found and fixed during this audit.** Seven non-critical gaps were documented; all are either known planned work (Phase 3B/C, Phase 6) or working-as-designed behaviours.

The critical fix: `_derive_base_spec_from_resolved()` and `run_propose_spec_from_request()` were including hard-unsupported structural fields (`universe`, `tickers`, `asset_class`, `frequency`, `data_source`) in the derived `base_spec`. When the proposed spec was re-resolved, these fields appeared as `user_specified_fields` → `detect_hard_unsupported_fields` returned them → `unsupported_plan` instead of `draft_created`. Fixed by filtering those fields at derivation and construction time.

---

## 1. Scenario and Chain Execution

### 1.1 Canonical Scenario

```
Request: "Use the canonical multi-asset ETF preset from 2015 to 2024. What should we try next?"
Provider: stub
```

### 1.2 Step-by-step Results (post-fix)

| Step | Function | Status | Key Output |
|------|----------|--------|------------|
| 1 | `parse_research_intent` | `parsed` | `{preset: canonical_multi_asset_etf, start_date: 2015-01-01, end_date: 2024-12-31}` |
| 2 | `resolve_research_spec` | `resolved` | `user_specified=[preset, start_date, end_date]`, model=RidgeRegression |
| 3 | `create_experiment_plan` | `planned` | `experiment_type=preset_baseline`, baseline=canonical_multi_asset_etf |
| 4 | `retrieve_research_memory_context` | `retrieved` | empty (no artefacts in stub env) |
| 5 | `recommend_next_actions` | `needs_more_evidence` | candidates=[`repeat_baseline`, `diagnostic_review`] |
| 6 | Candidate selection | — | `repeat_baseline` (index 0, only non-advisory candidate) |
| 7 | `propose_spec_from_next_action` | `proposal_created` | `proposed_changes={}`, no unsupported |
| 8 | `run_draft_from_spec_proposal(confirm=False)` | `confirmation_required` | governance gate held ✓ |
| 9 | `run_draft_from_spec_proposal(confirm=True)` | `draft_created` | 5 mapped changes, `hard_unsupported=[]` ✓ |
| 10 | Serialisation | ok | `WorkflowResult.to_dict()` JSON-safe ✓ |

All steps chained cleanly after the fix.

---

## 2. Field Handoff Gaps

### 2.1 parse → resolve: Partial field preservation (working as designed)

`parse_research_intent` emits `ResearchSpec` with only the fields the parser extracted. The resolver expands missing fields from the preset and defaults. `user_specified_fields` is correctly populated with only what was in the original request.

**Status:** No gap.

### 2.2 resolve → plan: Resolution dict structure

`resolve_research_spec` returns a `WorkflowResult` with `resolution` dict containing `resolved_spec` (flat field dict) and `user_specified_fields`. `create_experiment_plan` correctly reads `resolution["resolved_spec"]` for the plan.

**Status:** No gap.

### 2.3 plan → memory: Artefact context unavailable in stub

`retrieve_research_memory_context` depends on experiment artefacts being present (run IDs, metric files). In stub mode, nothing has run, so memory returns empty. This is expected: memory context is only meaningful for real past experiments.

**Status:** Working as designed. Gap is environmental, not structural.

### 2.4 memory → recommend: Empty context path

When memory returns no artefacts, `recommend_next_actions` falls through to `needs_more_evidence` with a fixed candidate set. The handoff still works: `WorkflowResult.candidates` is populated.

**Status:** No gap.

### 2.5 recommend → propose: Candidate dict serialisation

`WorkflowResult.candidates` stores candidates as dicts (`to_dict()` output). `run_propose_spec_from_request` deserialises via `NextActionCandidate.from_dict(candidates[i])`. Round-trip verified — no field loss.

**Status:** No gap.

---

## 3. Serialisation Gaps

### 3.1 WorkflowResult.to_dict() JSON safety

Tested: `json.dumps(WorkflowResult.to_dict())` on all status paths:
- `parsed`, `resolved`, `planned`, `retrieved`, `needs_more_evidence`
- `proposal_created`, `confirmation_required`, `draft_created`
- `unsupported_proposal`, `unsupported_plan`

All status paths produce JSON-safe dicts. Nested objects (`ResearchSpec`, `ResearchSpecProposal`, `DraftChange`) all have stable `to_dict()` implementations.

**Status:** No gap.

### 3.2 ResearchSpecProposal round-trip

`ResearchSpecProposal.from_dict(proposal.to_dict())` verified: all fields preserved, governance invariants re-enforced by `__post_init__`. Tested in `test_proposal_to_draft_workflow.py` (TestResearchSpecProposalFromDict, 6 tests).

**Status:** No gap.

---

## 4. Unsupported-Field Consistency Gaps

### 4.1 [FIXED] Hard-unsupported fields from base_spec — critical blocking gap

**Root cause:** `_derive_base_spec_from_resolved()` included all `ResolvedResearchSpec` fields including hard-unsupported structural fields (`universe`, `tickers`, `asset_class`, `frequency`, `data_source`). Similarly, `run_propose_spec_from_request()` built `base_spec = ResearchSpec.from_dict(resolved_fields)` from the full resolution dict. Both caused these fields to appear as `user_specified_fields` in the re-resolved proposed spec, which then triggered `detect_hard_unsupported_fields` → `unsupported_plan`.

**Fix applied:**
- `src/agent/next_actions/candidate_to_spec.py` — `_derive_base_spec_from_resolved()`: added `if name in _HARD_UNSUPPORTED_SPEC_FIELDS: continue` in the field-copy loop.
- `src/agent/workflow/spec_proposal_workflow.py` — `run_propose_spec_from_request()`: added dict comprehension filtering `_HARD_UNSUPPORTED_SPEC_FIELDS` before `ResearchSpec.from_dict(filtered)`.

**Verification:** All paths now produce `draft_created`:
- `repeat_baseline` with derived base: `draft_created`, 5 mapped_changes
- `model_comparison` with derived base: `draft_created`, 6 mapped_changes including `model.type=LassoRegression`
- End-to-end `run_propose_spec_from_request → run_draft_from_spec_proposal`: `draft_created`

**Status:** Fixed. No regressions (1752 tests passing).

### 4.2 Soft-unsupported fields (start_date, end_date)

`start_date` and `end_date` appear as `user_specified_fields` in proposed specs built from a full resolved spec. They are correctly reported in `WorkflowResult.unsupported_fields` (non-blocking, non-error) and do not prevent `draft_created`. This is the designed behaviour: date ranges are experiment boundaries set at run time, not config params.

**Status:** Working as designed. The `unsupported_fields` list in the draft result correctly reports them.

### 4.3 Silently-skipped fields

`_SILENTLY_SKIPPED_SPEC_FIELDS` = `{preset, objective, benchmark, notes, diagnostics, risk_constraints, validation_protocol}`. These appear in `proposed_changes` (e.g., from `validation_robustness` action which includes `validation_protocol=expanding`) but produce no `DraftChange` entries and no warning.

**Gap:** `validation_protocol` being silently skipped means a workflow suggesting protocol change looks like it produced a partial change set. The mapped_changes output shows only the `train_window → parameters.train_months` mapping, not the protocol. This could mislead an operator reviewing the draft.

**Severity:** Low — no data is lost; the change is noted in the proposal's `proposed_changes` dict. A reviewer comparing `proposed_changes` vs `mapped_changes` would see the discrepancy.

**Recommended future work:** Add a `silently_skipped_fields` list to `WorkflowResult` (Phase 3C or Phase 6) analogous to `unsupported_fields`.

**Status:** Gap documented, not blocking, not fixed (outside current scope).

---

## 5. Base Config / Preset Mapping Gaps

### 5.1 Preset expansion fidelity

When `proposed_spec` uses `preset=canonical_multi_asset_etf` without hard-unsupported fields, the resolver expands it to the full configuration via `load_registry()`. All `mapped_changes` in the resulting draft reflect actual deltas from the preset's base config.

The `repeat_baseline` action with derived base produces 5 `mapped_changes` (label, transaction_cost_bps, model_params.alpha, validation.train_months, validation.test_months). These are fields the derived base carries over from the resolved spec as "user-specified" that technically came from the preset.

**Gap:** The 5 changes in a `repeat_baseline` draft look like the user is changing 5 things, but in reality they are all preset-identical values. This is a cosmetic/UX issue: the draft is technically correct (it specifies the same values as the preset) but the change-set appears noisy.

**Severity:** Low — the draft is functionally correct. The `repeat_baseline` case represents "run this again exactly as before" and the mapped changes are idempotent.

**Recommended future work:** Filter out mapped changes whose `proposed_value == base_config_value` as a no-op suppression pass, or use only `user_specified_fields` as the base spec for `repeat_baseline`.

**Status:** Gap documented, not blocking.

### 5.2 feature_families → feature_specs mapping (Phase 3B gap)

When `feature_ablation` proposes `feature_families: ["Trend"]`, the draft produces 0 mapped changes with a warning:

> "Base config has 13 feature(s) not in your specified set ... These were NOT removed — Phase 3B supports add-only feature changes."

The feature set cannot be narrowed via `proposed_changes["feature_families"]` in the current phase.

**Status:** Known Phase 3B gap. Not blocking for non-feature-ablation actions. Documented.

---

## 6. Memory-to-Candidate Gaps

### 6.1 Empty memory fallback

When `retrieve_research_memory_context` returns no artefacts, `recommend_next_actions` returns `needs_more_evidence` with a static candidate set (`repeat_baseline`, `diagnostic_review`). No error is raised, and the fallback candidate list is valid and actionable.

**Status:** No gap.

### 6.2 Memory context format contract

`retrieve_research_memory_context` returns `WorkflowResult.memory_context` as a dict. The `recommend_next_actions` function accepts this dict and uses it to score candidates. If the memory context is malformed (missing keys), `recommend_next_actions` has internal fallback logic.

**Status:** Verified robust in existing tests. No gap.

---

## 7. Candidate-to-Spec Gaps

### 7.1 Advisory candidates (diagnostic_review) produce no proposal

`diagnostic_review` action type is advisory — it has no `suggested_spec_changes` and no proposal handler. `propose_spec_from_next_action` returns `unsupported_proposal` with an informative message.

**Status:** Correct and expected. Operators should select non-advisory candidates (index 0 = `repeat_baseline` in the canonical scenario).

### 7.2 repeat_baseline produces empty proposed_changes

`repeat_baseline` produces `proposed_changes={}` with a warning: "repeat_baseline produces no spec changes — this proposes re-running the current baseline configuration." This is correct: the action is "run this again."

**Status:** Working as designed. The warning is clear.

### 7.3 model_comparison spec change resolution

`model_comparison` with `suggested_spec_changes={"model": "LassoRegression"}` correctly maps to `proposed_changes={"model": "LassoRegression"}` in the proposal, and to `DraftChange(field="type", proposed_value="LassoRegression")` in the draft. Registry validation confirms LassoRegression is a supported model.

**Status:** No gap.

---

## 8. Spec Proposal to Draft Gaps

### 8.1 [FIXED] See Section 4.1 above.

### 8.2 Governance gate integrity

`run_draft_from_spec_proposal(proposal, confirm_proposal=False)` → `confirmation_required`. Gate cannot be bypassed: `__post_init__` re-enforces `approval_required=True`, `requires_user_confirmation=True` regardless of input. Tested via `object.__setattr__` intrusion attempts — gate holds.

**Status:** No gap.

### 8.3 Draft fields excluded from DraftChange

Fields with no change-path handler (e.g., `validation_protocol`, `benchmark`, `objective`) appear in `proposed_changes` but not in `mapped_changes`. See Section 4.3 for the silently-skipped list.

**Status:** Working as designed.

---

## 9. Governance Gaps

### 9.1 Approval gates

| Invariant | Enforced? |
|-----------|-----------|
| `approval_required=True` on all proposals | Yes — `__post_init__` |
| `requires_user_confirmation=True` on all proposals | Yes — `__post_init__` |
| `creates_draft=False` on all proposals | Yes — `__post_init__` |
| `executes_experiment=False` on all proposals | Yes — `__post_init__` |
| `draft_approved=False` on all WorkflowResults | Yes — `WorkflowResult` default |
| `execution_performed=False` on all WorkflowResults | Yes — `WorkflowResult` default |
| `confirm_proposal=True` required for draft creation | Yes — `_validate_confirmation_gate` |

All governance invariants hold. Attempting to force-set them via `object.__setattr__` and then calling `run_draft_from_spec_proposal` still fails at `_validate_proposal_governance`.

**Status:** No gap.

### 9.2 MCP tool governance annotations

`create_validated_draft_from_spec_proposal` tool description includes explicit governance language (`_PD_GOVERNANCE_NOTE`). The tool returns `ok=False` when `confirmation_required`, `unsupported_proposal`, or `unsupported_plan`. `approval_required` and `requires_user_confirmation` flags are always `True` in the response.

**Status:** No gap.

### 9.3 save_draft isolation

`save_agent_draft_result` is only called when `status == "draft_created"` AND `save_draft=True`. Saving does not change `draft_approved` or `execution_performed`. The saved draft is a preview artefact only.

**Status:** No gap.

---

## 10. NL / Parser Gaps

### 10.1 Preset name recognition

`parse_research_intent("Use canonical multi-asset ETF preset from 2015 to 2024...")` correctly extracts `preset=canonical_multi_asset_etf`, `start_date=2015-01-01`, `end_date=2024-12-31`. No other fields in `user_specified_fields`.

**Status:** No gap for the canonical scenario.

### 10.2 Model name extraction

Stub parser does not extract explicit model names from free text. A request like "Try LassoRegression instead" would need to be expressed as a `NextActionCandidate` with `suggested_spec_changes={"model": "LassoRegression"}`, not via natural language alone (stub path).

**Status:** Known stub limitation. The production Qwen/LM Studio path handles model extraction via the LLM. Not tested in this audit per governance boundaries.

### 10.3 Missing-field fallback in parse

When the request includes no preset, model, or date, `parse_research_intent` returns a minimal `ResearchSpec` with only the fields it could extract. `resolve_research_spec` applies defaults. The chain still runs, producing a minimal but valid proposal.

**Status:** No gap.

---

## 11. Output: Required Fixes

| # | Severity | Location | Gap | Fix Status |
|---|----------|----------|-----|------------|
| 1 | **Critical** | `src/agent/next_actions/candidate_to_spec.py:_derive_base_spec_from_resolved` | Hard-unsupported fields included in derived base spec → `unsupported_plan` | **Fixed in this audit** |
| 2 | **Critical** | `src/agent/workflow/spec_proposal_workflow.py:run_propose_spec_from_request` | Same — base_spec built from full resolved dict without filtering hard-unsupported fields | **Fixed in this audit** |
| 3 | Low | `WorkflowResult` | No `silently_skipped_fields` list — fields like `validation_protocol` disappear without a trace in the `mapped_changes` output | Documented; Phase 6 or Phase 3C |
| 4 | Low | `repeat_baseline` draft | Idempotent mapped_changes (preset defaults re-appear as explicit changes) | Documented; cosmetic; not blocking |
| 5 | Known | `feature_ablation` | `feature_families` narrowing not supported — Phase 3B add-only | Documented; Phase 3B/C |

---

## 12. Audit Scenarios Executed

| Scenario | Candidate | Base Spec | Result | Status |
|----------|-----------|-----------|--------|--------|
| A | repeat_baseline | full resolved (unfixed) | `unsupported_plan` | Pre-fix failure |
| B | repeat_baseline | user_specified_only | `draft_created`, 0 changes | Passes |
| C | repeat_baseline | resolved minus hard-unsupported | `draft_created`, 5 changes | Passes |
| D | repeat_baseline | None, resolved_spec in WR | `unsupported_plan` | Pre-fix failure |
| E | model_comparison | clean user base | `draft_created`, 6 changes | Passes |
| F | model_comparison | user-specified-only | `draft_created`, 1 change (model.type) | Passes |
| G | validation_robustness | user base | `draft_created`, 1 change (train_months) | Passes; protocol silently skipped |
| H | feature_ablation | user base | `draft_created`, 0 changes + warning | Phase 3B gap |
| I | repeat_baseline (e2e) | via `run_propose_spec_from_request` (unfixed) | `unsupported_plan` | Pre-fix failure |
| I (fixed) | repeat_baseline (e2e) | via `run_propose_spec_from_request` (fixed) | `draft_created` | **Fixed** |

---

## 13. Test Coverage Summary

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `tests/agent/test_proposal_to_draft_workflow.py` | 47 | proposal-to-draft workflow, governance gate, serialisation |
| `tests/mcp_server/test_proposal_to_draft_tools.py` | 32 | MCP handler, all status paths, governance flags |
| `tests/agent/test_spec_proposal_workflow.py` | (existing) | run_propose_spec_from_request, candidate selection |
| `tests/agent/test_candidate_to_spec.py` | (existing) | _derive_base_spec_from_resolved, all action types |
| All agent tests | 1396 | full agent layer, post-fix passing |
| All mcp_server tests | 356 | full MCP layer, post-fix passing |

---

## 14. Governance Boundaries Observed

The following boundaries were respected throughout this audit:

- No new features implemented beyond the critical base_spec field-pollution fix required to make the already-built handoff work.
- No Qwen / LM Studio tested.
- No experiments executed.
- No drafts approved.
- No final YAML rendered.
- No quant engine modified.
- Validation not weakened.
- Governance gates not bypassed.

---

## 15. Next Steps (outside audit scope)

1. **Phase 3B/C** — feature set narrowing support (`feature_families` → remove existing features, not add-only).
2. **Phase 6** — `silently_skipped_fields` list in `WorkflowResult` for operator transparency.
3. **Qwen/LM Studio integration testing** — now unblocked; the full stub chain from NL request to unapproved draft is clean.
4. **Idempotent-change suppression** — optionally filter mapped changes where `proposed_value == base_config_value` in `repeat_baseline` to clean up draft output.
