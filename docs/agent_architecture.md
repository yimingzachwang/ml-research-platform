# Governed AI Quant Research Agent Architecture

**Version:** Phase 4B (2026-06-10)
**Status:** Current canonical architecture reference.

---

## 1. Executive Summary

The platform implements a single, governed AI research agent architecture that spans two
cooperating source layers: `src/agent/` (research-design intelligence) and
`src/orchestration/` (governed lifecycle and operations). These are not competing systems.
They are distinct layers of one coherent whole, connected through `src/mcp/` (safe tool
exposure) and governed by explicit human approval gates at every action boundary.

The architecture is designed for **reproducible, auditable AI-assisted research** — not
autonomous trading or unchecked automation. Every consequential action requires a human
decision. The LLM's role is to extract intent, generate structured proposals, and interpret
evidence. The deterministic pipeline validates, resolves, plans, and drafts. The researcher
approves or rejects.

---

## 2. One Agent System, Multiple Layers

There are not two competing agent systems in this repository. The codebase is partitioned
into four cooperating layers:

```
┌────────────────────────────────────────────────────────────────────┐
│  src/mcp/                  Safe Tool Exposure Layer                │
│  44 MCP tools · envelope contract · governance flags always on     │
│  Delegates to src/agent/ and src/orchestration/ — no logic here    │
├────────────────────────────────────────────────────────────────────┤
│  src/agent/                Research Design Intelligence Core       │
│  registry · ResearchSpec · NL intent parser · resolver             │
│  compatibility critic · experiment planner · plan critic           │
│  plan-to-draft mapping · natural-language workflow façade          │
├────────────────────────────────────────────────────────────────────┤
│  src/orchestration/        Governance and Lifecycle Layer          │
│  sessions · reviews · comparisons · draft validation               │
│  draft persistence · approval gates · execution gates              │
│  artefact handling · research memory lifecycle                     │
├────────────────────────────────────────────────────────────────────┤
│  Quant Research Engine     Execution Backend (untouched by agent)  │
│  data · features · models · strategies                             │
│  validation · diagnostics · reporting                              │
└────────────────────────────────────────────────────────────────────┘
```

`src/agent/` and `src/orchestration/` are separate layers of one governed research agent
system. They do not duplicate each other. They are not alternatives to each other.

---

## 3. Layer Responsibilities

### `src/agent/` — Research Design Intelligence Core

`src/agent/` answers the question: **"what should the next experiment look like?"**

It accepts an incomplete research intent (natural language or structured spec), resolves it
against a validated registry of supported options, checks it for compatibility problems,
constructs a detailed experiment plan, and produces a validated draft configuration change —
all deterministically, without LLM involvement in the resolution or planning steps.

**Modules:**

| Subpackage | Responsibility |
|---|---|
| `registry/` | `ResearchConfigRegistry` — canonical supported values (models, features, universes, presets, signals, labels, validation protocols, diagnostics) loaded from `config/registry/` YAML |
| `intent/` | `ResearchSpec` (all-Optional 25-field dataclass) · `parse_research_intent()` — LLM-based extraction constrained to registry values; `parse_and_resolve_research_intent()` |
| `resolver/` | `resolve_research_spec()` — deterministic field completion using defaults, preset expansion, and mode policy (strict / recommended / demo); produces `ResolvedResearchSpec` |
| `critic/` | `check_compatibility()` — 11 rule checks on the resolved spec (e.g. panel signals on single-asset universe → invalid); produces `CompatibilityResult` |
| `planner/` | `create_experiment_plan()` — deterministic `ExperimentPlan` from a resolved spec; `critique_experiment_plan()` — 11 plan-critic checks (PLAN_A–K) |
| `drafting/` | `create_draft_from_plan()` — maps user-specified fields to `DraftChanges` against a base experiment config; detects hard-unsupported fields (block) and soft-unsupported fields (warn) |
| `workflow/` | `run_resolve_spec()`, `run_plan_from_spec()`, `run_draft_from_spec()` (structured-spec façade) · `run_resolve_from_request()`, `run_plan_from_request()`, `run_draft_from_request()` (NL façade) · `save_agent_draft_result()` (Phase 3D persistence bridge) |

**Governance invariants (src/agent/ enforces these):**
- The LLM extracts intent only. All research decisions (field defaults, compatibility, plan structure, draft mapping) are deterministic.
- `save_draft=True` must be explicitly passed — never inferred from natural language.
- `execution_performed: false` always. `draft_approved: false` always.
- Hard-unsupported fields block draft creation. Soft-unsupported fields are reported but do not block.
- `approval_required: true` on every draft result.

### `src/orchestration/` — Governance and Lifecycle Layer

`src/orchestration/` answers the question: **"what is the state of the research, and what
should be recorded, reviewed, or acted on?"**

It manages the session lifecycle, drives LLM review of artefacts, generates iteration
proposals, routes operation-level natural-language commands, persists comparison evidence,
and enforces approval and execution gates. It does not design experiments — that is
`src/agent/`'s job.

**Subsystems:**

| Subsystem | Responsibility |
|---|---|
| `session/` | Typed research session lifecycle — create, record events, summarise, persist |
| `context/` | Build structured `LLMContext` from experiment artefacts; deterministic failure mode detection |
| `llm/` | `call_llm()` vendor abstraction (Anthropic, OpenAI, stub); `run_llm_review()` with prompt grounding constraints; provider enum constants |
| `api/research_api.py` | All governed research operations (reviews, comparisons, draft generation, approval, rendering, execution) |
| `intents/` | Operation-level NL intent parsing → typed `Intent` dataclasses (see §5 below) |
| `router/` | `route(intent) → WorkflowResult` — dispatches typed intents to research_api; explicit prohibition on autonomous loops |
| `memory/` | Research memory indexing and retrieval (keyword and semantic) |
| `config_generation/` | `ExperimentDraft` schema, `draft_validator.py`, `approve_experiment_draft()`, `render_draft_to_yaml()` |

**Governance invariants (src/orchestration/ enforces these):**
- Human approval required before `render_draft_to_yaml()` (`approved=False` raises `ValueError`).
- No automated flow calls `approve_experiment_draft()`.
- No `RunExperimentIntent` exists. Execution requires `confirmation="RUN"` from the researcher.
- Router docstring: *"DO NOT introduce autonomous execution loops. DO NOT call quant engine functions directly."*
- All LLM prompts include explicit CRITICAL CONSTRAINTS sections prohibiting invented values, investment advice, and causal overclaiming.

### `src/mcp/` — Safe Tool Exposure Layer

`src/mcp/zeto_server.py` exposes 44 MCP tools to Claude/MCP clients. Every tool returns
an `_envelope` contract (`ok`, `stage`, `display`, `data`, `next_suggested_action`).

The MCP layer:
- delegates research-design operations to `src/agent/workflow/`
- delegates lifecycle operations to `src/orchestration/api/research_api`
- does not contain business logic of its own
- enforces governance flags in every response (`approval_required`, `execution_performed`, `draft_approved`)

**Phase 3C–4B tools (src/agent/ backed):**

| Tool | Phase | Purpose |
|---|---|---|
| `list_research_agent_options` | 3C | Registry listing |
| `resolve_research_spec` | 3C | Structured spec → resolution |
| `create_experiment_plan_from_spec` | 3C | Structured spec → plan |
| `create_validated_draft_from_spec` | 3C/3D | Full chain: spec → draft (+ optional save) |
| `parse_research_intent` | 4A | NL → ResearchSpec (registry-constrained) |
| `resolve_research_request` | 4B | NL → parse → resolve |
| `create_experiment_plan_from_request` | 4B | NL → parse → resolve → plan |
| `create_validated_draft_from_request` | 4B | NL → parse → resolve → plan → draft |

### Quant Research Engine — Execution Backend

`src/experiments/`, `src/ml/`, `src/backtesting/`, `src/features/`, `src/data/`,
`src/reporting/`, `src/strategies/`, `src/validation/`, `src/visualization/`.

The quant engine is the authoritative source of experiment results. The agent layer is
built on top of it, not inside it. The agent layer never modifies engine behaviour,
experiment configs, or validation logic.

---

## 4. Why `src/agent/` and `src/orchestration/` Are Not Competing Systems

A common architectural concern when reading this codebase for the first time: *"are there
two agent systems here?"* The answer is no.

The two layers operate at different levels of abstraction and are activated at different
points in the research lifecycle:

| Question | `src/agent/` | `src/orchestration/` |
|---|---|---|
| What it resolves | What the next experiment should look like | What the current research state is |
| Primary input | Research intent (NL or structured spec) | Experiment name / session ID |
| Primary output | Validated draft config change | Review, proposal, comparison, session event |
| LLM role | Intent extraction (constrained) | Evidence interpretation (constrained) |
| Deterministic core | Resolver, critic, planner, drafter | Failure mode detector, context builder, diff computation |
| Human gate | Draft approval before render | Approval confirmation before execution |
| When it runs | When a researcher wants to design a new experiment | When a researcher wants to analyse or act on existing experiments |

They compose naturally: a researcher uses `src/orchestration/` to review an existing
experiment and generate an iteration proposal, then uses `src/agent/` to resolve that
proposal into a new validated draft. The two layers pass the baton; they do not race.

---

## 5. Difference Between the Two Intent Systems

There are two intent parsers in the codebase. They serve entirely different purposes.

### Orchestration Intents (`src/orchestration/intents/`)

**Purpose:** Parse operation-level user commands — what lifecycle action to take on an
existing experiment.

**Produces:** Typed `Intent` dataclasses:

| Intent | Meaning |
|---|---|
| `ReviewExperimentIntent` | Generate an LLM review of experiment artefacts |
| `CompareExperimentsIntent` | Run a comparative analysis of two experiments |
| `GenerateIterationIntent` | Generate a next-step iteration proposal |
| `BuildEvolutionChainIntent` | Trace the evolution of an experiment family |
| `ListExperimentsIntent` | List experiments (optionally filtered) |
| `RankExperimentsIntent` | Rank experiments by performance |
| `RetrieveArtefactIntent` | Retrieve a specific experiment artefact |
| `BuildContextIntent` | Build structured LLM context for an experiment |
| `GenerateDraftIntent` | Generate an experiment draft from a review |
| `UnrecognisedIntent` | Fallback when intent cannot be classified |

**Dispatch:** `workflow_router.route(intent)` → `research_api.*`

There is deliberately no `RunExperimentIntent` or `ExecuteIntent` in this schema.

### ResearchSpec Intent Parser (`src/agent/intent/`)

**Purpose:** Parse research-design intent — what the next experiment should be.

**Produces:** `ResearchSpec` — a structured 25-field research specification containing:

```
preset / asset_class / universe / model / label / signal
start_date / end_date / features / feature_families / feature_specs
validation_protocol / train_window / test_window
transaction_cost_bps / benchmark / objective / notes
diagnostics / risk_constraints / data_source / frequency / tickers
```

**How it works:** An LLM extracts values from natural language, constrained to registry-
supported values only. Unsupported values go to `unrecognised_terms`, never to the spec.
The resulting `ResearchSpec` is then passed through the deterministic resolver, critic,
planner, and drafter.

**The two parsers are not interchangeable.** Orchestration intents answer *"do what"*;
ResearchSpec intent answers *"design what"*.

---

## 6. Current End-to-End Capabilities (Phase 4B)

```
Natural-language research request
  ↓ parse_research_intent()         [LLM extraction, registry-constrained]
  ↓ resolve_research_spec()         [deterministic: defaults, preset expansion, mode policy]
  ↓ check_compatibility()           [deterministic: 11 compatibility rules]
  ↓ create_experiment_plan()        [deterministic: ExperimentPlan + experiment_type inference]
  ↓ critique_experiment_plan()      [deterministic: 11 plan-critic checks PLAN_A–K]
  ↓ create_draft_from_plan()        [deterministic: DraftChanges against base config]
  ↓ validate_draft()                [deterministic: change-path vocabulary check]
  ↓ [optional] save_agent_draft_result()   [explicit save_draft=True only]
  ↓
  STOP — human approval required
  ↓
  approve_experiment_draft()        [human explicitly confirms]
  ↓
  render_draft_to_yaml()            [writes config file]
  ↓
  [researcher manually triggers execution]
```

All steps up to and including `validate_draft()` are deterministic. The LLM is involved
only in the first step (intent extraction) and in generating supporting evidence/proposals
from artefacts through `src/orchestration/`.

---

## 7. Governance Boundaries

The agent must not, under any circumstances:

- Approve drafts automatically
- Execute experiments automatically
- Bypass `validate_draft()` or `check_compatibility()`
- Write final experiment YAML directly from LLM output
- Apply fields the deterministic mapping does not support (hard-unsupported fields block; soft-unsupported fields warn and are excluded)
- Silently ignore unsupported fields — they must appear in `unsupported_fields` or `unrecognised_terms`
- Infer `save_draft=True` from natural language — only explicit `save_draft=True` triggers persistence
- Treat `approval_required` as optional — it is always `true` when a draft is present

These boundaries are enforced in code (not just by convention) and verified by the test
suite. Governance flags (`approval_required`, `execution_performed`, `draft_approved`) are
present in every `WorkflowResult.to_dict()` response.

---

## 8. Current Limitations

These are known, intentional constraints that define the scope of Phase 4B:

- **No autonomous closed-loop research.** The agent proposes; the researcher decides. There is no loop that automatically runs the next experiment based on prior results.
- **No execution from natural language.** Execution keywords (`run`, `execute`, `deploy`, `trade`) in a natural-language request trigger a warning and are never acted on.
- **Universe and date-range changes are not fully draft-mappable.** Changing `universe`, `tickers`, `frequency`, `data_source`, or `asset_class` is classified as hard-unsupported and blocks draft creation. These require direct config editing.
- **Unsupported models are rejected.** Models not present in `config/registry/models.yaml` (e.g. `RandomForestRegressor`, `GBM`) are rejected at the registry validation step and do not appear in the draft.
- **Research memory is not yet used for next-action planning.** Memory is indexed and retrievable, but the agent does not yet consult memory when deciding what to suggest next.
- **`parse_and_resolve_research_intent()` is a thin convenience helper.** The recommended path for Phase 4B workloads is `run_resolve_from_request()` / `run_plan_from_request()` / `run_draft_from_request()` in `src/agent/workflow/natural_language_workflow.py`.

---

## 9. Phase Status

### Phase 5 (Phases 5A–5F.2): Complete

Phase 5 delivered the full research-memory-aware next-action pipeline:

- **5A** — Research memory connector (`retrieve_research_memory_context`)
- **5B** — Memory-aware next-action planner (`recommend_next_actions`)
- **5C** — MCP tool exposure for next-action recommendations (49 tools total)
- **5D** — Candidate-to-spec proposals (`propose_spec_from_next_action`)
- **5E** — Spec proposal MCP tools + NL workflow façade
- **5F** — Explicit proposal-to-draft preview (`confirm_proposal=True` gate)
- **5F.2** — Local LLM / Qwen integration checkpoint (LM Studio demo script)

The full stub chain is now verified end-to-end:

```
NL request → parse → resolve → plan → memory → recommend
→ propose_spec → confirm_proposal=True → draft_created ✓
```

**Local LLM integration:** An optional LM Studio / Qwen checkpoint script verifies that a local
Qwen model can act as the intent parser while the deterministic pipeline remains authoritative.
See [`docs/lmstudio_qwen_agent_integration.md`](lmstudio_qwen_agent_integration.md) for setup
and usage. Run the demo:

```bash
# Dry-run (no LM Studio needed):
python scripts/demo_lmstudio_research_agent_loop.py --dry-run

# Live Qwen (LM Studio must be running):
python scripts/demo_lmstudio_research_agent_loop.py --base-url http://127.0.0.1:1234/v1
```

## 10. Future Direction

### Phase 6: Controlled Research Session Loop

Phase 6 will connect the full loop into a guided multi-turn research session:

- Memory → recommendation → spec proposal → user confirmation → draft preview
- Multi-step hypothesis sequences (not just one-shot proposals)
- Natural-language confirmation tool (intentionally deferred until Phase 6 spec)

Phase 6 remains governed: no autonomous execution, no autonomous approval.

---

*For historical context on the design decisions that led to this architecture, see
[`docs/agent_upgrade_audit.md`](agent_upgrade_audit.md).*

*For the end-to-end integration gap audit (Phase 5F), see
[`docs/full_agent_loop_gap_audit.md`](full_agent_loop_gap_audit.md).*

*For the LM Studio / Qwen integration checkpoint, see
[`docs/lmstudio_qwen_agent_integration.md`](lmstudio_qwen_agent_integration.md).*

*For the agent draft governance compatibility audit (Phase 5F.3), see
[`docs/agent_governance_compatibility_audit.md`](agent_governance_compatibility_audit.md).*

*For the governance and rigor controls in the orchestration layer, see
[`docs/AI_ORCHESTRATION_GOVERNANCE_AND_RIGOR_AUDIT.md`](AI_ORCHESTRATION_GOVERNANCE_AND_RIGOR_AUDIT.md).*
