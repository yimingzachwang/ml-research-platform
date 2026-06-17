# Zeto Operator Manual (LM Studio / local Qwen)

Compact operating rules for driving the Zeto AI-assisted research orchestration
tools from an LM Studio chat. Load this at the start of each session with the
read-only MCP tool `get_zeto_operator_manual` instead of pasting long rules.

First message in a new chat:

```text
Load the Zeto operator manual and reply READY. Do not call any other tools yet.
```

## Routing First Rule

**For any natural-language request that is not clearly an existing-experiment
operation, call `route_research_request` first and follow its `recommended_tool`.**

```
route_research_request(request="<user's message>")
```

Then call the tool in `recommended_tool`. Do not call tools in `discouraged_tools`.

### For new preset / model / timeframe requests, do NOT call:

```
create_research_session         ← wrong for new designs
build_context_summary           ← wrong for new designs
run_experiment_review           ← wrong for new designs
generate_iteration_proposal     ← wrong for new designs
generate_experiment_draft       ← wrong for new designs
execute_approved_config         ← wrong for new designs
```

**Use instead:**

```
create_validated_draft_from_request(request="<user's message>")
```

### Example correct flow for "Use the canonical multi-asset ETF preset and test Lasso"

```
# Step 1 — classify the request
route_research_request(request="Use the canonical multi-asset ETF preset from 2015 to 2024 and test Lasso instead of Ridge.")
# → request_type: new_research_design
# → recommended_tool: create_validated_draft_from_request

# Step 2 — follow the recommended tool
create_validated_draft_from_request(request="Use the canonical multi-asset ETF preset from 2015 to 2024 and test Lasso instead of Ridge.")
# → draft preview produced; draft_approved=false; execution_performed=false
```

Do not call `create_research_session` or `build_context_summary` for this request.

## Rules

1. Use exact snake_case MCP tool names. Do not camelCase tool names.
2. Show each tool's `display` field after every tool call.
3. Use compact tool outputs only. Reason from paths, hashes, and IDs.
4. Before major transitions, call `check_research_workflow_state` and follow its
   `next_suggested_action`.
5. Store the `session_id` UUID returned by `create_research_session`.
6. Pass `session_id` to all session-aware tools (review, proposal, draft,
   validate, approve, render, execute, post-run review, get_session_summary).
   If a session exists, always pass its `session_id` to approve/render/execute/
   post-run review — never pass `session_id=null` while an active session exists.
7. If `session_id` is lost, call `get_latest_research_session` before continuing.
8. Never use `experiment_name`, `context_hash`, `draft_id`, or `config_path` as
   `session_id`.
9. Never invent metrics. Use tool outputs only.
10. Never invent or hand-write draft configs. Drafts come from
    `generate_experiment_draft` or `generate_parameter_change_draft` only.
10a. For a specific user-requested config change (e.g. "set alpha to 2"), use
    `generate_parameter_change_draft`, not `generate_experiment_draft`.
10b. For Sharpe, OOS Sharpe, drawdown, hit rate, or "did performance improve?"
    questions, use `get_experiment_metrics` or `compare_experiment_metrics` —
    never `retrieve_research_memory` or `semantic_retrieve_research_memory`.
10c. Never invent report contents or sample metrics. Performance is authoritative
    only from experiment artefacts; RAG memory is research context, not proof of
    performance.
11. If a tool returns `ok=false`, report the failure verbatim and stop.
12. If validation fails, report the failure and stop. Do not repeatedly call
    `generate_experiment_draft`.
13. If validation fails because the proposed experiment name already exists, ask
    the user whether to use the next available suffix or clean existing local
    demo artefacts. Do not regenerate repeatedly.
14. Do not auto-approve drafts. Ask the user before approval. Approval requires
    `approval_confirmation="APPROVE"`. Validation success never authorises
    approval. Only a fresh user message that explicitly approves the draft (prefer
    "I approve the draft.") authorises `approve_experiment_draft` — never infer
    approval from "start a research cycle", validation success, or a
    `next_suggested_action`. After `approval_refused`, stop and wait for the user;
    do not retry automatically.
15. Do not render YAML unless the user approves the draft.
16. Do not execute unless the user explicitly provides `RUN`
    (`confirmation="RUN"`). Do not treat `execute`, `yes`, `proceed`, or
    `continue` as RUN — only a fresh user message containing the literal token
    `RUN` authorises execution. Never infer RUN from a previous approval, a
    previous render, or a prior failed/refused attempt. After `execution_refused`,
    stop and wait for a new user message; do not retry automatically.
17. Do not run extra experiments, optimise automatically, or loop.
18. Before review/proposal/draft, optionally call `retrieve_research_memory` for
    prior related evidence if relevant.
19. Research memory is evidence-only. Retrieved memory does not authorise
    execution or approval.
20. Do not use memory as proof of performance; quant metrics remain
    authoritative.
21. For local Qwen-backed review/proposal/draft calls, use provider `openai`,
    model `qwen2.5-7b-instruct`, base_url `http://127.0.0.1:1234/v1`
    (no real OpenAI API key needed).
22. Follow the governed sequence: create session → check state → execute
    baseline if needed → build context → review → proposal → draft → validate →
    approval → render YAML → RUN execution → post-run review → session summary.
23. The quant engine remains authoritative. Qwen coordinates and interprets only.

## Governance stopping rules

These are hard boundaries. A high-level request never carries through them.

- A high-level user request such as "start a research cycle" does **not**
  authorise approval, rendering, or execution. It only authorises the
  read/advisory steps (context, review, proposal, draft, validate).
- **Stop after validation** and ask the user for approval. Validation passing
  never authorises approval (`next_suggested_action: ask_user_for_approval`).
- **Approval is tool-enforced.** `approve_experiment_draft` requires
  `approval_confirmation="APPROVE"`; only a fresh user message that explicitly
  approves the draft authorises it. On `approval_refused`, stop and wait for the
  user — do **not** retry `approve_experiment_draft` automatically. The refusal's
  `next_suggested_action` is `ask_user_for_approval`, not `render_draft_to_yaml`.
- **Stop after YAML rendering** and ask the user for RUN. Rendering never
  authorises execution (`next_suggested_action: ask_user_for_execution_authorisation`).
- **On `execution_refused`** (confirmation was not the literal `RUN`), stop and
  wait for a fresh user message — do **not** retry `execute_approved_config`
  automatically. The refusal's `next_suggested_action` is
  `ask_user_for_execution_authorisation`, not `execute_approved_config`.
- Approve and render in one step only if the user explicitly asked for both in
  the same message; otherwise stop and ask before rendering.
- **Stop after the final session summary** — the cycle is complete
  (`next_suggested_action: stop_cycle_complete`).
- **Do not start a second iteration** (a new proposal, draft, or experiment)
  unless the user explicitly asks for one.

## Research Memory / Phase 1 RAG

A lightweight, controlled evidence layer. It indexes **compact summaries** of
existing Zeto artefacts (metadata, metrics, reviews, proposals, drafts, reports,
sessions) into a local JSONL index and retrieves prior research evidence by
deterministic keyword / metadata matching. Phase 1 only: no embeddings, no
vector database, no semantic retrieval, no agent framework.

Tools:

- `get_research_memory_status` — read-only; whether the index exists and how many
  items / experiments it holds.
- `index_research_memory` — controlled write; builds/refreshes the index from
  known artefact locations only. Runs no experiment, calls no LLM, approves/
  renders/executes nothing, mutates no source artefacts.
- `retrieve_research_memory` — read-only; retrieve by `query`, `experiment_name`,
  `failure_modes`, `artefact_type`, `top_k`. Returns compact items (summaries,
  paths, hashes, tags, matched terms) — never full artefacts.

Rules of use:

- Retrieval is **evidence-only**. It does not authorise execution or approval.
- Do not treat retrieved memory as proof of performance — quant metrics remain
  authoritative.
- Memory returns pointers; open the referenced artefact paths for full detail.

Example LM Studio prompts:

```text
Index the current Zeto research memory.
Have we seen this failure mode before?
Retrieve prior memory related to poor_oos_consistency and catastrophic_split.
Use retrieved memory to inform the next proposal, but do not execute anything.
```

## Research Memory / Phase 2 Semantic Retrieval

Phase 2 adds **local semantic retrieval** over the same Phase 1 records. Each
compact record is embedded with a local embedding model and ranked by cosine
similarity. Still local and minimal: no vector database, no LangChain/LangGraph,
no chat/completion LLM (embeddings endpoint only), no raw-document streaming.

Tools:

- `get_semantic_research_memory_status` — read-only; whether the semantic index
  exists, item count, and embedding model.
- `index_semantic_research_memory` — controlled write; embeds Phase 1 records via
  the local embeddings endpoint only (default provider `openai`, model
  `text-embedding-nomic-embed-text-v1.5`, base_url `http://127.0.0.1:1234/v1`).
  Runs no experiment, calls no chat LLM, approves/renders/executes nothing.
  Requires a Phase 1 index first.
- `semantic_retrieve_research_memory` — read-only; embeds the query, ranks by
  cosine similarity with optional `experiment_name` / `failure_modes` /
  `artefact_type` / `tags` filters, returns up to `top_k` compact items with
  similarity scores — never full artefacts.

Rules of use:

- Semantic research memory is **evidence-only**. Retrieved semantic matches are
  suggestions, not proof. Quant metrics remain authoritative.
- Memory retrieval does not authorise approval, rendering, or execution.
- If semantic retrieval fails (e.g. the embedding endpoint is down), report and
  stop. Do not auto-retry and do not invent prior evidence.
- If there is no Phase 1 index, run `index_research_memory` first; if there is no
  semantic index, run `index_semantic_research_memory` first.

Example LM Studio prompts:

```text
Index semantic research memory using local embeddings.
Retrieve semantically similar prior research about momentum instability and poor OOS consistency.
Use semantic memory as supporting evidence only. Do not execute anything.
```

## Comparison evidence memory

`compare_experiment_metrics` now persists a compact `comparison_evidence` record at
`results/comparisons/<base>__vs__<candidate>/comparison_evidence.json` every time it runs.
This record is indexed by Phase 1 and Phase 2 memory so before/after conclusions are
retrievable without re-reading full artefacts.

Enriching the evidence record (optional, improves future retrieval):
```text
compare_experiment_metrics(
  base_experiment_name="canonical_ml_showcase",
  candidate_experiment_name="canonical_ml_showcase_v12",
  research_question="Does risk-adjusted momentum improve OOS stability?",
  tested_change="added risk_adjusted_momentum_20"
)
```

Inspecting a specific comparison directly:
```text
inspect_comparison_evidence(
  base_experiment_name="canonical_ml_showcase",
  candidate_experiment_name="canonical_ml_showcase_v12"
)
```
Returns tested change, metric deltas (Sharpe, OOS Sharpe, MaxDD), failure modes,
and conclusion in one compact envelope. No LLM, no RAG, no execution.

Retrieving past conclusions by search:
- "Have we tested X before?" → `retrieve_research_memory(query="risk_adjusted_momentum", artefact_type="comparison_evidence")`
- "What have we learned overall?" → `retrieve_research_memory(artefact_type="comparison_evidence")`
- Semantic equivalent → `semantic_retrieve_research_memory(query="did momentum features help OOS consistency?")`

Routing rule:
- "What did we learn from X vs Y?" or "Inspect evidence for X vs Y" → `inspect_comparison_evidence`
- "Have we tested X before?" → `retrieve_research_memory(artefact_type=comparison_evidence)`
- No evidence exists yet → run `compare_experiment_metrics` first

Rules:
- Comparison evidence summarises **real metric deltas** from artefacts. Never treat it as proof of performance — use `compare_experiment_metrics` for authoritative numbers.
- Metrics remain authoritative; comparison evidence is a research pointer, not a performance claim.

## Artefact verification and incomplete-metrics stopping rules

After every `execute_approved_config`, call `verify_experiment_artifacts` before
proceeding to `review_post_run_result`.

```
verify_experiment_artifacts(experiment_name="canonical_ml_showcase_v13")
```

Returns `artefact_status: complete` or `incomplete`.

**Hard stopping rules (never override):**

- If `verify_experiment_artifacts` returns `ok=false` (`artefact_status=incomplete`):
  **stop**. Do not compare, review, index, or invent metrics. Report the failure
  verbatim to the user.
- If `get_experiment_metrics` returns `ok=false`: **stop**. Do not compare, review,
  index, or invent metrics for that experiment.
- If `compare_experiment_metrics` returns `ok=false` (`incomplete_metrics` or
  `not_found`): **stop**. Do not inspect evidence or invent conclusions.
- If any review tool returns a stage containing `blocked_incomplete_metrics`:
  **stop**. Never invent hypothetical Sharpe, OOS Sharpe, drawdown, or hit-rate
  values. RAG/memory retrieval cannot substitute for missing authoritative metrics.

The correct Zeto message when artefacts are incomplete:

> "Execution completed, but authoritative metrics are incomplete. The experiment
> cannot be compared, reviewed, or indexed as evidence. Stop and inspect the
> artefacts."

## Experiment intelligence (read-only; no LLM)

Four tools for reasoning across accumulated experiments. All use authoritative
artefact metrics and comparison evidence only — never RAG as proof.

- `rank_experiments` — rank experiment variants by `sharpe_ratio`,
  `mean_oos_sharpe`, `max_drawdown_pct`, or `robustness_score` (composite).
  Use to answer "which experiment is best?".
- `summarise_experiment_family` — aggregate metrics, common failure modes,
  comparison evidence highlights, and deterministic lessons for all experiments
  sharing a name prefix. Use to answer "what have we learned?".
- `check_research_idea_history` — scan comparison evidence by keyword
  relevance; returns matched changes with metric deltas and a recommendation.
  Advisory only — warns but does not block. Use before drafting to avoid
  repeating failed ideas.
- `propose_research_plan` — deterministic, governed plan (up to `max_steps`
  steps) derived from failure modes and comparison evidence. Does **not**
  generate drafts, approve, render, or execute. Each step is advisory only.

Routing:
- "What have we learned?" / "Summarise family" → `summarise_experiment_family`
- "Which experiment is best?" / "Rank experiments" → `rank_experiments`
- "Have we tested this before?" / "Idea history" → `check_research_idea_history`
- "What should we try next?" / "Propose plan" → `propose_research_plan`
- To act on a plan step → `generate_config_change_draft` or `generate_parameter_change_draft`

Rules:
- These tools use authoritative artefact metrics only; never RAG as proof.
- `propose_research_plan` output does not authorise execution or approval.
- Rankings and conclusions come from real metrics; never invented.

## Config introspection (read-only)

Four read-only tools let the agent discover what is changeable before building
any draft.  None of them calls an LLM, approves, renders, or executes anything.

- `inspect_experiment_config` — compact summary of a YAML config: model type,
  feature names, validation settings, and the list of changeable paths.  Never
  returns the raw YAML.
- `list_changeable_config_fields` — the controlled change surface: every field
  path accepted by `generate_config_change_draft`, with its operations (`set` /
  `add` / `remove` / `replace`) and current value when an experiment is given.
  Use this before building a draft to know what can be changed.
- `list_available_features` — schema-authoritative feature types (family, required
  params, currently-used flag when experiment is given).  Never invents type names.
  Supports optional `family` and `query` filters.
- `list_supported_models` — the 5 schema-valid model types.  Never invents model
  names.  Highlights the current model when experiment is given.

## Config-change draft (multi-change, no LLM)

- `generate_config_change_draft` — deterministic multi-change draft.  Accepts a
  JSON array of changes via the `changes` parameter.  Supported shapes:

  ```json
  {"field_path": "model.params.alpha", "operation": "set", "value": 2.0}
  {"field_path": "features.entries", "operation": "add",
   "value": {"name": "sma_50", "type": "sma", "params": {"window": 50}}}
  {"field_path": "features.entries", "operation": "remove", "value": "mom_60"}
  {"field_path": "features.entries", "operation": "replace",
   "old_value": "mom_20", "value": {"name": "sma_20", "type": "sma", "params": {"window": 20}}}
  ```

  All changes are validated before any draft is created — partial success is not
  possible.  Invalid feature types, absent features, duplicate names, and
  schema-incompatible values are refused with no fallback invented.  The draft
  is unapproved.  Use `generate_parameter_change_draft` for a single scalar
  `set` (convenience wrapper).

Routing:
- "What can we change?" → `list_changeable_config_fields`
- "What features are available?" → `list_available_features`
- "Which models are supported?" → `list_supported_models`
- "Show me the current config" → `inspect_experiment_config`
- Multiple changes or add/remove/replace → `generate_config_change_draft`
- Single scalar set → `generate_parameter_change_draft`

## Explicit config changes and authoritative metrics

Two routing rules keep the agent factual:

- **Explicit config change** → `generate_parameter_change_draft` (single field)
  or `generate_config_change_draft` (multiple changes or feature operations).
  Both are deterministic — never use the LLM-driven `generate_experiment_draft`
  for explicit changes, and never claim a value the diff does not show.  Invalid
  field paths / incompatible values / unsupported feature types are refused (no
  fallback invented).  Neither approves, renders, nor executes.
- **Performance question** → `get_experiment_metrics` or
  `compare_experiment_metrics`. For Sharpe, OOS Sharpe, drawdown, hit rate, or
  "did it improve?", read real artefact metrics. Never answer these from
  `retrieve_research_memory` or `semantic_retrieve_research_memory`, and never
  invent report text or sample metrics. If a metric is missing, say so.

Example LM Studio prompts:

```text
What features and model does canonical_ml_showcase_v9 use?
What can we change in canonical_ml_showcase_v9?
List available feature types in the momentum family.
Set model.params.alpha to 2 for canonical_ml_showcase_v9.
Add a sma_50 feature (sma, window=50) to canonical_ml_showcase_v9.
What is the Sharpe and OOS Sharpe for canonical_ml_showcase_v9_v2?
Did canonical_ml_showcase_v9_v2 improve over canonical_ml_showcase_v9?
```

## Tool Routing Rules

These rules govern which tool to use depending on the type of request. Following
them avoids wrong-tool routing loops.

### A. New research-design requests

For a **brand-new** natural-language request involving preset, model, timeframe,
or feature changes — where no config-backed experiment exists yet — use the
high-level agent workflow tools:

```
parse_research_intent                     ← parse NL intent into ResearchSpec
resolve_research_request                  ← parse + resolve
create_experiment_plan_from_request       ← parse + resolve + plan
create_validated_draft_from_request       ← parse + resolve + plan + draft (preferred)
```

**Preferred direct route** for a validated draft preview from natural language:

```
create_validated_draft_from_request
```

Examples of new research-design requests:

```
Use the canonical multi-asset ETF preset from 2015 to 2024 and test Lasso instead of Ridge.
Test a different model on the canonical ETF preset.
Use this timeframe with the canonical preset.
Create a new research design using a preset.
Change model/features/validation/timeframe from a registry preset.
```

### B. Memory-aware "what should we try next?" requests

For requests asking what to try next or requesting a recommendation:

```
What should we try next?
What is the next research action?
Given prior memory, what should we test?
```

Use:

```
recommend_next_actions_from_request       ← NL → recommended candidates
propose_spec_from_recommendation_request  ← NL → candidate → spec proposal
```

To convert an accepted proposal into a draft preview:

```
create_validated_draft_from_spec_proposal (with confirm_proposal=true)
```

### C. Existing experiment review and iteration workflows

Use the older orchestration tools **only** when the user is working with an
existing config-backed experiment that already has a YAML file in
`configs/experiments/`.

Examples:

```
Review canonical_ml_showcase.
Generate an iteration proposal for canonical_ml_showcase.
Generate a draft from the existing review/proposal.
Validate this existing draft.
Approve this existing draft.
Render this approved draft.
Execute this approved config.
```

Relevant tools:

```
build_context_summary          ← build LLM context for an existing experiment
run_experiment_review          ← review existing experiment artefacts
generate_iteration_proposal    ← propose next step for existing experiment
generate_experiment_draft      ← draft from existing review/proposal (NOT for new designs)
validate_experiment_draft
approve_experiment_draft
render_draft_to_yaml
execute_approved_config
```

### D. Do NOT use generate_experiment_draft for new research-design requests

**`generate_experiment_draft` requires an existing config-backed experiment with
a prior persisted iteration proposal.** It cannot create a new research design
from a preset or model change.

If the user's request describes a brand-new research design (e.g. "use the
canonical ETF preset and test Lasso"), do **not** call `generate_experiment_draft`.
Use `create_validated_draft_from_request` instead.

If `generate_experiment_draft` returns a routing redirect (`wrong_tool_for_research_design_request`),
do not retry it — switch to `create_validated_draft_from_request`.

### E. Do NOT invent experiment names

When the user references a preset (e.g. `canonical_multi_asset_etf`) or
a model/timeframe change, do **not** invent an experiment name like
`TestLasso2015_2024`. Use the registry/preset workflow through the high-level
agent tools. Experiment names come from the governed pipeline, not from LLM
inference.

### F. Retry rule

If a low-level orchestration tool returns:
- `unknown experiment`
- `missing config`
- `no YAML file found`
- `no IterationProposal found`
- `wrong_tool_for_research_design_request`

**Do not retry the same tool with the same or a modified experiment name.**
Re-route to the appropriate high-level agent workflow tool (see A or B above)
or ask the user for clarification. Repeating a failed low-level call does not
resolve the underlying wrong-routing issue.

### G. Governance gates remain unchanged

These gates are enforced in code regardless of tool routing:

```
approve_experiment_draft  requires approval_confirmation="APPROVE"
render_draft_to_yaml      requires an approved draft
execute_approved_config   requires confirmation="RUN"
```

Natural-language wording alone (e.g. "yes", "proceed", "start research cycle")
must never trigger approval, render, or execution. Only a fresh user message
containing the exact required confirmation token authorises these actions.

---

## How to load

`get_zeto_operator_manual` is read-only: it returns this fixed manual (a short
rules list and this file's path) and nothing else. It executes nothing, calls no
LLM, inspects no experiment artefacts, creates no session, mutates no state, and
reads no arbitrary files.
