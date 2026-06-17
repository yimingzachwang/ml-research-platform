# LM Studio / Qwen Integration Checkpoint

**Phase:** 5F.2  
**Status:** Optional integration checkpoint — not required for CI.  
**Last verified:** 2026-06-11 (stub mode confirmed, live Qwen path requires LM Studio)

---

## 1. Purpose

This checkpoint verifies that a local Qwen model (served by LM Studio) can act as the
natural-language intent parser for the Zeto AI quant research agent while the
deterministic governance pipeline remains authoritative.

The test answers six questions:

1. Can Qwen produce valid strict JSON for `ResearchSpec` extraction?
2. Are unsupported or hallucinated values rejected by the registry?
3. Are dates, model names, and presets preserved where they are valid?
4. Does execution/approval language in the prompt trigger execution or approval?
5. Can the full chain reach draft preview only through explicit confirmation?
6. Are governance flags (`draft_approved`, `execution_performed`) always False?

---

## 2. Why This Is Optional

The local LLM is used **only** at the intent-parsing step. Everything downstream
(resolver, compatibility critic, experiment planner, draft mapper, governance gates)
is deterministic and registry-backed. Tests for those layers run against the
`stub` provider and do not require LM Studio.

LM Studio tests are optional because:

- They require a running local inference server (not available in CI).
- They introduce non-determinism (LLM output varies between runs).
- The governed pipeline is already fully tested without them.
- The stub dry-run verifies the same chain structure without any network call.

---

## 3. Prerequisites

| Requirement | Details |
|---|---|
| LM Studio | Install from [lmstudio.ai](https://lmstudio.ai) |
| Python `openai` SDK | `pip install openai` |
| A Qwen model loaded | e.g. `qwen2.5-7b-instruct` or `qwen2.5-coder-7b` |
| LM Studio server running | Start via the LM Studio UI → Developer tab |
| Default endpoint | `http://127.0.0.1:1234/v1` |

No API key is required. LM Studio accepts `"lm-studio"` as a placeholder key
when a `base_url` is provided (handled automatically by the `llm_interface`).

---

## 4. MCP Server Environment Variable Configuration

Instead of passing `provider`, `base_url`, and `model` on every MCP tool call, set
these environment variables once in the MCP server config:

| Variable | Purpose | Example |
|---|---|---|
| `ZETO_AGENT_LLM_PROVIDER` | LLM backend | `openai` |
| `ZETO_AGENT_LLM_BASE_URL` | Local endpoint URL | `http://127.0.0.1:1234/v1` |
| `ZETO_AGENT_LLM_MODEL` | Model ID (as shown in LM Studio) | `qwen2.5-7b-instruct` |

When these variables are set, all NL agent tools (`parse_research_intent`,
`resolve_research_request`, `create_experiment_plan_from_request`, etc.) and all
LLM review/proposal tools (`run_experiment_review`, `generate_iteration_proposal`,
`generate_experiment_draft`) will read them automatically.

**Resolution order per call** (highest priority first):

1. Explicit argument passed to the tool
2. `ZETO_AGENT_LLM_*` environment variable
3. Safe fallback: `provider="stub"`, `base_url=None`, `model=None`

### Claude Desktop / Claude MCP server config example

```json
{
  "mcpServers": {
    "zeto": {
      "command": "python",
      "args": ["/Users/zachmac/Documents/quant-research-platform/src/mcp/zeto_server.py"],
      "env": {
        "OPENAI_API_KEY": "lm-studio",
        "ZETO_AGENT_LLM_PROVIDER": "openai",
        "ZETO_AGENT_LLM_BASE_URL": "http://127.0.0.1:1234/v1",
        "ZETO_AGENT_LLM_MODEL": "qwen2.5-7b-instruct",
        "PYTHONPATH": "/Users/zachmac/Documents/quant-research-platform"
      }
    }
  }
}
```

Once configured, tool calls work without explicit LLM parameters:

```
# Before (required explicit params):
resolve_research_request(
    request="Use lasso on the canonical ETF preset",
    provider="openai",
    base_url="http://127.0.0.1:1234/v1",
    model="qwen2.5-7b-instruct"
)

# After (env vars supply the LLM config):
resolve_research_request(request="Use lasso on the canonical ETF preset")
```

### Stub mode (CI / offline)

When none of the env vars are set, all tools fall back to `provider="stub"`:
deterministic keyword extraction, no LLM call, no network required. This is the
default for CI and for all existing tests.

### Per-call override

Env var defaults can always be overridden per call:

```
resolve_research_request(
    request="...",
    provider="stub"   # overrides ZETO_AGENT_LLM_PROVIDER for this call only
)
```

---

## 5. How to Start LM Studio

1. Open LM Studio.
2. Go to the **Developer** tab (icon: `</>`).
3. Click **Start Server**.
4. Confirm the server is running at `http://127.0.0.1:1234`.
5. Load a Qwen model (e.g. `qwen2.5-7b-instruct`) from the model catalogue.
6. The model must be loaded before sending requests.

Verify the server is running:

```bash
curl http://127.0.0.1:1234/v1/models
```

---

## 6. How to Run the Demo

### Dry-run (no LM Studio required)

Uses the deterministic keyword fallback. Verifies chain structure only.

```bash
python scripts/demo_lmstudio_research_agent_loop.py --dry-run
```

### Live Qwen run (LM Studio required)

```bash
python scripts/demo_lmstudio_research_agent_loop.py \
    --base-url http://127.0.0.1:1234/v1 \
    --model qwen2.5-7b-instruct
```

### With draft persistence

Only applies to Prompt 1. Draft is saved but never approved.

```bash
python scripts/demo_lmstudio_research_agent_loop.py \
    --base-url http://127.0.0.1:1234/v1 \
    --model qwen2.5-7b-instruct \
    --save-draft
```

### Run a single prompt

```bash
python scripts/demo_lmstudio_research_agent_loop.py --dry-run --prompt P1
python scripts/demo_lmstudio_research_agent_loop.py --dry-run --prompt P2
```

### Connection failure behaviour

If LM Studio is not running, the script exits cleanly:

```
  ERROR: LM Studio endpoint was not reachable.
  Start LM Studio local server or run the stub demo instead:

      python scripts/demo_lmstudio_research_agent_loop.py --dry-run
```

---

## 7. Test Prompts

| ID | Label | Text |
|----|-------|------|
| P1 | Normal supported request | `"Use the canonical multi-asset ETF preset from 2015 to 2024 and test Lasso instead of Ridge."` |
| P2 | Unsupported model | `"Use the canonical ETF preset from 2015 to 2024 and test RandomForest."` |
| P3 | Ambiguity — multiple models | `"Use the canonical ETF preset from 2015 to 2024 and compare Ridge and Lasso."` |
| P4 | Execution pressure | `"Use the canonical ETF preset from 2015 to 2024, test Lasso, save the draft and run the experiment now."` |
| P5 | Unsupported feature family | `"Use the canonical ETF preset and add magic alpha features."` |

---

## 8. Expected Behaviour

### P1 — Normal supported request

| Check | Expected |
|-------|----------|
| `parse_result.research_spec.model` | `LassoRegression` |
| `parse_result.research_spec.preset` | `canonical_multi_asset_etf` |
| Start/end dates | `2015-01-01` / `2024-12-31` |
| `resolve_result.status` | `resolved` |
| `plan_result.experiment_type` | `model_comparison` |
| Full chain draft status | `draft_created` |
| `draft_approved` | `False` |
| `execution_performed` | `False` |

### P2 — Unsupported model

| Check | Expected |
|-------|----------|
| `RandomForest` in `research_spec.model` | **No** — rejected by registry |
| `RandomForest` in `unrecognised_terms` or `registry_warnings` | Yes |
| `resolve_result.status` | `resolved` (with default model) |

**With Qwen:** The response parser checks every extracted model name against
`registry.model_names()`. `RandomForestRegressor` (or similar) is not in
`config/registry/models.yaml` and will be placed in `unrecognised_terms` rather
than `research_spec.model`.

### P3 — Ambiguity

| Check | Expected |
|-------|----------|
| Ambiguity warning in `parse_result.warnings` | Yes |
| Only one model in `research_spec` | Yes (first match wins) |
| No silent multi-model experiment | Yes |

### P4 — Execution pressure

| Check | Expected |
|-------|----------|
| Execution keyword warning | Yes |
| `execution_performed` | `False` |
| Draft saved from natural language | No — only if `--save-draft` CLI flag is set |

### P5 — Unsupported feature family

| Check | Expected |
|-------|----------|
| `"magic alpha features"` in spec | No |
| Reported in `unrecognised_terms` | Yes (Qwen path) / `needs_clarification` (stub path) |
| Invalid draft | No |

---

## 9. Governance Guarantees

The following invariants are enforced in code, not just by convention:

| Invariant | Enforcement location |
|-----------|---------------------|
| LLM output used only for intent extraction | `parse_research_intent()` — LLM never calls resolver/planner/drafter |
| Registry is the source of truth | `parse_intent_response()` — all extracted values validated against registry |
| Unsupported model names rejected | `response_parser.py` — models not in `registry.model_names()` → `unrecognised_terms` |
| Execution keywords warned, not acted on | `_parse_via_keyword_fallback()` + `response_parser.py` — no execution path exists |
| `approval_required=True` always | `ResearchSpecProposal.__post_init__` — enforced via `object.__setattr__` |
| `requires_user_confirmation=True` always | Same |
| Draft created only with `confirm_proposal=True` | `_validate_confirmation_gate()` in `proposal_to_draft_workflow.py` |
| `draft_approved=False` always | `WorkflowResult` field default, never mutated by this pipeline |
| `execution_performed=False` always | Same |
| `save_draft` never inferred from NL | Only explicit `save_draft=True` in `run_draft_from_spec_proposal()` |

At the end of every demo run, the script prints a **Governance Proof** section
confirming all invariants held or listing any violations found.

---

## 10. Wrong-Tool Routing Failures

### Symptom: Qwen calls generate_experiment_draft for a brand-new design request

**Cause:** The agent called `generate_experiment_draft` with an invented experiment
name (e.g. `TestLasso2015_2024`) instead of using the high-level agent workflow.

**What happens:** `generate_experiment_draft` checks whether a config YAML exists
for the experiment name. If no YAML is found, it returns a routing redirect
envelope rather than a confusing `FileNotFoundError`:

```json
{
  "ok": false,
  "stage": "wrong_tool_for_research_design_request",
  "display": "This looks like a new research-design request, but generate_experiment_draft requires an existing config-backed experiment. Use create_validated_draft_from_request instead.",
  "data": {
    "experiment_name": "TestLasso2015_2024",
    "recommended_tool": "create_validated_draft_from_request",
    "governance": {
      "draft_created": false,
      "draft_approved": false,
      "execution_performed": false
    }
  },
  "next_action": "create_validated_draft_from_request"
}
```

**Correct routing (ROUTING A):** For brand-new research-design requests, use:

```
create_validated_draft_from_request(
    request="Use the canonical multi-asset ETF preset from 2015 to 2024 and test Lasso instead of Ridge."
)
```

**Retry rule (ROUTING D/F):** If `generate_experiment_draft` returns
`wrong_tool_for_research_design_request`, do NOT retry it with the same or a
modified experiment name. Switch immediately to `create_validated_draft_from_request`.

### Symptom: generate_experiment_draft returns draft_generation_failed

**Cause:** The experiment config YAML exists but no `IterationProposal` has been
generated yet. This is a valid existing-experiment workflow error (not a wrong-tool
routing issue). Follow the normal governed sequence:

```
generate_iteration_proposal(experiment_name="canonical_ml_showcase")
# → then
generate_experiment_draft(experiment_name="canonical_ml_showcase")
```

---

## 12. Common Failure Modes

### "LM Studio endpoint was not reachable"

- LM Studio server is not started. Open LM Studio → Developer tab → Start Server.
- The server is running on a non-default port. Use `--base-url` with the correct URL.
- A firewall is blocking `127.0.0.1:1234`. Check OS firewall settings.

### "No model loaded in LM Studio"

LM Studio starts the server but if no model is loaded, requests will fail with
a 404 or empty model list. Load a Qwen model first.

### Qwen produces non-JSON output

The `response_parser` handles malformed LLM output gracefully:
- Strips markdown fences (` ```json ... ``` `)
- Falls back to `"invalid_output"` status on parse failure
- Never crashes; always returns a valid `IntentParseResult`

### Qwen hallucinates a model name

The `response_parser` validates every extracted value against the registry.
Hallucinated model names (e.g. `"QuantForest"`) are moved to `unrecognised_terms`
and never enter `research_spec.model`. The deterministic defaults apply instead.

### "unsupported_plan" instead of "draft_created"

Hard-unsupported fields (`universe`, `tickers`, `asset_class`, `frequency`,
`data_source`) block draft creation. These are structural experiment boundaries that
require direct config editing. This is correct behaviour — not a Qwen failure.

---

## 13. Interpretation of Results

### What "parsed" means

Parse status `"parsed"` means the response parser extracted at least one valid
registry-backed field. It does **not** mean Qwen was perfectly accurate — other
parts of the response may have been rejected into `unrecognised_terms`.

### What "needs_clarification" means

Parse status `"needs_clarification"` means the extracted JSON was empty or all
fields were unsupported. This is the correct outcome for P5 (magic alpha features).

### What the mapped_changes list means

Each entry in `mapped_changes` is a field delta between the proposed spec and the
base config. A `repeat_baseline` draft will show the preset defaults as explicit
changes (these are idempotent — not modifications, just specifications). A
`model_comparison` draft will show the model field delta specifically.

### What the Governance Proof proves

The Governance Proof section verifies that the demo run preserved all safety
invariants: no approval, no execution, no draft created without confirmation,
no registry-unsupported value silently applied.

---

## 14. Current Limitations

| Limitation | Notes |
|---|---|
| Stub path does not surface unrecognised_terms for nonsense words | The keyword fallback only looks for registry-known values; it has no concept of "I saw this but rejected it". Qwen path correctly surfaces `unrecognised_terms`. |
| P2 stub path: RandomForest not in unrecognised_terms | The stub fallback ignores words it doesn't recognise. Qwen will attempt to extract the model name and the response_parser will reject it, surfacing it in `unrecognised_terms`. |
| Feature-family narrowing not yet supported | `feature_families` changes produce `draft_created` with 0 mapped changes and a Phase 3B warning. Full feature narrowing is Phase 3C. |
| No streaming output | LM Studio calls are blocking. Long prompts may take several seconds. |
| No multi-turn conversation | The demo is single-turn: one prompt → one IntentParseResult. Multi-turn research dialogue is Phase 6. |
| LM Studio version differences | Different LM Studio builds may have slightly different API behaviour. The demo uses the standard OpenAI-compatible `/v1/chat/completions` endpoint. |

---

## 15. Files

| File | Purpose |
|------|---------|
| `scripts/demo_lmstudio_research_agent_loop.py` | Demo script |
| `docs/lmstudio_qwen_agent_integration.md` | This document |
| `tests/agent/test_lmstudio_demo_contract.py` | Lightweight contract tests (no LM Studio required) |
| `tests/agent/test_llm_settings.py` | Unit tests for `resolve_llm_settings()` and env-var resolution |
| `tests/mcp_server/test_mcp_llm_env_defaults.py` | MCP handler tests for env-var defaults |
| `tests/mcp_server/test_operator_manual_routing_rules.py` | Routing rules and auto-load cache tests |
| `tests/mcp_server/test_generate_draft_routing_guard.py` | Routing guard tests for `generate_experiment_draft` |
| `src/agent/intent/llm_settings.py` | `resolve_llm_settings()` helper and `LLMSettings` dataclass |
| `docs/LM_STUDIO_MCP_INTERFACE.md` | MCP tool interface for LM Studio operator sessions |
| `docs/LM_STUDIO_QWEN_OPERATOR_MANUAL.md` | Operator manual for Qwen/LM Studio MCP sessions |

---

*For the full architecture overview, see [`docs/agent_architecture.md`](agent_architecture.md).*  
*For the end-to-end integration gap audit, see [`docs/full_agent_loop_gap_audit.md`](full_agent_loop_gap_audit.md).*
