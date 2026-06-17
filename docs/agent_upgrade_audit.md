# AI Quant Research Agent Upgrade Audit

**Date:** 2026-06-09
**Scope:** quant-research-platform — full codebase audit for agent upgrade design
**Status:** Read-only. No implementation changes made.

---

> **Status update (2026-06-10):**
> This audit is now a historical design record. The core architecture proposed here has been
> implemented through Phases 1–4B: registry, ResearchSpec, resolver, compatibility critic,
> planner, draft mapping, MCP exposure, draft persistence, and natural-language workflow.
> The current canonical architecture is documented in [`docs/agent_architecture.md`](agent_architecture.md).

---

## 1. Executive Summary

The platform is a production-grade quantitative research system with a well-engineered 7-layer architecture. The quant engine (data → features → model → strategy → backtest → diagnostics → reporting) is solid, schema-driven, and largely configurable. The AI orchestration layer is well-governed with 36 MCP tools, explicit approval gates, deterministic memory, and clean provenance.

However, the current agent design is narrow: it is optimized for iterating on one canonical single-asset ML showcase (SPY Ridge regression) and does not support user-initiated research intent over an open universe of asset classes, strategies, or configurations. The proposed upgrade requires a **research design resolver** sitting above the existing quant engine — a layer that can accept incomplete user intent, detect missing fields, consult a registry of valid options, check compatibility, resolve a full research specification, and then hand a complete validated config to the existing engine.

**The existing engine should not be touched.** The gap is entirely in the layers above it.

**Key conclusions:**
- The quant engine is reusable as-is for the upgrade.
- The orchestration infrastructure (sessions, memory, context, approval, execution) is reusable with minor extension.
- The missing layer is: registry → intent → resolver → planner → critic → human approval → existing engine.
- The biggest architectural debt is that "asset class", "universe", "preset", and "strategy type" are not first-class queryable objects — they are YAML files and frozensets scattered across the codebase.
- Three hardcoded assumptions drive the most risk: the single-asset (SPY) focus of all canonical ML configs, the ticker-set-based asset classification in reporting, and the 20-path draft validator whitelist that doesn't know about universes or strategies.

---

## 2. Current Repository Structure

```
quant-research-platform/
├── agents/               # Role definitions (7 research agents)
├── configs/
│   ├── ai_agents/        # LLM agent role YAML (research_reviewer.yaml)
│   ├── backtests/        # Backtest methodology (default_daily.yaml)
│   ├── data/             # Data ingestion profiles (daily_prices.yaml, extended_prices.yaml)
│   ├── experiments/      # 30+ experiment YAML configs (v1 + v2 schemas)
│   │   ├── canonical_ml_showcase.yaml ... v13.yaml
│   │   ├── equal_weight.yaml
│   │   ├── momentum_rotation.yaml
│   │   ├── allocation_study/ (4 variants)
│   │   ├── signal_geometry/ (4 alpha sweep variants)
│   │   └── examples/
│   ├── features/         # Feature template (placeholder_features.yaml)
│   ├── mlflow/           # MLflow tracking config
│   ├── models/           # Model template (noop_model.yaml)
│   ├── portfolio/        # Portfolio construction (long_only.yaml)
│   ├── reports/          # Report spec (default_markdown.yaml)
│   ├── risk/             # Risk spec (basic_risk.yaml)
│   └── universes/        # Universe definitions (core_etfs.yaml, extended_etfs.yaml)
├── data/                 # Local data lake (gitignored)
├── docs/                 # 27 documentation files
├── reports/              # Executed experiment outputs (markdown, HTML, figures)
├── results/              # Runtime artefacts (gitignored)
├── scripts/              # 18 research entrypoint scripts
├── skills/               # 7 reusable AI-agent skill templates
├── src/                  # 30 production Python packages
└── tests/                # 139 test files, 2,271+ passing tests
```

**Observation:** The top-level layout is clean and well-organized. The `configs/` tree already suggests a registry-like structure (universes, models, features) but does not yet expose these as queryable first-class registries.

---

## 3. Current Research Engine Capabilities

### Layer architecture (7 layers, bottom to top)

| Layer | Package | Status | Reusable |
|-------|---------|--------|---------|
| Data | `src/data/`, `src/ingestion/`, `src/cleaning/` | Complete | Yes — fully parameterized |
| Features | `src/features/` | Complete | Yes — 16 typed families |
| Model | `src/ml/`, `src/models/` | Complete | Yes — 5 types, protocol-driven |
| Strategy | `src/strategies/` | Complete | Yes — protocol + implementations |
| Backtest | `src/backtesting/`, `src/portfolio/`, `src/execution/` | Complete | Yes |
| Diagnostics/Validation | `src/validation/`, `src/evaluation/`, `src/risk/` | Complete | Yes |
| Reporting | `src/reporting/`, `src/visualization/` | Complete | Yes — 18 sections, 4 presets |

### Config schema

The quant engine uses two schemas:

**v1 (legacy, simple strategies):** `src/experiments/config.py`
```python
ExperimentSpec: experiment_name, strategy_name, universe: list[str],
                start_date, end_date, rebalance_frequency, parameters, tags
```

**v2 (ML-driven, current standard):** `src/experiments/ml_config.py`
```python
MLExperimentSpec: name, universe: list[str], date_range,
                  features: FeatureSpec, labels: LabelSpec,
                  model: ModelSpec, signal: SignalSpec,
                  portfolio_construction: PortfolioConstructionSpec,
                  validation: ValidationSpec, execution: ExecutionSpec
```

Both schemas are Pydantic-validated, deterministically hashed (SHA-256, first 12 chars), and written as YAML.

### Registered capabilities (authoritative frozensets in `ml_config.py`)

| Registry | Members |
|----------|---------|
| Feature types | 16 (momentum, rolling_volatility, trend_strength, rolling_beta, risk_adjusted_momentum, sma, ema, vol_compression, trend_persistence, breakout_strength, drawdown_distance, rolling_zscore, bollinger_distance, rolling_skewness, rolling_autocorrelation, vol_percentile, downside_volatility, vol_of_vol) |
| Label types | 4 (forward_returns, binary_direction, volatility_target, ranking_target) |
| Model types | 5 (LinearRegression, RidgeRegression, LassoRegression, ElasticNetRegression, LogisticRegression) |
| Signal types | 5 (sign, threshold, top_n, long_short, normalize) |
| Weighting schemes | 3 (equal_weight, zscore_softmax, confidence_weighted) |

These frozensets are the authoritative truth for schema validation. They exist inside `_VALID_FEATURE_TYPES`, `_VALID_MODEL_TYPES`, etc. but are not yet surfaced as a user-queryable registry object.

### Data layer

- **Sources:** Yahoo Finance (OHLCV daily), FRED (macro)
- **Configurable:** Universe is any list of tickers; data contracts accept any symbol
- **Not configurable from config files:** data source type (always yfinance for OHLCV), frequency (always 1d)
- **Key reusable class:** `DataAgentV1Config` in `src/data/config.py` — supports pluggable sources via `supported_sources` dict

### Validation (walk-forward)

Walk-forward is clean and leakage-safe (`src/validation/walk_forward.py`). Parameters are:
- `train_months`, `test_months`, `step_months`, `gap_days` — all in v2 YAML config
- Rolling or expanding windows
- Optional `fit()` hook called before each test window on training slice only

This is robust and fully reusable for any asset class or strategy.

### Diagnostics

- Turnover, concentration (HHI), rolling correlation — in `src/visualization/diagnostics.py`
- Prediction quality (IC, Spearman correlation) — in `src/ml/diagnostics/`
- Failure mode detection — in `src/orchestration/context/failure_mode_detector.py`
- All deterministic, computed from persisted artefacts

---

## 4. Current AI Orchestration Capabilities

### What the current agent can do

| Capability | Status | Evidence |
|-----------|--------|---------|
| Parse user intent | Partial | `intent_schema.py` exists; no deep NL parser |
| Generate experiment drafts | Yes | `draft_generator.py` — LLM + deterministic paths |
| Validate drafts | Yes | `draft_validator.py` — whitelist + schema |
| Approve drafts | Yes | `approve_experiment_draft` with explicit "APPROVE" token |
| Execute approved configs | Yes | `execute_approved_config` with explicit "RUN" token |
| Review results (LLM) | Yes | `run_experiment_review`, `review_post_run_result` |
| Compare experiments | Yes | `compare_experiment_metrics` + `comparison_evidence.json` |
| Index and retrieve memory | Yes | Phase 1 (keyword) + Phase 2 (semantic embeddings) |
| Recommend next actions | Partial | `propose_research_plan` (5 hardcoded suggestion types) |
| Detect missing config fields | No | Not implemented |
| Know which choices are valid | Partial | Whitelist exists; not surfaced as discovery mechanism |
| Check compatibility between fields | No | Not implemented |
| Handle incomplete user research intent | No | No resolver or default-filling layer |
| Support multiple asset classes | No | Architecture supports it; no user-facing routing |
| Accept custom universe | No direct path | User must write YAML manually; no guided flow |

### The 36 MCP tools (zeto_server.py)

Tools are cleanly organized into functional groups:

| Group | Count | Purpose |
|-------|-------|---------|
| Operator manual | 1 | Rules delivery |
| Memory (Phase 1) | 4 | Index + retrieve by keyword |
| Memory (Phase 2) | 3 | Index + retrieve by embedding |
| Config introspection | 4 | Inspect current config, list valid options |
| Config drafting | 2 | Deterministic change drafts (no LLM) |
| Metrics & evidence | 3 | Authoritative performance + comparison evidence |
| Experiment intelligence | 4 | Rank, summarize, idea history, research plan |
| Session management | 5 | Create, get, list, recover sessions |
| Workflow state | 2 | State machine check, context summary |
| LLM review & proposal | 2 | Interpret diagnostics, propose next steps |
| LLM draft generation | 1 | Synthesize config change from proposal |
| Draft validation & approval | 2 | Validate + approve draft |
| YAML & execution | 2 | Render YAML + execute with "RUN" |
| Post-execution | 2 | Verify artefacts + post-run review |
| **Total** | **37** | |

Note: The operator manual counts the tool as a source of rules. Actual tool count in `_TOOLS` list should be confirmed but is approximately 36–37 based on analysis.

### Session management

`src/orchestration/session/session_manager.py` implements a clean event-log pattern:
- Session: UUID, research_goal, root_experiment, active_draft_id, status, events
- 10 event types covering the full governed workflow
- Pure state machine; zero domain knowledge
- Fully reusable for the upgraded agent

### Approval and execution governance

Two hard gates exist:
1. `approve_experiment_draft(approval_confirmation='APPROVE')` — token checked in code
2. `execute_approved_config(confirmation='RUN')` — token checked in code

These are not bypassable. The upgraded agent inherits this governance as-is.

---

## 5. Current Config and Preset Handling

### Config structure (v2 YAML)

Experiment configs are flat YAML files under `configs/experiments/`. Each config is self-contained with all parameters inline. There is no inheritance, no template system, no preset composition.

A v2 config contains: `version`, `name`, `universe.tickers`, `date_range`, `features`, `labels`, `model`, `signal`, `validation`, `execution`, `output`, `portfolio_construction`, `metadata`.

### Existing presets (implicit, not formal)

| File | What it represents |
|------|-------------------|
| `configs/experiments/canonical_ml_showcase.yaml` | Single-asset SPY Ridge regression baseline |
| `configs/experiments/equal_weight.yaml` | Multi-asset equal-weight benchmark (5 ETFs) |
| `configs/experiments/momentum_rotation.yaml` | Multi-asset momentum rotation (7 ETFs) |
| `configs/experiments/examples/` | Example configs for other strategy types |

**These are not formal presets.** They are standalone YAML files with no metadata declaring them as presets, no registry entry, and no inheritance mechanism.

### Universe definitions

`configs/universes/core_etfs.yaml` — 10 tickers (SPY, QQQ, IWM, TLT, GLD, XLF, XLK, XLE, EEM, EFA)  
`configs/universes/extended_etfs.yaml` — 15 tickers (base 10 + XLV, HYG, TIP, DBC, VNQ)

These are referenced from some configs via `universe: configs/universes/core_etfs.yaml` path string, but not from v2 ML configs (which inline the ticker list directly).

### What is configurable without code changes

- Tickers (pass any list in `universe.tickers`)
- Date range (any ISO dates)
- Features (any subset/combination from the 16 registered types)
- Model type and hyperparameters (any of the 5 registered types)
- Signal type and parameters
- Validation window sizes
- Transaction costs
- Portfolio weighting scheme

### What requires code changes to extend

- Adding a new feature type → add to `_VALID_FEATURE_TYPES` in `ml_config.py` + implement function in `src/features/`
- Adding a new model type → add to `_VALID_MODEL_TYPES` + implement class in `src/ml/models/`
- Adding a new data source → implement downloader + add to `DataAgentV1Config.supported_sources`
- Adding a new signal type → add to `_VALID_SIGNAL_TYPES` + implement in `src/signals/`
- Adding a new asset classification category → edit `_categorize_tickers()` in `src/reporting/markdown.py`
- Supporting intraday frequency → requires changes across data, features, and validation layers

---

## 6. Current Approval, Execution, and Review Flow

### Full governed workflow (current)

```
create_research_session
  → check_research_workflow_state
  → [baseline execution if needed] → execute_approved_config (RUN)
  → verify_experiment_artifacts
  → build_context_summary
  → run_experiment_review (LLM)
  → generate_iteration_proposal (LLM)
  → generate_experiment_draft (LLM) OR generate_config_change_draft (deterministic)
  → validate_experiment_draft
  → [human approval gate] → approve_experiment_draft (APPROVE)
  → render_draft_to_yaml
  → [human execution gate] → execute_approved_config (RUN)
  → verify_experiment_artifacts
  → review_post_run_result (LLM)
  → compare_experiment_metrics
  → get_session_summary
  → [human decision to iterate or complete]
```

### Execution bridge

`execute_approved_config` in `zeto_server.py` calls `_api.execute_approved_config` which calls the actual quant engine runner. The runner:
1. Loads YAML config → validates → runs backtest → computes metrics → saves artefacts
2. Optionally generates report
3. Returns `ExecutionResult(success, artefact_root, report_path, experiment_name, ...)`

The bridge is clean. The quant engine is entirely separated from orchestration.

### Review engine

Reviews are generated by `src/orchestration/llm/review_engine.py`:
- Aggregates all artefacts into `LLMContext` (deterministic, no LLM)
- Renders Jinja2 prompt template (StrictUndefined — unresolved tokens raise RuntimeError)
- Calls LLM (Claude, GPT-4, or LM Studio local model)
- Parses markdown sections from response
- Persists review JSON + context hash for provenance

The review engine is robust and reusable. The prompts are Jinja2 templates but currently stored inline in the engine files, not as separate configurable resources.

---

## 7. Current Memory and Research State Handling

### Research memory (Phase 1 — keyword/metadata)

`src/orchestration/memory/memory_indexer.py` scans `results/` for persisted artefacts and builds compact `MemoryRecord` objects (memory_id, experiment_name, artefact_type, short_summary, tags, failure_modes).

`src/orchestration/memory/memory_retriever.py` scores records by keyword match + metadata overlap (weighted additive scoring). Returns compact pointers, not full artefacts.

**Status:** Complete and reusable. Evidence-only (never authorizes execution).

### Research memory (Phase 2 — semantic embeddings)

`src/orchestration/memory/semantic_indexer.py` embeds Phase 1 records via local embedding endpoint (nomic-embed-text-v1.5 via LM Studio).

`src/orchestration/memory/semantic_retriever.py` embeds query, computes cosine similarity, returns ranked items.

**Status:** Complete and reusable. Requires local embedding endpoint.

### Comparison evidence memory

Each call to `compare_experiment_metrics` writes a `comparison_evidence.json` to `results/comparisons/<base>__vs__<candidate>/`. This record is indexed by Phase 1 and Phase 2 memory.

**Status:** Complete and reusable.

### Research intelligence (deterministic, no LLM)

Four tools built on top of artefacts and comparison evidence:
- `rank_experiments` — composite score (0.5×OOS + 0.3×Sharpe + 0.2×DD with penalties)
- `summarise_experiment_family` — aggregate metrics + common failure modes + lessons from prefix
- `check_research_idea_history` — scan comparison_evidence by keyword
- `propose_research_plan` — 5 hardcoded suggestion types driven by failure modes + evidence

**Status:** Complete and reusable. `propose_research_plan` is limited by hardcoded suggestion surface.

### Artefact registry

`src/orchestration/registry/artefact_registry.py` and `src/experiments/registry.py` track experiments by ID. Query by tag, strategy name, min Sharpe.

**Status:** Functional but limited. The registry stores experiment metadata + metrics summary. It does not store config-level metadata (model type, universe, feature families) in a queryable form.

---

## 8. Current API and Demo Script Support

### FastAPI (`src/api/`)

`src/api/app.py` exposes a FastAPI application with routers for:
- Experiment execution
- Report generation
- Data querying
- Artefact retrieval
- Orchestration intents

**Status:** Exists but scope is narrower than the MCP server. The MCP layer (`zeto_server.py`) is the primary interface for the governed AI workflow.

### Scripts (`scripts/`)

18 scripts covering data management, experiment running, comparison, reporting, and demo orchestration. Key:
- `run_from_config.py` — load any YAML config + run + report (directly reusable)
- `demo_ai_orchestration.py` — full multi-turn research cycle with Claude (references canonical showcase)
- `allocation_study.py`, `signal_geometry_study.py` — parametric sweep demos (hardcoded to canonical setups)

**Canonical ETF assumption:** Most scripts default to or hard-reference the canonical ETF configs. A new user cannot easily start a fresh research cycle from a different universe without editing script files.

---

## 9. Hardcoded Assumptions and Canonical ETF Dependencies

### Critical hardcodings

| Location | Nature | Risk | Impact |
|----------|--------|------|--------|
| `configs/experiments/canonical_ml_showcase*.yaml` (×13) | Single-asset universe: `tickers: [SPY]` | High | All iteration memory is SPY-centric |
| `src/reporting/markdown.py` — `_categorize_tickers()` | Hardcoded ticker sets for "US equities", "rates", "commodities", "sectors" | High | Reports misclassify or fail for novel tickers |
| `src/visualization/portfolio_plots.py` — `_TICKER_COLORS` dict | 24-ticker semantic color palette | Medium | Visual artifacts for custom universes |
| `src/experiments/ml_factory.py` line ~178 | `market_ticker` default = `"SPY"` for rolling_beta feature | Medium | Silently uses SPY as market proxy for any universe |
| `src/orchestration/api/research_api.py` — `_PLAN_SUGGESTIONS` | 5 hardcoded research plan suggestions (regularisation, model change, validation change, feature replacement, feature addition) | Medium | Plan proposals ignore universe, asset class, strategy type |
| `src/orchestration/config_generation/draft_validator.py` — `_VALID_CHANGE_PATHS` | 20 hardcoded (section, field) pairs | High | Adding new changeable fields requires code change |
| `scripts/demo_ai_orchestration.py` | References canonical_ml_showcase by name | Low | Demo only; not architectural |
| `src/llm/prompts.py` | Example tickers `["SPY", "QQQ"]` in prompts | Low | Illustrative; not enforced |

### What is NOT hardcoded (good practices to preserve)

- Universe passes through the stack as `list[str]` from YAML — no ticker hardcoding in core engine layers
- Data loading, feature computation, model training, backtesting, and walk-forward validation are all ticker-agnostic
- Report section rendering is parametric
- Feature types and model types use frozenset registries (not if-chains)

### Where the canonical ETF setup lives

The canonical single-asset ETF setup lives in:
1. `configs/experiments/canonical_ml_showcase*.yaml` — 13 versions, all `universe: {tickers: [SPY]}`
2. All comparison evidence in `results/comparisons/canonical_ml_showcase*/` — specific to SPY
3. Research memory index (if built) — SPY-centric artefact summaries
4. Session history — all prior research sessions anchored to canonical_ml_showcase*

The multi-asset ETF setup lives in:
1. `configs/experiments/equal_weight.yaml`, `momentum_rotation.yaml`
2. `configs/universes/core_etfs.yaml`, `extended_etfs.yaml`
3. `configs/experiments/signal_geometry/` — 15-ETF universe variants

---

## 10. Comparison Against Desired Agent Architecture

### Proposed structure vs. current state

```
config/
  schemas/
    research_spec.py       MISSING — no ResearchSpec concept exists
    experiment_plan.py     MISSING — no ExperimentPlan concept
    registry_schema.py     MISSING — frozensets exist but not as registry schemas

  registry/
    asset_classes.yaml     MISSING — implied by data contracts but no formal registry
    universes.yaml         PARTIAL — configs/universes/ has two YAML files; not queryable
    models.yaml            MISSING — model types are in _VALID_MODEL_TYPES frozenset only
    features.yaml          MISSING — feature types are in _VALID_FEATURE_TYPES frozenset only
    strategies.yaml        MISSING — strategy types are informal; no registry
    validation.yaml        MISSING — validation types are in frozenset only
    portfolio_construction.yaml  MISSING — schemes are in frozenset only
    diagnostics.yaml       MISSING — no diagnostic capability registry

  presets/
    canonical_multi_asset_etf.yaml  PARTIAL — exists as configs/experiments/equal_weight.yaml; not tagged as preset
    us_sector_etf_rotation.yaml     MISSING
    custom_ticker_template.yaml     MISSING

  defaults/
    default_policy.yaml    MISSING — no defaults system

  compatibility/
    rules.yaml             MISSING — no compatibility checking exists anywhere

agent/
  intent/
    parse_user_intent.py   PARTIAL — intent_schema.py exists; no deep parser

  resolver/
    detect_missing_fields.py   MISSING
    apply_defaults.py          MISSING
    resolve_research_spec.py   MISSING

  planner/
    create_experiment_plan.py  PARTIAL — propose_research_plan exists (5 types, limited)
    compare_with_memory.py     PARTIAL — check_research_idea_history exists

  critic/
    validate_research_design.py  PARTIAL — draft_validator.py exists (whitelist + schema)
    check_compatibility.py       MISSING

  memory/
    research_memory.py       EXISTS — Phase 1 keyword RAG (complete)
    experiment_memory.py     EXISTS — comparison_evidence + artefact registry (complete)

  routing/
    route_user_request.py    PARTIAL — check_research_workflow_state exists; no NL routing

  approvals/
    approval_policy.py       EXISTS — hardcoded in zeto_server.py governance rules

orchestration/
  sessions/               EXISTS — src/orchestration/session/ (complete)
  drafts/                 EXISTS — src/orchestration/config_generation/ (complete)
  execution/              EXISTS — execute_approved_config (complete)
  review/                 EXISTS — src/orchestration/llm/ (complete)
```

### Detailed component assessment

| Proposed Component | Current Status | Notes |
|-------------------|----------------|-------|
| ResearchSpec (incomplete user intent) | Missing | No concept; user must provide complete config |
| ExperimentPlan (hypothesis + steps) | Missing | propose_research_plan returns advisory steps but no formal plan object |
| Registry schemas | Missing | Frozensets in ml_config.py are de-facto registries but not queryable |
| Asset class registry | Missing | DataType enum exists (OHLCV, MACRO) but no "equity/fixed_income/commodity" taxonomy |
| Universe registry | Partial | Two YAML files; no formal registry class or query interface |
| Model registry | Partial | Frozenset + implementations; no structured metadata (params, constraints) |
| Feature registry | Partial | Frozenset + families.py; no structured metadata (required params, compatible strategies) |
| Strategy registry | Missing | Protocol + 4 implementations; no formal registry |
| Validation protocol registry | Missing | Frozenset of type names only |
| Preset registry | Missing | Existing YAML configs are not tagged as presets; no composition mechanism |
| Default policy | Missing | Hardcoded defaults scattered across config files and class defaults |
| Compatibility rules | Missing | No checking anywhere; engine will accept incompatible combinations until runtime |
| Intent parser | Missing (schema only) | intent_schema.py defines types but no parser |
| Missing field detector | Missing | — |
| Default applicator | Missing | — |
| Research spec resolver | Missing | — |
| Experiment planner (hypothesis-driven) | Missing (limited) | propose_research_plan is post-hoc, not hypothesis-first |
| Compatibility critic | Missing | — |
| Research design validator | Partial | draft_validator.py validates schema; not research design |
| Memory (experiment evidence) | Exists | Complete and reusable |
| Memory (research judgement) | Partial | Phase 1+2 memory indexes artefact summaries; no structured judgement capture |
| User request router | Missing | No NL-to-workflow routing |
| Approval policy | Exists | Hardcoded token gates; reusable |

---

## 11. Gap Matrix

| Desired Capability | Current Status | Evidence | Gap | Recommended Action | Priority |
|-------------------|----------------|----------|-----|--------------------|---------|
| Registry of valid models with metadata | Partial | `_VALID_MODEL_TYPES` frozenset in ml_config.py | No structured metadata (params, constraints, requirements) | Create `config/registry/models.yaml` + registry loader | P0 |
| Registry of valid feature types with metadata | Partial | `_VALID_FEATURE_TYPES` frozenset in ml_config.py | No required_params, compatible_strategies, or family classification queryable | Create `config/registry/features.yaml` + registry loader | P0 |
| Universe registry (queryable) | Partial | `configs/universes/*.yaml` (2 files) | No formal class; ticker lists not enriched with metadata | Create `config/registry/universes.yaml` + `UniverseRegistry` class | P0 |
| Asset class taxonomy | Missing | `DataType` enum (OHLCV/MACRO only) | No equity/fixed_income/commodity/alternatives taxonomy | Create `config/registry/asset_classes.yaml` | P0 |
| Preset registry | Missing | Informal experiment YAML files | No preset metadata, no composition, not discoverable | Create `config/presets/` directory with metadata headers | P0 |
| Default policy | Missing | Scattered defaults in class definitions | No single source of truth for defaults | Create `config/defaults/default_policy.yaml` | P0 |
| ResearchSpec schema (incomplete intent) | Missing | No concept exists | Agent cannot represent partial user intent | Create `agent/intent/research_spec.py` | P0 |
| Compatibility rules | Missing | No checking anywhere | Incompatible combinations fail at runtime | Create `config/compatibility/rules.yaml` + checker | P0 |
| Intent parser (NL → ResearchSpec) | Missing (schema only) | `intent_schema.py` exists | No NL parsing; no field extraction from user message | Build `agent/intent/parse_user_intent.py` (LLM-backed) | P1 |
| Missing field detector | Missing | — | Agent cannot detect what's missing from user intent | Build `agent/resolver/detect_missing_fields.py` | P1 |
| Default applicator | Missing | — | Agent cannot fill missing fields from policy | Build `agent/resolver/apply_defaults.py` | P1 |
| Research spec resolver | Missing | — | End-to-end intent → resolved spec pipeline | Build `agent/resolver/resolve_research_spec.py` | P1 |
| Hypothesis-driven experiment planner | Missing (limited) | `propose_research_plan` (5 types, post-hoc) | No hypothesis-first planning; no structured experiment plan object | Build `agent/planner/create_experiment_plan.py` | P1 |
| Compatibility critic | Missing | — | No pre-execution design validation | Build `agent/critic/check_compatibility.py` | P1 |
| Asset classification (configurable) | Partial (hardcoded) | `_categorize_tickers()` in markdown.py | Classification is hardcoded to canonical ticker sets | Move to config-driven taxonomy | P1 |
| Visualization palette (configurable) | Partial (hardcoded) | `_TICKER_COLORS` in portfolio_plots.py | 24-ticker palette is hardcoded | Accept config override or derive from universe | P2 |
| Market ticker default (configurable) | Hardcoded | ml_factory.py default="SPY" | Non-equity universes need non-SPY market proxy | Make configurable via universe config | P1 |
| Extend draft validator whitelist | Partial | 20 paths in `_VALID_CHANGE_PATHS` | New fields (universe, date_range, strategy) not changeable via drafts | Extend `_VALID_CHANGE_PATHS` for new fields | P1 |
| Multi-universe support in agent flow | Partial | Engine supports any tickers | Agent layer assumes canonical SPY setup | Add universe selection to research spec flow | P1 |
| Strategy registry | Missing | Protocol + 4 implementations | No formal registry; strategy types not queryable | Create `config/registry/strategies.yaml` + registry loader | P1 |
| Validation protocol registry | Missing | Frozenset of type names | No metadata (purpose, requirements, tradeoffs) | Create `config/registry/validation.yaml` | P2 |
| Structured research judgement memory | Partial | Comparison evidence + artefact summaries | No explicit "lesson learned" or design decision capture | Extend memory with judgement records | P2 |
| User request router (NL → workflow step) | Missing | check_research_workflow_state (artefact-based only) | Cannot route NL request to correct workflow entry point | Build `agent/routing/route_user_request.py` | P1 |
| Multi-step research planning with state | Missing | Single-iteration session model | No multi-step plan tracking; sessions are per-iteration | Extend session schema with research_plan field | P2 |
| Custom ticker template (quick-start) | Missing | No template mechanism | Users must manually construct full YAML | Create `config/presets/custom_ticker_template.yaml` | P2 |
| New data source support (beyond yfinance/FRED) | Partial | Pluggable downloader architecture exists | No additional downloaders; config doesn't expose source selection | P3 |
| Intraday frequency support | Missing | Data layer is 1d-only | Would require deep changes across stack | P3 |
| Alternative model families (RF, XGBoost) | Missing | Only linear models | Would require new implementations | P3 |

---

## 12. Recommended Build Sequence

### Phase 0: Foundation (P0 — required before agent implementation)

These create the "registry" that the agent will consult. Without them, the agent has nothing to query when detecting missing fields or proposing valid options.

1. **`config/registry/` YAML files** — models, features, strategies, universes, asset_classes, validation, portfolio_construction
   - Each file: name, description, valid_params, defaults, constraints
   - Models YAML: 5 entries with params + defaults
   - Features YAML: 16 entries with required_params, family, compatible_strategies
   - Universes YAML: 10+ named universe presets (core_etfs, extended_etfs, us_sectors, custom, etc.)
   - Asset classes YAML: equity/fixed_income/commodity/alternatives with member tickers

2. **`config/presets/` YAML files** — formal preset definitions with metadata
   - `canonical_multi_asset_etf.yaml` — the current equal_weight setup formalized as a preset
   - `single_asset_ml.yaml` — the canonical_ml_showcase formalized as a preset
   - `custom_ticker_template.yaml` — empty template with required fields marked

3. **`config/defaults/default_policy.yaml`** — single source of truth for all defaults
   - Default date range, validation protocol, model, features, transaction costs

4. **`config/compatibility/rules.yaml`** — explicit compatibility constraints
   - Examples: `rolling_beta requires equity universe`, `top_n signal requires multi-asset universe`, `LogisticRegression requires binary_direction label`

5. **Registry loader class** (`src/agent/registry/registry_loader.py`)
   - Reads all registry YAML files
   - Exposes: `get_model(name)`, `get_feature(type)`, `get_universe(name)`, `list_models()`, `list_features()`, `list_universes()`, `list_presets()`
   - Used by: intent parser, resolver, critic, MCP introspection tools

6. **`agent/intent/research_spec.py`** — ResearchSpec dataclass
   - Fields: asset_class (optional), universe (optional), preset (optional), date_range (optional), features (optional), model (optional), strategy (optional), validation (optional), hypothesis (optional), constraints (optional)
   - All fields Optional — this is the schema for incomplete intent

### Phase 1: Core Agent Loop (P1 — first implementation)

7. **`agent/intent/parse_user_intent.py`**
   - LLM-backed extraction of ResearchSpec from natural language
   - Input: user message string
   - Output: ResearchSpec (partial — only fields mentioned by user)
   - Provider-agnostic (uses existing `llm_interface.py`)

8. **`agent/resolver/detect_missing_fields.py`**
   - Deterministic: given ResearchSpec, which required fields are still None?
   - Output: list of missing fields with human-readable prompts
   - Consults registry to know what's required vs optional

9. **`agent/resolver/apply_defaults.py`**
   - Deterministic: given partial ResearchSpec, fill from default_policy.yaml
   - Returns: augmented ResearchSpec (still may be partial if user overrides disable a default)

10. **`agent/resolver/resolve_research_spec.py`**
    - Combines: parse → detect missing → apply defaults → propose valid choices → prompt user
    - Output: fully-resolved ResearchSpec OR list of clarifying questions
    - Calls: compatibility checker before finalizing

11. **`agent/critic/check_compatibility.py`**
    - Deterministic: given resolved ResearchSpec, check against compatibility/rules.yaml
    - Output: list of violations + suggested fixes

12. **`agent/planner/create_experiment_plan.py`**
    - Given resolved ResearchSpec + hypothesis, generate ExperimentPlan
    - ExperimentPlan: baseline config + 1-3 candidate iterations with rationale
    - Each step: full MLExperimentSpec (ready for YAML rendering)

13. **New MCP tools for the agent** (additions to zeto_server.py)
    - `parse_research_intent(user_message)` — returns partial ResearchSpec
    - `resolve_research_spec(partial_spec)` — returns resolved spec or clarifying questions
    - `list_research_presets()` — returns all presets from registry
    - `get_preset(name)` — returns full preset config
    - `check_research_compatibility(spec)` — returns violations
    - `create_research_plan(resolved_spec, hypothesis)` — returns ExperimentPlan

14. **Extend draft validator whitelist** — add `universe`, `date_range`, `name` to `_VALID_CHANGE_PATHS`

15. **Move `_categorize_tickers()` to registry-driven** — read from asset_classes.yaml instead of hardcoded sets

### Phase 2: Memory and Iteration (P2 — enhanced capabilities)

16. **Research judgement memory** — structured "lesson learned" records written after each review
17. **Multi-step plan tracking** — extend session schema with research_plan field
18. **`route_user_request.py`** — map NL requests to correct workflow entry point
19. **Visualization palette from config** — accept universe-derived color override

### Phase 3: Extensions (P3 — future)

20. New data sources, intraday support, tree-based models, etc.

---

## 13. Files to Reuse

These files should be used as-is or with minimal modification:

| File | Reuse Rationale |
|------|----------------|
| `src/orchestration/session/session_manager.py` | Clean event-log session management; zero domain knowledge |
| `src/orchestration/session/session_schema.py` | Session state constants; extend with new event types only |
| `src/orchestration/config_generation/draft_validator.py` | Whitelist validation; extend `_VALID_CHANGE_PATHS` |
| `src/orchestration/config_generation/draft_generator.py` | LLM + deterministic draft synthesis; reuse as-is |
| `src/orchestration/config_generation/yaml_renderer.py` | Approved draft → YAML; reuse as-is |
| `src/orchestration/config_generation/draft_schema.py` | ExperimentDraft dataclass; reuse as-is |
| `src/orchestration/context/context_builder.py` | Artefact aggregation; reuse as-is |
| `src/orchestration/context/failure_mode_detector.py` | Failure mode detection; reuse as-is |
| `src/orchestration/llm/review_engine.py` | LLM review pipeline; reuse as-is |
| `src/orchestration/llm/iteration_engine.py` | Proposal generation; reuse as-is |
| `src/orchestration/llm/llm_interface.py` | Provider-agnostic LLM calls; reuse as-is |
| `src/orchestration/memory/memory_indexer.py` | Artefact → memory records; reuse as-is |
| `src/orchestration/memory/memory_retriever.py` | Keyword retrieval; reuse as-is |
| `src/orchestration/memory/semantic_retriever.py` | Embedding retrieval; reuse as-is |
| `src/orchestration/api/research_api.py` | Top-level API composition; extend for new tools |
| `src/experiments/ml_config.py` | v2 config schema + validation; extend with new field options |
| `src/experiments/ml_factory.py` | ML experiment factory; extend market_ticker default |
| `src/validation/walk_forward.py` | Walk-forward validation; reuse as-is |
| `src/mcp/zeto_server.py` | MCP tool layer; extend with new tools |
| All of `src/data/`, `src/features/`, `src/ml/`, `src/strategies/`, `src/backtesting/` | Core quant engine; reuse completely |

---

## 14. Files to Extend

These files need targeted extension (not replacement):

| File | What to Extend | How |
|------|---------------|-----|
| `src/experiments/ml_config.py` | `_VALID_FEATURE_TYPES`, `_VALID_MODEL_TYPES` | Load from registry YAML instead of hardcoded frozensets |
| `src/orchestration/config_generation/draft_validator.py` | `_VALID_CHANGE_PATHS` | Add universe, date_range, name; load from registry |
| `src/orchestration/api/research_api.py` | New API functions | Add `parse_research_intent`, `resolve_research_spec`, `check_compatibility`, `create_experiment_plan` |
| `src/mcp/zeto_server.py` | New MCP tools | Expose new agent functions as governed tools |
| `src/orchestration/session/session_schema.py` | New event types | Add RESEARCH_PLAN_CREATED, SPEC_RESOLVED, etc. |
| `src/reporting/markdown.py` | `_categorize_tickers()` | Load from asset_classes.yaml registry |
| `src/visualization/portfolio_plots.py` | `_TICKER_COLORS` | Accept universe-derived palette override |
| `src/experiments/ml_factory.py` | `market_ticker` default | Read from universe config or strategy config |
| `docs/LM_STUDIO_QWEN_OPERATOR_MANUAL.md` | New tools section | Document new registry + resolver tools |

---

## 15. Files to Create

New files required for the upgrade:

### Registry layer
```
config/registry/
  asset_classes.yaml        — equity, fixed_income, commodity, alternatives with member tickers
  universes.yaml            — named universe presets with metadata
  models.yaml               — model types with params, defaults, constraints
  features.yaml             — feature types with required_params, family, compatible_strategies
  strategies.yaml           — strategy types with requirements
  validation.yaml           — validation protocols with guidance
  portfolio_construction.yaml  — weighting schemes with params

config/presets/
  canonical_multi_asset_etf.yaml  — multi-asset ETF rotation preset
  single_asset_ml.yaml            — single-asset ML forecasting preset
  custom_ticker_template.yaml     — empty template

config/defaults/
  default_policy.yaml       — global defaults for all fields

config/compatibility/
  rules.yaml                — compatibility constraint rules
```

### Agent layer
```
src/agent/
  __init__.py
  registry/
    __init__.py
    registry_loader.py      — loads and queries all registry YAML files
    registry_schema.py      — typed dataclasses for registry entries

  intent/
    __init__.py
    research_spec.py        — ResearchSpec dataclass (all Optional fields)
    parse_user_intent.py    — LLM extraction of ResearchSpec from NL

  resolver/
    __init__.py
    detect_missing_fields.py  — which required fields are absent
    apply_defaults.py         — fill missing fields from default_policy
    resolve_research_spec.py  — orchestrate full resolution flow

  planner/
    __init__.py
    experiment_plan.py        — ExperimentPlan dataclass
    create_experiment_plan.py — hypothesis + resolved spec → plan

  critic/
    __init__.py
    check_compatibility.py    — ResearchSpec → compatibility violations

  routing/
    __init__.py
    route_user_request.py     — NL → workflow entry point
```

### Tests
```
tests/agent/
  test_registry_loader.py
  test_research_spec.py
  test_intent_parser.py
  test_missing_field_detector.py
  test_default_applicator.py
  test_resolver.py
  test_compatibility_checker.py
  test_experiment_planner.py
```

---

## 16. Risks and Design Warnings

### Risk 1: Registry drift
If registry YAML files diverge from `ml_config.py` frozensets, validation will produce inconsistent results. The authoritative truth must be the registry (loaded at import time), not the frozensets.  
**Mitigation:** At startup, assert that frozensets match registry. Or remove frozensets entirely and derive from registry.

### Risk 2: ResearchSpec → MLExperimentSpec translation complexity
The resolved ResearchSpec maps to a v2 MLExperimentSpec. The mapping is not trivial: `asset_class="equity"` + `universe="core_etfs"` + `model="ridge"` + `strategy="ml_forecast"` must produce a valid, complete config. Errors here will reach the quant engine.  
**Mitigation:** Build a translation layer with comprehensive tests before wiring to execution. Test against all 5 model types × all 3 signal types × several universes.

### Risk 3: Compatibility rules are incomplete at launch
Compatibility rules.yaml will be incomplete initially. Some invalid combinations will pass the critic and fail at the quant engine (late failure).  
**Mitigation:** Treat the quant engine validation as a final safety net. Log all engine validation failures and add corresponding compatibility rules.

### Risk 4: LLM intent parsing is unreliable
LLM extraction of ResearchSpec from NL will sometimes misparse or miss fields. If the resolver silently fills defaults for misparse fields, the user gets a wrong spec.  
**Mitigation:** Always show the resolved spec back to the user for confirmation before generating a plan. Never proceed to execution without explicit review of the resolved spec.

### Risk 5: Whitelist expansion creates new attack surface
Expanding `_VALID_CHANGE_PATHS` to include `universe` and `date_range` means the draft generator can now propose changes that significantly change experiment identity (not just hyperparameter tuning).  
**Mitigation:** Require explicit user approval for universe or date_range changes (flag in UI/operator manual). Do not allow universe changes through the LLM draft path — only through the deterministic `generate_config_change_draft` path.

### Risk 6: Memory is SPY-centric
All existing comparison_evidence and artefact memory records are SPY-centric. A new universe-agnostic agent may retrieve SPY evidence in response to a query about a different asset class.  
**Mitigation:** Add `asset_class` and `universe` fields to MemoryRecord and SemanticRecord. Filter memory retrieval by these fields when retrieving evidence for a different universe.

### Risk 7: Large changes to zeto_server.py
Adding ~6 new MCP tools to an already 2000+ line file increases maintenance burden. Each new tool increases operator manual size (already at 7200 chars).  
**Mitigation:** Consider splitting zeto_server.py into domain modules (registry tools, resolver tools, session tools, execution tools) with a thin orchestrating server file.

### Risk 8: canonical_ml_showcase history pollution
13 versions of canonical_ml_showcase with comparison evidence create a large, specific memory footprint. When a user asks the agent "what have we learned?", this history dominates.  
**Mitigation:** Add experiment_family metadata to MemoryRecord. Allow memory retrieval to filter by family or asset class, keeping canonical_ml_showcase history scoped to SPY/ML queries.

---

## 17. Suggested Test Plan

Tests to add before and during implementation:

### P0 tests (registry layer)
```python
tests/agent/test_registry_loader.py
  - test_all_registry_files_load_without_error
  - test_model_registry_contains_five_models
  - test_feature_registry_contains_sixteen_types
  - test_universe_registry_lists_named_universes
  - test_registry_matches_ml_config_frozensets  # critical consistency check
  - test_preset_registry_has_required_fields

tests/agent/test_compatibility_checker.py
  - test_rolling_beta_requires_equity_universe
  - test_top_n_signal_requires_multi_asset_universe
  - test_logistic_regression_requires_binary_label
  - test_compatible_spec_passes_all_rules
  - test_incompatible_spec_returns_violations_with_messages
```

### P1 tests (intent + resolver)
```python
tests/agent/test_research_spec.py
  - test_research_spec_all_fields_optional
  - test_research_spec_serialization_roundtrip
  - test_partial_spec_has_missing_fields

tests/agent/test_intent_parser.py
  - test_parses_asset_class_from_user_message
  - test_parses_model_type_from_user_message
  - test_parses_date_range_from_user_message
  - test_returns_partial_spec_for_vague_input
  - test_no_invented_fields_for_absent_concepts
  - test_stub_provider_returns_deterministic_partial_spec

tests/agent/test_resolver.py
  - test_detects_missing_required_fields
  - test_applies_defaults_for_missing_optional_fields
  - test_fully_resolved_spec_passes_compatibility
  - test_resolver_does_not_overwrite_explicit_user_fields
  - test_resolver_returns_clarifying_questions_when_required_fields_absent

tests/agent/test_experiment_planner.py
  - test_plan_contains_baseline_config
  - test_plan_produces_valid_ml_experiment_spec
  - test_plan_respects_resolved_universe
  - test_plan_does_not_invent_feature_types
```

### Integration tests
```python
tests/integration/test_agent_e2e.py
  - test_full_cycle_from_intent_to_approved_config
  - test_multi_universe_experiment_runs_correctly
  - test_compatibility_violation_prevents_execution
  - test_new_universe_memory_does_not_pollute_canonical_etf_memory
```

---

## 18. Final Recommendation

### Build first (P0)
Start with the registry layer — this is the foundation everything else depends on. Write the 7 registry YAML files, the registry loader class, and the registry consistency tests. No agent code should be written until the registries are queryable and tested.

### Reuse aggressively (P1)
The quant engine, execution bridge, session management, draft validation, LLM interface, and both memory phases are production-quality and should be reused without modification. The upgrade adds layers above and around the existing engine, not inside it.

### Do not touch yet
- `src/data/`, `src/features/`, `src/ml/`, `src/strategies/`, `src/backtesting/` — the quant engine
- `src/orchestration/session/`, `src/orchestration/config_generation/`, `src/orchestration/llm/` — clean and reusable
- The 36 existing MCP tools — only add new tools, do not modify existing ones

### Refactor carefully
- `src/reporting/markdown.py` — `_categorize_tickers()` must be driven by asset_classes.yaml, not hardcoded sets. This change touches report output but is straightforward.
- `src/experiments/ml_config.py` — frozensets should be loaded from registry YAML (or at minimum, tests should assert consistency). Do not change before the registry is built.
- `src/mcp/zeto_server.py` — add new tools in a new block; do not reorganize existing tools.

### Biggest risks
1. Registry/frozenset drift — build the consistency assertion test first
2. LLM intent parsing quality — show resolved spec to user before any plan generation
3. Whitelist expansion — restrict universe changes to deterministic draft path only
4. Memory pollution from SPY history — add universe/asset_class filters to MemoryRecord immediately

### Most important tests to add first
1. Registry consistency test (registry YAML ↔ ml_config.py frozensets)
2. Compatibility checker against all known valid and invalid combinations
3. Resolver roundtrip (NL → partial spec → defaults → resolved spec → valid config)
4. End-to-end: "I want to research US sector ETFs using Ridge regression" → approved YAML

### Summary judgment

The platform is an excellent foundation for the upgrade. The upgrade is a **greenfield agent layer** sitting on top of a proven engine — not a rewrite. The total new code volume is moderate (8–12 new Python files, 7 registry YAML files, ~30 new tests). The highest-leverage first step is the registry layer: once the agent can query "what models exist?", "what features are compatible with this universe?", and "what defaults apply?" — all other capabilities follow naturally.
