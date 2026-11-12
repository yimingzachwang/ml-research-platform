# Machine Learning Research & Experimentation Platform

A modular Python research platform for reproducible machine learning experimentation, time-series prediction, out-of-sample generalisation assessment, and AI-assisted research workflows. It combines supervised learning, leakage-aware feature and label pipelines, walk-forward validation, model diagnostics, and provenance-tracked experiments, using financial time series as a real-world research testbed.

[![CI](https://github.com/yimingzachwang/ml-research-platform/actions/workflows/ci.yaml/badge.svg)](https://github.com/yimingzachwang/ml-research-platform/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Test modules](https://img.shields.io/badge/test_modules-173-blue.svg)](tests/)

## Overview

The platform connects three parts of applied ML research:

- **Machine learning methodology:** deterministic feature engineering, explicit prediction targets, regularised supervised models, temporal evaluation, and model stability analysis.
- **Research engineering:** typed data contracts, dataset registries, declarative experiment configurations, persisted artefacts, automated tests, and reproducible reporting.
- **AI-assisted experimentation:** structured LLM research reviews, experiment comparisons, configuration proposals, and human-governed research iteration.

Financial time series provide noisy observations, temporal dependencies, changing distributions, and limited predictive signal. These properties make ETF datasets a demanding environment for examining model behaviour and assessing generalisation across historical periods. Regime-conditional diagnostics describe variation in observed performance; they do not establish robustness to future distribution changes.

Portfolio construction and historical backtesting provide downstream evaluation of predictions. The research focus is the relationship between data, model assumptions, temporal validation, and reproducible evidence.

## ML Research Showcase

The linked reports document completed experiments. Their datasets, model settings, diagnostics, and results describe those saved runs; current YAML configurations may have evolved since report generation.

### Canonical ML Showcase

[Research report](reports/markdown/canonical_ml_showcase.md) · SPY · 2013–2024

Supervised Ridge regression predicts 21-day forward returns from seven features covering momentum, realised volatility, rolling z-score normalisation, and trend strength. The experiment demonstrates historical feature construction, explicit feature-label alignment, L2 regularisation, and rolling walk-forward evaluation with 48-month training and 12-month test windows.

The report examines out-of-sample performance alongside coefficient magnitude and sign stability, feature-family information coefficients, and regime-dependent behaviour. These diagnostics expose unstable predictive relationships and qualify the aggregate performance results.

### Canonical ML Multi-Asset

[Research report](reports/markdown/canonical_ml_multi_asset.md) · 9-ETF cross-sectional universe · 2013–2024

A shared Ridge model learns from pooled date–asset observations to predict 21-day cross-sectional return ranks. The saved report describes 15 engineered features across SPY, QQQ, IWM, EEM, TLT, GLD, XLF, XLK, and XLE. Features include multi-horizon momentum, risk-adjusted momentum, rolling market beta, volatility compression, trend persistence, drawdown distance, and breakout proximity.

Evaluation includes cross-sectional rank information coefficients, prediction-score grouping, coefficient stability, multi-asset walk-forward diagnostics, and top-N signal translation. Portfolio simulation examines how ranking predictions translate into allocation, turnover, and returns after transaction costs.

### Controlled Research Studies

| Study | Controlled comparison | Research emphasis |
|---|---|---|
| [Signal geometry](reports/signal_geometry/) | Four Ridge configurations varying regularisation strength α | Prediction dispersion, ranking behaviour, and walk-forward stability under a fixed experimental design. |
| [Allocation sensitivity](reports/allocation_study/) | Equal-weight allocation and three softmax temperature settings | The relationship between prediction-score dispersion, portfolio concentration, turnover, and downstream performance. |

The [signal geometry synthesis](reports/signal_geometry/signal_geometry_synthesis.md) and [allocation synthesis](reports/allocation_study/allocation_study_synthesis.md) retain the experiment matrices and diagnostic findings, including limited sensitivity within the tested settings. Their conclusions are specific to those datasets and configurations.

### Non-ML Financial Research Application

[Momentum Rotation](reports/markdown/example_momentum_rotation.md) · 7-ETF universe · 2015–2024

A rule-based trailing 252-day momentum signal with month-end rebalancing demonstrates the non-ML research path. Its report covers signal construction, portfolio rules, walk-forward evaluation, and risk factors such as momentum reversals, concentration, lookback sensitivity, and transaction-cost drag. It provides a baseline methodology alongside the supervised learning examples, rather than a matched benchmark with identical experimental settings.

## Machine Learning Methodology

```text
Data ingestion → Feature engineering → Label construction → Model training
    → Temporal validation → Model diagnostics → Experiment comparison → Research reporting
```

### Features, Targets, and Temporal Alignment

[`src/features/`](src/features/) implements historical transformations grouped into trend, volatility, mean-reversion, and market-structure families. [`src/ml/`](src/ml/) composes these into feature matrices and supervised datasets. Single-asset datasets use date-indexed observations; panel utilities pool per-asset features under a date–asset index.

Labels are constructed separately from features. Implemented targets include forward returns, binary return direction, forward volatility, cross-sectional return ranks, and risk-adjusted return ranks. Dataset construction aligns features and labels by index and removes incomplete rows, including feature warm-up periods and unavailable forward targets.

Leakage prevention depends on both transformation choice and training boundaries. In the config-driven ML strategy path, labels are constructed from each training price slice, so targets requiring prices beyond that slice remain unavailable and are removed before fitting. Rolling transformations use historical windows; full-series normalisation utilities are intended for diagnostics and must not be treated as leakage-safe time-series preprocessing. Label horizons, data timing, and split gaps remain explicit research assumptions.

### Supervised Models and Regularisation

Implemented scikit-learn adapters cover ordinary least squares, Ridge, Lasso, ElasticNet, and logistic regression. Models expose a typed `fit`/`predict` protocol using `SupervisedDataset` and `PredictionSeries`; logistic regression returns positive-class probability scores. L1 and L2 penalties provide configurable regularisation for the implemented linear models.

The completed ML showcases above use Ridge regression. The model protocol provides an extension point; additional model families require separate implementation and validation.

### Temporal Validation and Generalisation Assessment

[`src/validation/`](src/validation/) generates chronological rolling or expanding train/test splits with configurable calendar-day gaps. Walk-forward evaluation refits models on successive training windows and evaluates subsequent test periods. The validation runner also supports periodic retraining within outer test windows, with the retraining frequency configured separately from portfolio rebalancing.

Split-level metrics, stitched out-of-sample results, and stability summaries assess how behaviour varies across historical periods. A strong aggregate result does not establish consistent performance across splits or guarantee future generalisation. Experiment reports retain assumptions, failure analysis, and regime-conditional diagnostics alongside headline metrics.

### Diagnostics and Controlled Comparison

Implemented model diagnostics include regression errors, prediction correlation, directional accuracy, cross-sectional Spearman information coefficients, coefficient dispersion and sign consistency, rolling prediction means, and signal turnover. Reports add feature correlations, feature-family IC summaries, prediction-score outcome analysis, and regime-conditional comparisons where the required artefacts are available.

Declarative YAML configurations specify features, labels, model parameters, signal translation, validation, and execution assumptions. The [`src/research_search/`](src/research_search/) framework enumerates controlled comparisons of Ridge, Lasso, and ElasticNet across feature groups and transaction-cost settings. These comparisons and the showcase studies make experimental choices inspectable; they do not constitute a general-purpose automated hyperparameter optimiser.

## AI-Assisted Research Orchestration

The AI layer helps researchers interpret experimental evidence and propose subsequent work. LLM reasoning is separated from deterministic configuration resolution, validation, and numerical experiment execution. Workflows are researcher-initiated, and generated drafts require explicit human approval before rendering to experiment YAML.

| Capability | Implemented workflow |
|---|---|
| Research reviews | LLM interpretation of structured artefacts covering performance, signal quality, validation, failure modes, feature contribution, and recommendations. OpenAI, Anthropic, and stub providers share an interface. |
| Experiment comparison | Deterministic metric tables and differences, with LLM comparative reviews grounded in the available metrics, features, universes, and validation outputs. |
| Configuration synthesis | Typed `ExperimentDraft` objects describe proposed changes. Allowed change paths, schema checks, provenance hashes, and approval state constrain rendering. |
| Research design | `src/agent/` extracts natural-language intent into a structured specification, then resolves supported options, checks compatibility, constructs a plan, and maps supported changes to a draft deterministically. |
| Sessions and iteration tracking | Persisted session events, experiment lineage, and evolution summaries record reviews, proposals, draft states, configuration changes, and linked experiments. |
| Research memory | Historical artefact indexing and retrieval provide related evidence. Memory-aware next-action planning generates recommendations and specification proposals without executing them. |
| Governed interfaces | FastAPI and MCP expose research operations through typed interfaces. Proposal-to-draft workflows require explicit confirmation; draft approval and experiment execution remain separate researcher actions. |

The LLM receives curated context assembled from persisted artefacts. Reviews and recommendations interpret that evidence; numerical results come from the research engine. The agent workflow does not autonomously approve drafts or run a closed research loop.

See the [orchestration overview](docs/ai_orchestration/overview.md) and [governed research agent architecture](docs/agent_architecture.md) for subsystem responsibilities and workflow boundaries.

## Research Infrastructure

The architecture connects data preparation, supervised learning, experiment execution, and reporting through modular Python packages. Typed exchange objects and explicit configuration keep research assumptions visible. The FastAPI bridge calls the Research API, with tests checking that routers do not reach directly into numerical engine internals.

The existing layered architecture is shown below. ML training and diagnostics sit alongside the portfolio components that evaluate their outputs.

```
┌──────────────────────────────────────────────────────────────────────┐
│  API Layer                                                           │
│  FastAPI bridge · Research API · typed request/response schemas      │
│  HTTP transport only — no business logic in routing                  │
├──────────────────────────────────────────────────────────────────────┤
│  Orchestration Layer                                                 │
│  LLM review engine · config synthesis · session management           │
│  intent parsing · research routing · context building · retrieval    │
├──────────────────────────────────────────────────────────────────────┤
│  Reporting · Visualization                                           │
│  Provenance-tracked markdown/HTML reports · IC diagnostics           │
│  Publication-quality figures · coefficient and regime plots          │
├──────────────────────────────────────────────────────────────────────┤
│  Experiment · Backtesting                                            │
│  YAML config orchestration · MLflow tracking · artefact persistence  │
│  Vectorized historical simulation · walk-forward evaluation          │
├──────────────────────────────────────────────────────────────────────┤
│  ML · Portfolio · Strategies                                         │
│  Model training/inference · walk-forward pipelines · diagnostics     │
│  Cross-sectional portfolio construction · allocation · constraints   │
├──────────────────────────────────────────────────────────────────────┤
│  Features · Cleaning · Signals                                       │
│  Leakage-safe deterministic transformations · family taxonomy        │
│  Timestamp normalisation · missing value policy · signal translation │
├──────────────────────────────────────────────────────────────────────┤
│  Data Layer                                                          │
│  Typed contracts · dataset registry · download adapters              │
│  Schema validation · freshness checks · DatasetQuery resolution      │
└──────────────────────────────────────────────────────────────────────┘
```

[Architecture diagrams](docs/architecture/) cover the dependency graph, ML experiment lifecycle, validation workflow, and artefact persistence.

### Data and Experiment Artefacts

The data layer provides typed requests, schema validation, dataset manifests, and registry-backed `DatasetQuery` resolution. Manifests identify sources, frequencies, date coverage, schema versions, storage paths, and request hashes. ETF examples use Yahoo Finance data through `yfinance`, persisted locally as Parquet.

Experiment persistence is filesystem-first. Configuration snapshots, metrics, portfolio outputs, ML diagnostics, and research figures are saved for subsequent comparison and reporting. ML runs include an `ml_provenance.json` sidecar with the ML specification hash and model, feature, label, signal, and allocation settings. Report generation reads persisted artefacts and writes Markdown, optional HTML, a manifest, and a provenance sidecar; available fields depend on the saved run.

An optional MLflow adapter records run parameters, metrics, and artefact manifests when enabled. This integration complements local persistence; it is not required for every experiment path.

### Repository Structure

```text
ml-research-platform/
├── agents/          # AI research agent role definitions and collaboration protocols
├── configs/         # Declarative YAML for data, features, models, experiments, portfolios
├── data/            # Local data lake (raw · processed · features · external — contents gitignored)
├── docker/
├── docs/
│   ├── architecture/        # Layer design, extension patterns, Mermaid system diagrams
│   ├── ai_orchestration/    # Orchestration layer design and rationale
│   ├── diagrams/            # Rendered system diagrams
│   ├── manuals/             # Platform manual, reporting framework, visualization philosophy
│   ├── DESIGN_PRINCIPLES.md
│   └── VISUALIZATION_PHILOSOPHY.md
├── reports/         # Executed experiment output (figures · markdown · HTML · provenance JSON)
│   ├── allocation_study/
│   ├── figures/
│   ├── html/
│   ├── markdown/
│   └── signal_geometry/
├── results/         # Runtime artefact storage (gitignored; see results/README.md)
├── scripts/         # Runnable research workflow entrypoints
├── skills/          # Reusable AI-agent procedures for repeatable research tasks
├── src/
│   ├── agent/           # Research intent resolution, planning, and governed draft proposals
│   ├── api/             # FastAPI bridge and HTTP routers
│   ├── backtesting/     # Vectorised backtesting and performance evaluation
│   ├── cleaning/        # Data cleaning and normalisation
│   ├── data/            # Data contracts, registry, download adapters, loaders
│   ├── experiments/     # Config loading, orchestration, MLflow tracking hooks
│   ├── features/        # Deterministic historical feature transformations
│   ├── ml/              # Models, walk-forward pipelines, diagnostics, signals
│   ├── mcp/             # MCP interfaces to governed research workflows
│   ├── orchestration/   # Research reviews, comparisons, sessions, and approval workflows
│   ├── portfolio/       # Cross-sectional portfolio construction and allocation
│   ├── research_search/ # Controlled model, feature-group, and cost comparisons
│   ├── reporting/       # Artefact-driven report generation
│   ├── signals/         # Signal construction and translation
│   ├── strategies/      # Strategy definitions and cross-strategy comparison
│   ├── validation/      # Walk-forward splits and out-of-sample stability diagnostics
│   └── visualization/   # Publication-quality figure rendering
├── tests/           # 173 test modules covering research and interface contracts
├── .github/
├── AGENTS.md
├── Dockerfile
├── docker-compose.yaml
└── pyproject.toml
```

## Engineering Quality

The test suite covers data contracts, feature transformations, supervised datasets, models, temporal validation, experiment persistence, reporting, orchestration, and API boundaries. Unit, contract, integration, and architectural tests separate implementation correctness from claims about predictive performance.

A static inventory of this revision contains:

| Scope | Python files | Physical lines, including comments and blank lines |
|---|---:|---:|
| `src/` | 286 | 55,829 |
| `tests/` | 193, including 173 `test_*.py` modules | 55,347 |

There are 4,827 test function definitions before pytest parametrisation. The [earlier engineering audit](docs/agent_upgrade_audit.md) records a baseline of 2,271+ passing tests; it is a historical result, not a current test-run claim.

[`tests/api/test_api_non_coupling.py`](tests/api/test_api_non_coupling.py) checks all five router modules for forbidden direct imports, experiment execution calls, automatic lineage registration, and background-task patterns. These are explicit API boundary checks rather than a claim that every dependency in the repository is enforced by a universal layer rule.

Ruff checks linting and import ordering; Black handles formatting; `mypy` is configured in strict mode for `src/`. Pre-commit hooks run Ruff, formatting, and pytest. [CI](.github/workflows/ci.yaml) runs Ruff, Black, and pytest on Python 3.11 for pull requests and pushes to `main`.

## Financial Research Applications

The implemented application workflows include ETF return forecasting, cross-sectional asset ranking, portfolio construction, historical backtesting, and strategy comparison. Prediction outputs can be translated into top-N selection and alternative allocation rules, with drawdown, turnover, transaction-cost, and regime-conditional analysis used to examine downstream consequences.

The platform is intended for research rather than live trading. It has no live broker integration, order management, or live position tracking. Historical model and portfolio results support research review; they are not deployment guarantees.

## Installation and Usage

Clone the repository and create a local Python environment:

```bash
git clone https://github.com/yimingzachwang/ml-research-platform.git
cd ml-research-platform
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[research,dev]"
pre-commit install
```

For LLM-assisted workflows, copy the environment template and supply provider API keys:

```bash
cp .env.example .env
# Edit .env — add OPENAI_API_KEY and/or ANTHROPIC_API_KEY
```

Run the full check suite:

```bash
ruff check .
black --check .
pytest
```

Run a config-driven experiment end-to-end:

```bash
python scripts/run_from_config.py configs/experiments/canonical_ml_showcase.yaml --report --preset canonical
```

Use absolute imports throughout:

```python
from src.data.loaders import load_dataset
from src.features import compute_returns, rolling_volatility
from src.ml.models.linear import RidgeRegressionModel
from src.orchestration.api import research_api
```

---

## Docker

Build and run the research environment in a container:

```bash
docker compose run --rm research
```

Start MLflow experiment tracking locally:

```bash
docker compose up mlflow
```

---

## Data Preparation

Data directories (`data/raw/`, `data/processed/`, `data/features/`, `data/external/`) are gitignored. The repository includes directory placeholders and registry infrastructure; populate data locally using the provided scripts:

```bash
python scripts/update_dataset.py --config configs/data/daily_prices.yaml
python scripts/validate_data.py
python scripts/list_datasets.py
```

Raw, processed, and feature datasets occupy separate paths. Retain the source data and registry metadata alongside experiment configurations when reproducing a historical run.
