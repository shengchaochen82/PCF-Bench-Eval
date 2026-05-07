<div align="center">

# PCF-Bench

### PCFBench: How Far Can Large Vision-Language Models Go in Physics-aware Photonic Inverse Design?

[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Track](https://img.shields.io/badge/NeurIPS%202026-Datasets%20%26%20Benchmarks-purple.svg)](#)
[![Tasks](https://img.shields.io/badge/tasks-28-orange.svg)](#-task-catalogue)
[![Models](https://img.shields.io/badge/providers-4-lightgrey.svg)](#-supported-providers)

A configuration-driven evaluation harness for **28 vision-language reasoning tasks** spanning geometry perception, physics understanding, multimodal reasoning, inverse design, and code generation, on a domain-specific benchmark**.

[**Quickstart**](#-quickstart) ·
[**Tasks**](#-task-catalogue) ·
[**CLI**](#-cli-reference) ·
[**Add a model**](#-evaluate-your-own-model) ·
[**Cite**](#-citation)

</div>

---


## ✨ Highlights

| | |
| --- | --- |
| 🧩 **28 tasks, 6 categories** | Geometry · Physics · Text · Multimodal · Inverse Design · Code Generation, plus a 4-stage MAS pipeline |
| 🔌 **OpenRouter, multi-key pool** | One provider routes to 200+ frontier and open-weight VLMs; round-robin across an arbitrary number of OpenRouter keys |
| 🤖 **12 MAS baselines** | DyLAN · MacNet · AutoGen · EvoMAC · MAV · AgentVerse · CAMEL · Self-Consistency · CoT · LLM-Debate · MAD · Vanilla |
| 🎛 **One config, one CLI** | Single pydantic-validated YAML drives every run; eight subcommands cover the full life-cycle |
| 📊 **Standard outputs** | Each run emits `<id>.json`, `<id>.csv` (leaderboard-ready), and `<id>.md` (human summary) |
| 🧪 **Reproducible** | Pin samples with a stratified-subset manifest; every output records the full config |

---

## 📚 Table of Contents

1. [Quickstart](#-quickstart)
2. [Installation](#-installation)
3. [Task catalogue](#-task-catalogue)
4. [CLI reference](#-cli-reference)
5. [Configuration](#-configuration)
6. [Supported provider](#-supported-provider)
7. [Multi-agent (MAS) evaluation](#-multi-agent-mas-evaluation)
8. [Evaluate your own model](#-evaluate-your-own-model)
9. [Repository layout](#-repository-layout)
10. [FAQ](#-faq)
11. [License](#license)

---

## 🚀 Quickstart

```bash
# 1. install (replace <repo-url> with the URL displayed at the top of the page)
git clone <repo-url> PCF-Bench-Eval && cd PCF-Bench-Eval
pip install -e .

# 2. set provider keys (any number; supports round-robin)
export OPENROUTER_API_KEY_1=sk-or-...
export OPENROUTER_API_KEY_2=sk-or-...

# 3. scaffold a config (the dataset is materialized automatically on first eval)
pcfbench init my_run --dataset PCF-Bench/PCF-Bench
# or, for a local parquet copy:
# pcfbench init my_run --dataset '/data/PCF-Bench-HF/data/full-*.parquet'

# 4. run
pcfbench eval --config my_run.yaml
#   first run does a one-time materialization into output_dir/.cache/<slug>;
#   subsequent runs reuse it. Override with cache_dir: in the YAML if needed.
```

You'll see live progress per task, then artefacts under `results/`:

```
results/
├── my_run_2026...Z.json    # full record (config + per-sample results)
├── my_run_2026...Z.csv     # one row per task, leaderboard-friendly
└── my_run_2026...Z.md      # human-readable summary
```

> **Tip.** No install? `./bin/pcfbench …` works straight from the checkout — same CLI, just runs the local source.

---

## 📦 Installation

**Requirements**: Python ≥ 3.9.

```bash
pip install -e .                 # core
pip install -e .[mas]            # + scipy/Pillow for some MAS baselines
```

**One-liner for a clean conda env**:

```bash
conda create -n pcfbench python=3.10 -y && conda activate pcfbench
pip install -e .
```

Verify the install:

```bash
pcfbench info     # shows provider key detection + python/openai/pydantic versions
pcfbench tasks    # lists all 28 registered tasks
```

---

## 🧩 Task catalogue

PCF-Bench organises the 28 tasks into a four-tier hierarchy: **Perception → Understanding → Reasoning → Design**. Each task ships an instruction prompt, an output extractor, and a primary metric. Use `pcfbench tasks` to print the live registry; the table below summarises by category.

<details>
<summary><b>Geometry Perception</b> — 5 tasks</summary>

| # | Task name | Image | Primary metric |
|---|---|---|---|
| 1 | `family_classification` | structure | Accuracy + Macro-F1 |
| 2 | `parameter_prediction_pitch` | structure | NRMSE↓ |
| 3 | `parameter_prediction_hole_d` | structure | NRMSE↓ |
| 4 | `structure_counting_rings` | structure | Exact-match Acc |
| 5 | `topology_recognition` | structure | Accuracy + Macro-F1 |
</details>

<details>
<summary><b>Physics Understanding</b> — 5 tasks</summary>

| # | Task name | Image | Primary metric |
|---|---|---|---|
| 6 | `loss_prediction_1310` | field_1310 | Log-MAE↓ |
| 7 | `loss_prediction_1550` | field_1550 | Log-MAE↓ |
| 8 | `application_classification` | field_1550 | Accuracy + Macro-F1 |
| 9 | `mode_classification` | field_1550 | Accuracy + Macro-F1 |
| 10 | `mechanism_reasoning` | structure | Mean keyword recall |
</details>

<details>
<summary><b>Text Generation</b> — 3 tasks</summary>

| # | Task name | Image | Primary metric |
|---|---|---|---|
| 11 | `geometry_description` | structure | BERTScore-F1 |
| 12 | `conversation_qa` | field_1550 | BERTScore-F1 |
| 13 | `report_generation` | structure + field_1550 | Completeness |
</details>

<details>
<summary><b>Multimodal Reasoning</b> — 6 tasks</summary>

| # | Task name | Image | Primary metric |
|---|---|---|---|
| 14 | `wavelength_comparison_v1` (Paired) | structure + field_1310 + field_1550 | Accuracy |
| 15 | `wavelength_comparison_v2` (Field-only) | field_1310 + field_1550 | Accuracy |
| 16 | `mechanism_diagnosis_v1` (Paired) | structure + field_1550 | Macro-F1 |
| 17 | `mechanism_diagnosis_v2` (Field-only) | field_1550 | Macro-F1 |
| 18 | `evidence_explanation_v1` (Paired) | structure + field_1550 | BERTScore-F1 |
| 19 | `evidence_explanation_v2` (Field-only) | field_1550 | BERTScore-F1 |
</details>

<details>
<summary><b>Inverse Design</b> — 4 tasks</summary>

| # | Task name | Image | Primary metric |
|---|---|---|---|
| 20 | `parameter_selection_v1` | structure + field_1550 | Accuracy + Macro-F1 |
| 21 | `design_direction_pitch` | structure + field_1550 | Acc + binary F1 |
| 22 | `design_direction_hole_d` | structure + field_1550 | Acc + binary F1 |
| 23 | `closed_loop_improvement_pitch` | structure + field_1550 | Success Rate |
</details>

<details>
<summary><b>Code Generation</b> — 3 tasks</summary>

| # | Task name | Image | Primary metric |
|---|---|---|---|
| 24 | `simulation_code_generation_complete` | structure | Quality score Q |
| 25 | `parameter_modification_complete` | structure | Syntax + completeness |
| 26 | `code_refactoring` | structure | Q<sub>refactor</sub> |
</details>

<details>
<summary><b>Multi-Agent Pipeline</b> — 2 tasks</summary>

| # | Task name | Image | Primary metric |
|---|---|---|---|
| 27 | `mas_pipeline_cascaded` | full | E2E success + per-stage breakdown |
| 28 | `mas_pipeline_oracle` | full | E2E success + per-stage breakdown |
</details>

---

## 🛠 CLI reference

A single command, nine subcommands:

```
pcfbench info                          Show provider keys + Python env diagnostic.
pcfbench fetch --out-dir D             Pre-materialize the dataset (optional;
       (--repo-id REPO  | --shards-glob 'PATH/full-*.parquet')                eval auto-materializes on first run).
       [--max-samples N] [--streaming]
pcfbench tasks [--filter STR]          List the 28 registered tasks.
pcfbench init   NAME                   Scaffold a YAML config from a template.
       [--template TPL] [--path P] [--dataset SRC] [--force]
pcfbench validate CONFIG               Type-check a config.
pcfbench eval --config CONFIG          Run a configured evaluation.
       [--override-model M]              swap cfg.model for sweeps
       [--override-dataset SRC]          swap cfg.dataset (HF repo or parquet glob)
       [--override-max-samples N]
       [--dry-run]                       same as `validate`
pcfbench score RUN.json [--output-dir D]   Re-render JSON → CSV + Markdown.
pcfbench compare LEFT.json RIGHT.json   Print a Δ-table between two runs.
pcfbench mas-eval --method M --data_dir D     Run a MASLab multi-agent baseline
       --model M [--method_config NAME]          (DyLAN, MacNet, AutoGen, EvoMAC,
       [--maslab_config FILE.json]               CAMEL, MAV, AgentVerse, MAD,
       [--max_samples N] [--num_workers W]       LLM-Debate, Self-Consistency,
       [--no-score] [--output_dir D]             CoT, Vanilla) on PCFBench data
                                                 and score it.
```

**Examples**:

```bash
pcfbench tasks --filter physics              # list physics-understanding tasks
pcfbench eval --config my.yaml \
              --override-model gpt-4o \
              --override-max-samples 100     # quick smoke test
pcfbench compare results/baseline.json results/improved.json
```

---

## 🎛 Configuration

The full schema lives in `pcfbench/config.py`. Unknown fields raise `ValidationError` immediately so typos are caught before any model call.

```yaml
run_name: my_run

# Parquet input — HuggingFace repo id, or a local glob over parquet shards.
dataset: PCF-Bench/PCF-Bench
# dataset: /data/PCF-Bench-HF/data/full-*.parquet

# Optional. Default: '<output_dir>/.cache/<dataset_slug>'.
# cache_dir: /scratch/pcfbench-cache

output_dir: results

provider:
  name: openrouter           # the only supported provider in the public release
  timeout_s: 120
  max_retries: 3

model: openai/gpt-4o-mini    # any OpenRouter model id

api_keys_env:                # round-robin pool; up to 100 keys
  - OPENROUTER_API_KEY_1
  - OPENROUTER_API_KEY_2

tasks: all                   # or a list, e.g. [family_classification, loss_prediction_1550]
max_samples: 200
# subset_manifest: examples/subset_v1.json  # optional stratified subset

runner:
  num_workers: 16
  rpm_limit: 600
  temperature: 0.1
  max_tokens: 2048

# mas:                        # only for tasks 27/28
#   mode: pcfbench            # or maslab
#   method: dylan
```

Three ready-to-use templates ship in `examples/`:

| Template | Purpose |
| --- | --- |
| `examples/single_model.yaml` | one model on all tasks (default) |
| `examples/multi_model.yaml`  | sweep skeleton — pair with `--override-model` |
| `examples/mas.yaml`          | sequential MAS pipeline or MASLab-bridged method |

---

## 🔌 Supported provider

The public release routes every model through **OpenRouter** so that the same 41-VLM cohort can be evaluated under one OpenAI-compatible API without per-vendor SDK plumbing. Supply one or more `OPENROUTER_API_KEY_*` environment variables and the round-robin pool will use them all.

| Provider | env-var pattern | base URL |
| --- | --- | --- |
| `openrouter` | `OPENROUTER_API_KEY_*` | `https://openrouter.ai/api/v1` |

Set `model:` in your config to any OpenRouter id (`openai/gpt-4o-mini`, `anthropic/claude-sonnet-4-5`, `google/gemini-2.0-flash`, `qwen/qwen-vl-max`, ...). The client implementation is `OpenRouterClient` in `tools/llm_client.py`; subclass `BaseLLMClient` if you need a private endpoint.

---

## 🤖 Multi-agent (MAS) evaluation

`MAS/` ships the 12-baseline MASLab cohort plus the native PCFBench pipeline. Two complementary entry points:

**(a) Native sequential pipeline** — Perception → Understanding → Reasoning → Design, implemented as the registered tasks `mas_pipeline_cascaded` and `mas_pipeline_oracle` (run via the standard `pcfbench eval`):

```bash
pcfbench init mas_native --dataset PCF-Bench/PCF-Bench
# edit tasks: [mas_pipeline_cascaded, mas_pipeline_oracle]
pcfbench eval --config mas_native.yaml
```

**(b) MASLab baselines** — DyLAN, MacNet, AutoGen, EvoMAC, MAV, AgentVerse, CAMEL, Self-Consistency, CoT, LLM-Debate, MAD, Vanilla. Each consumes the same enriched query format and is scored on the same PSel / Direction / Closed-Loop rubric:

```bash
pcfbench mas-eval --method dylan \
    --model gpt-4o-mini \
    --data_dir results/.cache/<your-slug> \
    --max_samples 200
```

The wrapper invokes `scripts/mas/run_mas_benchmark.py` (per-sample inference) followed by `scripts/mas/score_maslab_results.py` (PSel + Direction + CL accuracy) and writes a `*_scored.json` next to the raw output. Method classes live in `MAS/methods/<method>/` following the upstream MASLab layout. Human Score (HS) is a separate expert-rated metric (Appendix~\ref{app:human_eval} in the paper) and is not computed by this script.

---

## 🚢 Evaluate your own model

If your model already speaks an OpenAI-compatible chat-completion API (with vision support for image-conditioned tasks), point pcfbench at it:

```bash
# Option A: behind one of the four built-in providers
export OPENROUTER_API_KEY_1=sk-or-...
pcfbench eval --config my.yaml --override-model openai/your-model-id

# Option B: self-hosted endpoint
# in your config:
provider:
  name: openrouter
  base_url: http://localhost:8000/v1
model: your-local-model
```

For models with non-OpenAI APIs, subclass `BaseLLMClient` in `tools/llm_client.py`, register it under a new provider name, and update `EvalConfig.provider.name` to accept it. About 50 LOC.

---

## 📂 Repository layout

```
PCFBench_Eval/
├── pcfbench/             ← public package: CLI + runner + config + reporting
│   ├── cli.py            8 subcommands
│   ├── config.py         pydantic v2 schema
│   ├── providers.py      multi-key round-robin client
│   ├── runner.py         orchestrator over eval/tasks/*
│   └── reporting.py      JSON / CSV leaderboard / Markdown
├── eval/                 ← task and metric implementations
│   ├── tasks/            21 task classes, 28 registered task names
│   └── metrics/          classification / regression / generation / code
├── tools/                ← minimal supporting modules
│   ├── llm_client.py     4-provider OpenAI-compatible base
│   ├── env_utils.py      API-key / worker / RPM discovery
│   └── io_utils.py       atomic-write JSON
├── MAS/                  ← bundled multi-agent baselines
├── bin/pcfbench          ← zero-install launcher
├── examples/             ← 3 starter configs
├── pyproject.toml
├── requirements.txt
└── README.md             ← you are here
```


---

## License

The evaluation suite is released under the MIT License. The PCF-Bench dataset is released separately under its own license; see the dataset repository for details.

---

<div align="center">
<sub>Built for NeurIPS 2026 D&B-track. Issues, discussions, and pull requests welcome.</sub>
</div>
