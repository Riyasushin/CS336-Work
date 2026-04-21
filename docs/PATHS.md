# Path mapping: original assignments → this repo

CS336 / CSE234 docs reference paths from the original assignment layout
(e.g. `cs336_basics/`, `/data/classifiers`, `cse234-w25-PA/pa2`). This
table translates each reference into where the equivalent lives here.

## Module paths

| In docs / code | In this repo |
|---|---|
| `cs336_basics/` (student module, Assignment 1) | `tt/nn/`, `tt/models/`, `tt/optim/`, `tt/utils/`, `tt/data/` |
| `cs336_systems/` (student module, Assignment 2) | `tt/kernels/`, `tt/parallel/` |
| `cs336_data/` (student module, Assignment 4) | `tt/data/pipeline/` |
| `cs336_alignment/` (student module, Assignment 5) | `tt/rl/` |
| `cs336_data/assets/` (downloaded classifier bins) | `assets/cs336_data/` (created on demand by `scripts/fetch_cs336_data_assets.sh`) |
| `cs336_alignment/prompts/` (prompt templates) | Not carried over; copy from Assignment 5 handout if needed |
| `cse234-w25-PA/pa3/part1/moe.py` (PA3 MoE) | `tt/moe/layers.py` + `tt/moe/models.py` |

## Cluster-specific absolute paths

The CS336 assignment handouts were written assuming the Stanford cluster.
Outside that environment, override via env var or edit the config.

| Original absolute path | Override mechanism |
|---|---|
| `/data/a5-alignment/models/Qwen2.5-Math-1.5B` (alignment model) | `ALIGNMENT_MODEL_ID` env var. Defaults still to the cluster path so tests fail loud when unset. |
| `/data/classifiers/*.bin` (data classifier binaries) | `SOURCE_DIR` env var for `scripts/fetch_cs336_data_assets.sh`. |
| `/data/paloma/tokenized_paloma_c4_100_domains_validation.bin` | Referenced only in `docs/cs336/assignment4_CHANGELOG.md`; runtime paths use Assignment 4's own config. |
| `/home/shared/Meta-Llama-3-70B-Instruct` | `--model-name-or-path` CLI arg of `scripts/evaluate_safety.py`. |
| `/home/shared/Meta-Llama-3.3-70B-Instruct` | `model_name:` key in `scripts/alpaca_eval_vllm_llama3_3_70b_fn/configs.yaml`. |
| `/home/sgugger/tmp/llama/llama-7b/` (in `pa3_configs/llama_7b_config.json`) | HF metadata only; harmless. |

## Fixtures carried over

| Original | Here |
|---|---|
| `assignment1-basics/tests/fixtures/*` (GPT-2 vocab, corpora, ts_tests/model.pt) | `tests/basics/fixtures/` |
| `assignment1-basics/tests/_snapshots/*` (.npz / .pkl) | `tests/basics/_snapshots/` |
| `assignment2-systems/tests/fixtures/*` | `tests/systems/fixtures/` |
| `assignment4-data/tests/fixtures/*` (html samples, dedup inputs) | `tests/data/fixtures/` |
| `assignment5-alignment/tests/fixtures/*` (tiny-gpt2, Llama-3 tokenizer, sft sample) | `tests/alignment/fixtures/` |
| `assignment5-alignment/tests/_snapshots/*` | `tests/alignment/_snapshots/` |

## Tests carried over (import contract)

Every test imports via `from .adapters import …` and `from .common import …`.
`adapters.py` in each `tests/<name>/` is a thin shim routing into `tt/*`.
You don't edit test files; implementing `tt/*` classes makes tests pass.
