# MultiModal-Conv-Bench

> **A benchmark for multi-turn, multi-modal reasoning in vision-language models.**
>
> Each item is a realistic conversation where images arrive across turns — testing capabilities that single-turn benchmarks simply cannot measure.

---

## Why This Exists

State-of-the-art VLMs can handle complex single-turn visual queries. But real users don't send all their images at once. They debug hardware over several photos, iterate on a design, or walk through a medical case turn by turn.

**MultiModal-Conv-Bench** forces models to reason under *evolving, incomplete information* — maintaining a mental model of the visual world, revising beliefs when contradicted, and tracking entities and references across turns.

---

## Taxonomy

Six reasoning categories, each targeting a distinct capability gap:

| Alias | Full Name | What It Tests |
|-------|-----------|---------------|
| `ist`  | **Incremental State Tracking** | Detect and accumulate changes across a scene that evolves turn-by-turn |
| `br`   | **Belief Revision** | Update initial interpretations when later images contradict them |
| `ctet` | **Cross-Turn Entity Tracking** | Re-identify entities across images using natural-language references ("the one from earlier") |
| `tcr`  | **Temporal & Causal Reasoning** | Reconstruct event order and causality from images that may arrive out of sequence |
| `ivd`  | **Interactive Visual Dialogue** | Maintain coherent reasoning when the image stream is shaped by the dialogue itself |
| `sia`  | **Strategic Information Acquisition** | Recognise gaps in visual evidence and proactively request the right images |

---

## Pipeline Overview

```
  ┌─────────────────┐
  │   synthesize    │  ← LLM auto-generates scenarios then writes multi-turn
  │                 │    conversations with image descriptions per turn
  └────────┬────────┘
           │  pipeline/output/<taxonomy>/<timestamp>.jsonl
           ▼
  ┌─────────────────┐
  │   gen_images    │  ← Gemini turns each image_description into a real PNG,
  │                 │    maintaining visual consistency across turns
  └────────┬────────┘
           │  pipeline/output_images/<taxonomy>/<conv_id>/turn_N.png
           ▼
  ┌─────────────────┐
  │    evaluate     │  ← Run VLMs on the benchmark, score \boxed{} answers
  └────────┬────────┘
           │  eval_results/<model>_<timestamp>.jsonl
           ▼
  ┌─────────────────┐
  │  analysis.ipynb │  ← Aggregate scores, generate paper figures
  └─────────────────┘
           │
           ▼
  ┌─────────────────┐
  │   build_demo    │  ← Self-contained HTML viewer for conversations
  └─────────────────┘
```

---

## Installation

```bash
pip install -r requirements.txt
# vLLM (for local model serving)
pip install vllm
# Vertex AI / Gemini (for image generation)
pip install google-genai
```

---

## Step 1 — Synthesize Conversations

An LLM first auto-generates a bank of diverse scenarios for each taxonomy, then writes full multi-turn conversations. Each record contains the dialogue, per-turn image descriptions, and ground-truth Q&A.

### Quick start (single conversation)

```bash
python pipeline/synthesize.py \
    --taxonomy ist \
    --scenario "A cluttered desk being gradually tidied up across a workday" \
    --n 1 --backend vllm --model qwen
```

### All taxonomies from the config

```bash
python pipeline/synthesize.py \
    --config configs/scenarios.yaml --taxonomy all \
    --n 3 --backend vllm --model gemma --tp 2
```

### SLURM (Delta cluster)

```bash
# Defaults: Qwen, temporal_causal_reasoning, 1 sample/scenario
sbatch pipeline/run.sh

# Override via env vars
MODEL=gemma TAXONOMY=ist N=3 sbatch pipeline/run.sh
MODEL=qwen  TAXONOMY=all N=5 sbatch pipeline/run.sh
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | — | `gemma`, `qwen`, or any HuggingFace ID / local path |
| `--taxonomy` | — | `all` or any alias from the table above |
| `--n` | 1 | Conversations sampled per scenario |
| `--n-themes` | 20 | Auto-generated scenario themes per taxonomy |
| `--n-per-theme` | 5 | Scenarios generated per theme |
| `--tp` | 1 | Tensor-parallel size (= number of GPUs) |
| `--memory-efficient` | off | Unload model between taxonomy chunks |
| `--regen` | off | Force regeneration of the scenario cache |

**Model shorthands:**

| Shorthand | Full HuggingFace ID |
|-----------|---------------------|
| `gemma` | `google/gemma-4-31B-it` |
| `qwen` | `Qwen/Qwen3.5-27B` |

**Output:** `pipeline/output/<taxonomy>/<timestamp>.jsonl`

---

## Step 2 — Generate Images

Turns each `image_description` field into a real PNG via Gemini on Vertex AI. Images within a conversation are generated sequentially — each new image receives all prior images as context to maintain visual consistency.

```bash
# Single taxonomy output file
python pipeline/gen_images.py \
    --input-dir pipeline/output/temporal_causal_reasoning/20260503_072214.jsonl \
    --output-dir pipeline/output_images

# All taxonomies under a directory
python pipeline/gen_images.py --input-dir pipeline/output/ --taxonomy all

# Test run: first 10 conversations only
python pipeline/gen_images.py --input-dir pipeline/output/ --limit 10
```

### SLURM (CPU partition — no GPU needed)

```bash
INPUT=pipeline/output/temporal_causal_reasoning/20260503_072214.jsonl \
sbatch pipeline/gen_images.sh

# Cross-turn entity tracking variant
INPUT=pipeline/output/cross_turn_entity_tracking/ sbatch pipeline/gen_images_cross.sh
```

**Output:** `pipeline/output_images/<taxonomy>/<conv_id>/`
- `turn_1.png`, `turn_2.png`, … — generated images per turn
- `conversation.jsonl` — updated record with `image_path` fields added

> Requires Vertex AI credentials and access to `gemini-3.1-flash-image-preview`.

---

## Step 3 — Evaluate Models

Runs a VLM on the benchmark. The pipeline strips the final assistant turn to prevent leakage, presents the full conversation with real images, and expects answers in `\boxed{}` format.

```bash
# Text-only (no real images — uses image descriptions as text)
python pipeline/evaluate.py \
    --input-dir pipeline/output/ \
    --model qwen --tp 2

# With real images (after Step 2)
python pipeline/evaluate.py \
    --input-dir pipeline/output_images/ \
    --model gemma --tp 2

# Single taxonomy, first 20 items, verbose output
python pipeline/evaluate.py \
    --input-dir pipeline/output_images/ \
    --taxonomy ist --limit 20 --verbose

# OpenAI-compatible API backend
python pipeline/evaluate.py \
    --backend openai --model gpt-4o \
    --input-dir pipeline/output_images/
```

### SLURM (single model)

```bash
# Default: Qwen3.5-27B, all taxonomies, 2 GPUs
sbatch pipeline/eval.sh

# Override
MODEL=google/gemma-4-31B-it TAXONOMY=br TP=2 sbatch pipeline/eval.sh

# Text-only baseline
sbatch pipeline/eval_textonly_baseline.sh
```

### Evaluate all models in one run

```bash
bash pipeline/run_all_evals.sh

# With overrides
INPUT_DIR=pipeline/output_images \
TAXONOMY="belief_revision incremental_state_tracking" \
LIMIT=50 \
bash pipeline/run_all_evals.sh
```

Models evaluated out of the box:

| Model | Size | GPUs |
|-------|------|------|
| `Qwen/Qwen3.5-2B` | 2B | 1 |
| `Qwen/Qwen3.5-4B` | 4B | 1 |
| `Qwen/Qwen3.5-9B` | 9B | 1 |
| `Qwen/Qwen3-VL-2B-Instruct` | 2B | 1 |
| `Qwen/Qwen3-VL-4B-Instruct` | 4B | 1 |
| `Qwen/Qwen3-VL-8B-Instruct` | 8B | 1 |
| `google/gemma-4-E4B-it` | ~4B | 1 |
| `google/gemma-4-26B-A4B-it` | 26B | 2 |
| `google/gemma-4-31B-it` | 31B | 2 |
| `Qwen/Qwen3.5-27B` | 27B | 2 |

**Key flags:**

| Flag | Description |
|------|-------------|
| `--backend vllm` | Local vLLM serving (default) |
| `--backend openai` | OpenAI-compatible API |
| `--tp N` | Tensor-parallel size |
| `--memory-efficient` | Unload model between taxonomy chunks |
| `--taxonomy ALIAS` | Restrict to one taxonomy |
| `--limit N` | Cap conversations (useful for smoke tests) |

**Output:** `eval_results/<model>_<timestamp>.jsonl`

---

## Step 4 — Analyse Results

Open the analysis notebook to compute per-taxonomy accuracy, failure mode breakdowns, and reproduce paper figures.

```bash
jupyter notebook analysis.ipynb
```

Pre-generated figures (in repo root):
- `fig_overall_accuracy.pdf` — model accuracy by taxonomy
- `fig_item_difficulty.pdf` — item-level difficulty distribution
- `fig_taxonomy_sensitivity_box.pdf` — score variance across taxonomies
- `fig_ctet_pie.pdf` — failure mode breakdown for CTET

---

## Step 5 — Build the Demo

Generates a self-contained HTML file for browsing benchmark conversations in any browser — no server required.

```bash
# All conversations with images
python pipeline/build_demo.py \
    --input-dir pipeline/output_images \
    --output demo.html

# Single taxonomy, first 20 items
python pipeline/build_demo.py \
    --input-dir pipeline/output_images \
    --taxonomy cross_turn_entity_tracking \
    --limit 20 \
    --output demo.html
```

---

## Repository Layout

```
multimodal-conv-bench/
├── configs/
│   └── scenarios.yaml              # Scenario descriptions per taxonomy
├── pipeline/
│   ├── synthesize.py               # Step 1: generate conversations
│   ├── gen_images.py               # Step 2: generate per-turn images
│   ├── evaluate.py                 # Step 3: run & score models
│   ├── build_demo.py               # Step 5: HTML demo builder
│   ├── prompts.py                  # Prompt templates for all 6 taxonomies
│   ├── run.sh                      # SLURM: synthesis
│   ├── gen_images.sh               # SLURM: image generation
│   ├── gen_images_cross.sh         # SLURM: image generation (CTET)
│   ├── eval.sh                     # SLURM: single-model evaluation
│   ├── eval_textonly_baseline.sh   # SLURM: text-only baseline
│   ├── run_all_evals.sh            # Evaluate all models sequentially
│   └── output/                     # Synthesized conversations
│       └── <taxonomy>/<timestamp>.jsonl
├── pipeline/output_images/         # Conversations with real images
│   └── <taxonomy>/<conv_id>/
│       ├── turn_1.png
│       └── conversation.jsonl
├── eval_results/                   # Evaluation outputs
│   └── <model>_<timestamp>.jsonl
├── analysis.ipynb                  # Results analysis & figures
├── demo.html                       # Interactive conversation browser
└── requirements.txt
```

---

## Data Format

Each synthesized conversation record:

```json
{
  "scenario_title": "...",
  "scenario_description": "...",
  "taxonomy": "incremental_state_tracking",
  "turns": [
    { "turn_id": 1, "role": "user", "text": "...", "image_description": "..." },
    { "turn_id": 2, "role": "assistant", "text": "..." },
    { "turn_id": N, "role": "user", "text": "<final question — no image>" }
  ],
  "ground_truth": {
    "question_type": "multiple_choice | yes_no | count",
    "answer": "B",
    "reasoning_chain": "...",
    "key_difficulty": "..."
  }
}
```

Models must answer in `\boxed{}` format — e.g. `\boxed{B}`, `\boxed{yes}`, `\boxed{3}`.

---

## Citation

```bibtex
@misc{multimodal-conv-bench-2026,
  title   = {MultiModal-Conv-Bench: Towards Evaluating Conversational Visual Reasoning in Large Language Models},
  author  = {Lechen Zhang and Xuejun Zhang and Peixuan Han and Shujin Wu},
  year    = {2026},
}
```
