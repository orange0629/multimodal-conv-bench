#!/usr/bin/env python3
"""
Benchmark Evaluation
====================
Evaluates LLMs on the generated multi-turn visual reasoning benchmark.

Per conversation:
  - Drops the last assistant turn (prevents info leakage before the final question)
  - Shows the model the full conversation + real images (no image_description text)
  - Expects answer in \\boxed{} format for easy parsing
  - Stores the full raw generation alongside the parsed answer

Backends
--------
  --backend vllm    local vLLM (default)
  --backend openai  OpenAI-compatible API

Usage
-----
  # Text-only benchmark (no real images)
  python pipeline/evaluate.py --input-dir output/ --model qwen --tp 2

  # With real images (after running gen_images.py)
  python pipeline/evaluate.py --input-dir output_images/ --model gemma --tp 2

  # Single taxonomy, first 20 items, verbose
  python pipeline/evaluate.py --input-dir output_images/ --taxonomy ist --limit 20 -v
"""

import argparse
import base64
import io
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

# ── Reuse model infrastructure from synthesize.py ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from synthesize import (
    MODEL_SHORTHANDS, _strip_thinking,
    resolve_model, get_model_config, load_vllm,
)

DEFAULT_EVAL_DIR = Path(__file__).parent.parent / "eval_results"

EVAL_SYSTEM = (
    "Please reason step by step, and put your final answer within \\boxed{}. E.g. \\boxed{A}, \\boxed{yes}, \\boxed{3}."
)

BOXED_INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}. For multiple-choice, respond with just the letter, e.g. \\boxed{A}."


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _img_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def parse_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...} in text, lowercased."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip().lower() if matches else None


_MC_LETTERS = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}

def _classify_failure(
    correct: bool,
    prediction: str | None,
    ground_truth: str,
    question_type: str,
) -> str:
    """Classify each result into one of four mutually-exclusive buckets."""
    if correct:
        return "correct"
    if prediction is None:
        return "format_failure"   # no \boxed{} found at all
    if question_type == "multiple_choice" and ground_truth in _MC_LETTERS and prediction not in _MC_LETTERS:
        return "option_text_not_letter"  # gave object name / full phrase instead of A/B/C/D
    return "wrong_answer"


# ─── Conversation builder ──────────────────────────────────────────────────────

def build_eval_messages(record: dict, text_only: bool = False) -> tuple[list[dict], bool]:
    """
    Build the [system, ...turns] message list for evaluation.

    - Drops the last assistant turn to prevent info leakage before the final question.
    - text_only=True: strips all images (text-only baseline mode).
    - text_only=False: loads real images from image_path when available.
    - Appends \\boxed{} instruction to the final user question.

    Returns (messages, has_images).
    """
    turns = list(record.get("turns", []))

    # Find and drop the last assistant turn
    assistant_idxs = [i for i, t in enumerate(turns) if t["role"] == "assistant"]
    if assistant_idxs:
        turns.pop(assistant_idxs[-1])

    messages: list[dict] = [{"role": "system", "content": EVAL_SYSTEM}]
    has_images = False

    for idx, turn in enumerate(turns):
        role = turn["role"]
        text = turn.get("text", "")
        image_path = turn.get("image_path")
        is_last = (idx == len(turns) - 1)

        if is_last and role == "user":
            text += BOXED_INSTRUCTION

        if not text_only and image_path and Path(image_path).exists():
            has_images = True
            img = Image.open(image_path)
            content = [
                {"type": "image_url",
                 "image_url": {"url": _img_to_data_url(img)}},
                {"type": "text", "text": text},
            ]
        else:
            content = text

        messages.append({"role": role, "content": content})

    return messages, has_images


# ─── Backend calls ─────────────────────────────────────────────────────────────

def _vllm_batch(llm, tok, model_id: str, all_messages: list[list[dict]],
                has_images_flags: list[bool], temperature: float) -> list[str]:
    """Submit all prompts in one vLLM call, split by text-only vs multimodal."""
    from vllm import SamplingParams
    cfg = get_model_config(model_id)
    sp  = SamplingParams(**{**cfg["sampling"], "temperature": temperature, "max_tokens": 8192})

    # Partition by modality, preserving original indices
    text_idxs  = [i for i, h in enumerate(has_images_flags) if not h]
    image_idxs = [i for i, h in enumerate(has_images_flags) if h]

    raws = [""] * len(all_messages)

    if text_idxs:
        prompts = [
            tok.apply_chat_template(all_messages[i], tokenize=False,
                                    add_generation_prompt=True,
                                    **cfg["chat_template_kwargs"])
            for i in text_idxs
        ]
        print(f"[vllm] text batch: {len(prompts)} prompts ...", file=sys.stderr)
        outputs = llm.generate(prompts, sp)
        for i, out in zip(text_idxs, outputs):
            raws[i] = _strip_thinking(out.outputs[0].text.strip())

    if image_idxs:
        messages_batch = [all_messages[i] for i in image_idxs]
        print(f"[vllm] multimodal batch: {len(messages_batch)} prompts ...", file=sys.stderr)
        outputs = llm.chat(messages=messages_batch, sampling_params=sp)
        for i, out in zip(image_idxs, outputs):
            raws[i] = _strip_thinking(out.outputs[0].text.strip())

    return raws


def _call_openai(model_id: str, messages: list[dict], temperature: float) -> str:
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model_id, messages=messages, temperature=temperature, max_tokens=8192,
    )
    return resp.choices[0].message.content.strip()


# ─── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(
    backend: str,
    model_id: str,
    records: list[dict],
    llm=None,
    tok=None,
    temperature: float = 0.0,
    verbose: bool = False,
    text_only: bool = False,
) -> list[dict]:
    n = len(records)
    if text_only:
        print("[eval] text-only baseline mode — images stripped", file=sys.stderr)

    # Build all messages upfront
    all_messages   = []
    has_images_flags = []
    for record in records:
        msgs, has_img = build_eval_messages(record, text_only=text_only)
        all_messages.append(msgs)
        has_images_flags.append(has_img)

    # Generate — one batch for vllm, sequential for openai
    if backend == "vllm":
        raws = _vllm_batch(llm, tok, model_id, all_messages, has_images_flags, temperature)
    else:
        raws = []
        for idx, msgs in enumerate(all_messages):
            try:
                raws.append(_call_openai(model_id, msgs, temperature))
            except Exception as e:
                print(f"  [error] item {idx}: {e}", file=sys.stderr)
                raws.append("")

    # Score
    results = []
    for idx, (record, raw) in enumerate(zip(records, raws)):
        gt           = record.get("ground_truth", {})
        meta         = record.get("_meta", {})
        prediction   = parse_boxed(raw)
        ground_truth = gt.get("answer", "").strip().lower()
        correct      = (prediction is not None and prediction == ground_truth)

        turns = record.get("turns", [])
        question = next((t.get("text", "") for t in reversed(turns) if t["role"] == "user"), "")

        question_type = gt.get("question_type", "")
        failure_mode = _classify_failure(correct, prediction, ground_truth, question_type)

        result = {
            "idx":                  idx,
            "taxonomy":             meta.get("taxonomy", ""),
            "scenario":             meta.get("scenario", ""),
            "scenario_title":       record.get("scenario_title", ""),
            "scenario_description": record.get("scenario_description", ""),
            "question":             question,
            "question_type":        question_type,
            "ground_truth":         ground_truth,
            "reasoning_chain":      gt.get("reasoning_chain", ""),
            "key_difficulty":       gt.get("key_difficulty", ""),
            "prediction":           prediction,
            "correct":              correct,
            "failure_mode":         failure_mode,
            "has_images":           has_images_flags[idx],
            "eval_model":           model_id,
            "gen_model":            meta.get("model", ""),
            "generated_at":         meta.get("generated_at", ""),
            "raw_output":           raw,
        }
        results.append(result)

        if verbose or (idx + 1) % 10 == 0:
            mark = "✓" if correct else "✗"
            print(
                f"  [{idx+1:3d}/{n}] {mark}  gt={ground_truth!r:6s}  "
                f"pred={str(prediction)!r:6s}  "
                f"({result['taxonomy']}, {result['question_type']})",
                file=sys.stderr,
            )

    return results


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(results: list[dict], model_id: str) -> None:
    total   = len(results)
    correct = sum(r["correct"] for r in results)

    print(f"\n{'='*62}")
    print(f"Model : {model_id}")
    print(f"Total : {correct}/{total} = {correct/total*100:.1f}%")
    print(f"{'='*62}")

    by_tax = defaultdict(list)
    for r in results:
        by_tax[r["taxonomy"]].append(r)
    print("\nBy taxonomy:")
    for tax, rs in sorted(by_tax.items()):
        nc = sum(r["correct"] for r in rs)
        print(f"  {tax:45s} {nc:3d}/{len(rs):3d}  {nc/len(rs)*100:5.1f}%")

    by_qt = defaultdict(list)
    for r in results:
        by_qt[r["question_type"]].append(r)
    print("\nBy question type:")
    for qt, rs in sorted(by_qt.items()):
        nc = sum(r["correct"] for r in rs)
        print(f"  {qt:20s} {nc:3d}/{len(rs):3d}  {nc/len(rs)*100:5.1f}%")

    img_results = [r for r in results if r["has_images"]]
    if img_results:
        nc = sum(r["correct"] for r in img_results)
        print(f"\nWith real images: {nc}/{len(img_results)} = {nc/len(img_results)*100:.1f}%")

    from collections import Counter
    modes = Counter(r["failure_mode"] for r in results)
    print("\nFailure mode breakdown:")
    for mode in ["correct", "wrong_answer", "option_text_not_letter", "format_failure"]:
        n = modes.get(mode, 0)
        print(f"  {mode:30s} {n:4d} / {total:4d}  ({n/total*100:5.1f}%)")


# ─── Record loading ────────────────────────────────────────────────────────────

def load_records(input_dir: Path, taxonomy: list[str] | None, limit: int | None) -> list[dict]:
    if input_dir.suffix == ".jsonl":
        paths = [input_dir]
    else:
        # Prefer conversation.jsonl files (gen_images output, has image_path fields)
        conv_paths = sorted(input_dir.rglob("conversation.jsonl"))
        flat_paths = sorted(p for p in input_dir.rglob("*.jsonl")
                            if p.name != "conversation.jsonl")
        paths = conv_paths if conv_paths else flat_paths

    if taxonomy:
        paths = [p for p in paths if any(t in str(p) for t in taxonomy)]

    records = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line or '"_error"' in line:
                    continue
                records.append(json.loads(line))

    if limit:
        records = records[:limit]
    print(f"[eval] loaded {len(records)} conversations from {len(paths)} file(s)", file=sys.stderr)
    return records


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate models on the multi-turn visual reasoning benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input-dir", required=True,
                   help="Directory with JSONL files (output/ or output_images/) or a single JSONL")
    p.add_argument("--taxonomy", nargs="+", default=None,
                   help="Evaluate only these taxonomies (space-separated, e.g. --taxonomy belief_revision incremental_state_tracking)")
    p.add_argument("--limit", type=int, default=None, help="Max conversations to evaluate")

    p.add_argument("--backend", choices=["vllm", "openai"], default="vllm")
    p.add_argument("--model", "-m", default="qwen",
                   help=f"Model shorthand or full HF ID. Shorthands: {MODEL_SHORTHANDS}")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (default 0 = greedy)")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel size for vLLM")
    p.add_argument("--memory-efficient", action="store_true",
                   help="Enable memory-saving vLLM options (see synthesize.py)")

    p.add_argument("--output-dir", default=str(DEFAULT_EVAL_DIR),
                   help=f"Directory to save per-item results. Default: {DEFAULT_EVAL_DIR}")
    p.add_argument("--text-only", action="store_true",
                   help="Baseline mode: strip all images, evaluate on text only")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print every item result")
    return p.parse_args()


def main():
    args      = parse_args()
    model_id  = resolve_model(args.model)
    input_dir = Path(args.input_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] backend={args.backend}  model={model_id}  tp={args.tp}"
          f"  text_only={args.text_only}", file=sys.stderr)

    llm, tok = (load_vllm(model_id, args.tp, args.memory_efficient)
                if args.backend == "vllm" else (None, None))

    records = load_records(input_dir, args.taxonomy, args.limit)
    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    results = evaluate(
        args.backend, model_id, records, llm, tok,
        temperature=args.temperature, verbose=args.verbose,
        text_only=args.text_only,
    )

    # Save full results (including raw_output for every item)
    ts         = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_slug = model_id.replace("/", "_")
    tax_slug   = f"_{args.taxonomy}" if args.taxonomy else ""
    mode_slug  = "_textonly" if args.text_only else ""
    out_file   = out_dir / f"{model_slug}{tax_slug}{mode_slug}_{ts}.jsonl"

    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print_summary(results, model_id)
    print(f"\nFull results (with raw generations) → {out_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
