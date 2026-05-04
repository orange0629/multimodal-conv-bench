"""
Evaluate a VLM on DialogVis conversations (static evaluation).

For each conversation we build a multimodal chat: each user/assistant turn is
delivered in order with its image inlined. A final user message poses the
multiple-choice question. The model must reply with a single letter (A-D).

Backends:
  - openai-compatible chat.completions with image_url content (works with vLLM,
    OpenAI, Together, etc.)

Output:
  outputs/eval_results/<model-tag>_<UTC-timestamp>.jsonl
  one line per conversation:
    {
      "id": ..., "taxonomy": ..., "difficulty": ...,
      "predicted_letter": "A", "correct_letter": "C",
      "is_correct": false,
      "raw_response": "...full model output...",
      "elapsed_ms": 4123
    }

  Plus a final summary line with overall + breakdown stats prefixed by
  "summary": true, suitable for `jq 'select(.summary)'`.

Run:
  python scripts/evaluate.py \
      --model Qwen/Qwen3-VL-7B-Instruct \
      --api-base http://blender13.cs.illinois.edu:8005/v1 \
      --taxonomy belief_revision incremental_state_tracking \
      --conv-suffix v2 \
      --concurrency 4
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval")

ROOT = Path(__file__).resolve().parent.parent
CONV_DIR = ROOT / "outputs" / "conversations"
EVAL_DIR = ROOT / "outputs" / "eval_results"


SYSTEM_PROMPT = (
    "You are a careful multimodal reasoner. You will be shown a multi-turn "
    "dialogue with images that arrive across turns. After the dialogue you will "
    "be asked a multiple-choice question about it. Reason step-by-step internally "
    "but reply with ONLY a single capital letter (A, B, C, or D) — nothing else."
)


@dataclass
class EvalResult:
    id: str
    taxonomy: str
    difficulty: str
    predicted_letter: str | None
    correct_letter: str | None
    is_correct: bool
    raw_response: str
    elapsed_ms: int
    error: str | None = None


# ---------------------------------------------------------------------------
# Multimodal prompt assembly
# ---------------------------------------------------------------------------

def encode_image_data_url(path: Path) -> str:
    """PNG -> data URL for OpenAI/vLLM chat content."""
    b = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


def build_messages(conv: dict, text_only: bool = False) -> tuple[list[dict], str | None]:
    """Build chat messages. Returns (messages, error_or_None).

    If text_only, images are dropped. Their description is inlined as
    '[Image: <description>]' so a text LLM still has *some* signal.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in conv["turns"]:
        role = "user" if t["speaker"] == "user" else "assistant"
        text = t.get("text", "")
        img = t.get("image")
        if text_only:
            parts = []
            if img:
                parts.append(f"[Image: {img.get('description','')}]")
            if text:
                parts.append(text)
            messages.append({"role": role, "content": "\n".join(parts) if parts else ""})
            continue
        content: list[dict] = []
        if img and img.get("source_path"):
            p = ROOT / img["source_path"]
            if not p.exists():
                return [], f"missing image file: {img['source_path']}"
            content.append({"type": "image_url", "image_url": {"url": encode_image_data_url(p)}})
        if text:
            content.append({"type": "text", "text": text})
        if not content:
            content = [{"type": "text", "text": ""}]
        messages.append({"role": role, "content": content})

    # Final question
    options_str = "\n".join(f"{o['label']}. {o['text']}" for o in conv["mcq_options"])
    q = (
        f"{conv['final_question']}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Reply with ONLY the single letter of the correct answer."
    )
    if text_only:
        messages.append({"role": "user", "content": q})
    else:
        messages.append({"role": "user", "content": [{"type": "text", "text": q}]})
    return messages, None


def correct_letter_of(conv: dict) -> str | None:
    for o in conv["mcq_options"]:
        if o.get("is_correct"):
            return o["label"]
    return None


_LETTER_RE = re.compile(r"\b([A-D])\b")


def parse_letter(raw: str) -> str | None:
    raw = (raw or "").strip().upper()
    if not raw:
        return None
    if raw[0] in "ABCD":
        return raw[0]
    m = _LETTER_RE.search(raw)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class OpenAIChatBackend:
    def __init__(self, model: str, api_base: str, api_key: str, max_tokens: int = 16, timeout: float = 120.0):
        from openai import OpenAI
        self.client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens

    def call(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def evaluate_one(backend: OpenAIChatBackend, conv: dict, max_retries: int = 2,
                 text_only: bool = False) -> EvalResult:
    correct_letter = correct_letter_of(conv)
    msgs, err = build_messages(conv, text_only=text_only)
    if err:
        return EvalResult(
            id=conv["id"], taxonomy=conv.get("taxonomy", "?"),
            difficulty=conv.get("difficulty", "?"),
            predicted_letter=None, correct_letter=correct_letter,
            is_correct=False, raw_response="", elapsed_ms=0, error=err,
        )
    raw = ""
    last_exc = None
    t0 = time.time()
    for attempt in range(1, max_retries + 1):
        try:
            raw = backend.call(msgs)
            last_exc = None
            break
        except Exception as e:
            last_exc = e
            log.warning("Attempt %d/%d failed for %s: %s", attempt, max_retries, conv["id"][:8], e)
            time.sleep(2 ** attempt)
    elapsed = int((time.time() - t0) * 1000)
    if last_exc is not None:
        return EvalResult(
            id=conv["id"], taxonomy=conv.get("taxonomy", "?"),
            difficulty=conv.get("difficulty", "?"),
            predicted_letter=None, correct_letter=correct_letter,
            is_correct=False, raw_response="", elapsed_ms=elapsed, error=str(last_exc)[:200],
        )

    pred = parse_letter(raw)
    return EvalResult(
        id=conv["id"], taxonomy=conv.get("taxonomy", "?"),
        difficulty=conv.get("difficulty", "?"),
        predicted_letter=pred, correct_letter=correct_letter,
        is_correct=(pred is not None and pred == correct_letter),
        raw_response=raw[:1000], elapsed_ms=elapsed,
    )


def load_convs(taxonomies: list[str], suffix: str | None) -> list[dict]:
    convs = []
    for tax in taxonomies:
        name = f"{tax}.{suffix}.jsonl" if suffix else f"{tax}.jsonl"
        path = CONV_DIR / name
        if not path.exists():
            log.warning("Missing %s — skipping", path)
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    convs.append(json.loads(line))
    return convs


def model_tag(model: str) -> str:
    """Filesystem-safe tag for a model id (used in output filename)."""
    return model.replace("/", "_").replace(":", "_")


def summarize(results: list[EvalResult]) -> dict:
    total = len(results)
    if total == 0:
        return {"summary": True, "total": 0}
    correct = sum(1 for r in results if r.is_correct)
    invalid = sum(1 for r in results if r.predicted_letter is None and not r.error)
    errors = sum(1 for r in results if r.error)

    by_tax: dict[str, list[EvalResult]] = {}
    by_diff: dict[str, list[EvalResult]] = {}
    for r in results:
        by_tax.setdefault(r.taxonomy, []).append(r)
        by_diff.setdefault(r.difficulty, []).append(r)
    def acc(rs):
        return round(sum(1 for r in rs if r.is_correct) / max(1, len(rs)), 4)

    return {
        "summary": True,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "invalid_format": invalid,
        "errors": errors,
        "avg_elapsed_ms": int(sum(r.elapsed_ms for r in results) / total),
        "by_taxonomy": {k: {"n": len(v), "accuracy": acc(v)} for k, v in by_tax.items()},
        "by_difficulty": {k: {"n": len(v), "accuracy": acc(v)} for k, v in by_diff.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model id, e.g. Qwen/Qwen3-VL-7B-Instruct")
    ap.add_argument("--api-base", required=True, help="OpenAI-compatible base URL, e.g. http://host:8005/v1")
    ap.add_argument("--api-key", default=os.getenv("EVAL_API_KEY") or os.getenv("DIALOGVIS_API_KEY") or "EMPTY",
                    help="API key (default: EMPTY for vLLM)")
    ap.add_argument("--taxonomy", "-t", action="append", required=True,
                    help="Repeatable. Taxonomy name (e.g. belief_revision)")
    ap.add_argument("--conv-suffix", default=None,
                    help="Suffix before .jsonl (e.g. 'v2' to load belief_revision.v2.jsonl)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=16, help="Cap response tokens (only need a letter)")
    ap.add_argument("--limit", type=int, default=None, help="Eval only first N convs (smoke test)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle conv order before --limit")
    ap.add_argument("--text-only", action="store_true",
                    help="Drop images and inline their descriptions as text. Use for text-only LLMs "
                         "(e.g. Qwen3.5, Gemma-3-it). Provides a useful no-vision baseline.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=EVAL_DIR)
    args = ap.parse_args()

    convs = load_convs(args.taxonomy, args.conv_suffix)
    if args.shuffle:
        random.Random(args.seed).shuffle(convs)
    if args.limit:
        convs = convs[: args.limit]
    if not convs:
        log.error("No conversations loaded. Check --taxonomy / --conv-suffix.")
        return 1
    log.info("Loaded %d conversations", len(convs))

    backend = OpenAIChatBackend(args.model, args.api_base, args.api_key, max_tokens=args.max_tokens)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"{model_tag(args.model)}_{ts}.jsonl"
    log.info("Writing results to %s", out_path)

    results: list[EvalResult] = []
    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        # Header line with run metadata
        fout.write(json.dumps({
            "header": True,
            "model": args.model,
            "api_base": args.api_base,
            "taxonomies": args.taxonomy,
            "conv_suffix": args.conv_suffix,
            "text_only": args.text_only,
            "n_conversations": len(convs),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }) + "\n")
        fout.flush()

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(evaluate_one, backend, c, 2, args.text_only): c for c in convs}
            for i, fut in enumerate(as_completed(futures), 1):
                r: EvalResult = fut.result()
                results.append(r)
                fout.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
                fout.flush()
                written += 1
                if i % 10 == 0:
                    correct = sum(1 for x in results if x.is_correct)
                    log.info("Progress: %d/%d (acc=%.3f)", i, len(convs), correct / len(results))

        # Final summary
        summary = summarize(results)
        fout.write(json.dumps(summary, ensure_ascii=False) + "\n")

    log.info("Done. Wrote %d results to %s", written, out_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
