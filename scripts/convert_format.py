"""
Convert DialogVis conversation JSONL into the team's shared schema.

Input:  outputs/conversations/<taxonomy>.jsonl  (or .v2.jsonl)
Output: outputs/conversations_export/<taxonomy>[.v2].export.jsonl

Per-line schema (matches the team example):

{
  "scenario_title": "...",
  "scenario_description": "...",
  "turns": [
    {"turn_id": 1, "role": "user", "text": "...", "image_description": "..."},
    {"turn_id": 2, "role": "assistant", "text": "..."},
    ...
    # final turn: user question with the MCQ options inlined
    {"turn_id": N, "role": "user", "text": "Q? A) ... B) ... ...", "image_description": null}
  ],
  "ground_truth": {
    "question_type": "multiple_choice",
    "answer": "B",
    "reasoning_chain": "...",
    "key_difficulty": "..."
  },
  "_meta": {
    "taxonomy": "belief_revision",
    "scenario": "<full scenario text>",
    "mode": "multimodal",
    "model": "<source model>",
    "backend": "vllm",
    "generated_at": "<ISO ts>"
  }
}

Notes:
  - scenario_title is derived from the scenario string (text before ':' if any).
  - scenario_description is the full scenario string.
  - assistant turns carry no image_description (per the team example, only user
    turns include it).
  - The final user turn embeds the MCQ options in `text` so a text-only LLM
    can answer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONV_DIR = ROOT / "outputs" / "conversations"
OUT_DIR = ROOT / "outputs" / "conversations_export"


def derive_title(scenario: str) -> str:
    """Cheap title from the existing scenario string."""
    if not scenario:
        return ""
    # If "Domain: detail" split, take detail; else take whole.
    if ":" in scenario:
        _, detail = scenario.split(":", 1)
        title = detail.strip()
    else:
        title = scenario.strip()
    # Drop trailing period and limit to ~80 chars.
    title = title.rstrip(".").strip()
    return title[:80]


def correct_letter(conv: dict) -> str | None:
    for o in conv.get("mcq_options") or []:
        if o.get("is_correct"):
            return o["label"]
    return None


def options_block(conv: dict) -> str:
    return "\n".join(f"{o['label']}) {o['text']}" for o in conv.get("mcq_options") or [])


def convert_one(conv: dict, default_mode: str, image_dir: str | None) -> dict:
    new_turns = []
    conv_id = conv.get("id", "")
    for t in conv.get("turns", []):
        role = "user" if t.get("speaker") == "user" else "assistant"
        entry = {"turn_id": t["turn_id"], "role": role, "text": t.get("text", "")}
        img = t.get("image")
        # Only user turns carry image_description; assistants don't (per team example).
        if role == "user":
            entry["image_description"] = (img.get("description") if img else None)
            if img:
                img_id = img.get("id")
                entry["image_id"] = img_id
                # Path preference (highest first):
                #   1) <image_dir>/<conv_id>/<img_id>.png if it exists on disk
                #      (this is where Phase v2 puts the file)
                #   2) the source_path recorded in the JSONL
                #   3) synthesized canonical <image_dir>/<conv_id>/<img_id>.png
                #      (will be valid once Phase 4 generates it)
                synth = f"{image_dir}/{conv_id}/{img_id}.png" if image_dir else None
                if synth and (ROOT / synth).exists():
                    entry["image_path"] = synth
                elif img.get("source_path"):
                    entry["image_path"] = img["source_path"]
                elif synth:
                    entry["image_path"] = synth
        new_turns.append(entry)

    # Append the final question turn as user with options inline.
    next_id = (new_turns[-1]["turn_id"] if new_turns else 0) + 1
    final_text = f"{conv.get('final_question', '').strip()}\n\n{options_block(conv)}".strip()
    new_turns.append({
        "turn_id": next_id,
        "role": "user",
        "text": final_text,
        "image_description": None,
    })

    meta = conv.get("metadata") or {}
    return {
        "id": conv.get("id"),
        "scenario_title": derive_title(conv.get("scenario", "")),
        "scenario_description": conv.get("scenario", ""),
        "turns": new_turns,
        "ground_truth": {
            "question_type": "multiple_choice",
            "answer": correct_letter(conv) or "",
            "reasoning_chain": conv.get("reasoning_chain", ""),
            "key_difficulty": conv.get("why_sequential", ""),
        },
        "_meta": {
            "id": conv.get("id"),
            "taxonomy": conv.get("taxonomy"),
            "scenario": conv.get("scenario", ""),
            "difficulty": conv.get("difficulty"),
            "mode": default_mode,
            "model": meta.get("model", ""),
            "backend": meta.get("provider", ""),
            "generated_at": meta.get("generated_at", ""),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", "-t", action="append",
                    help="Repeatable. If omitted, processes all *.jsonl in conversations dir.")
    ap.add_argument("--suffix", default=None,
                    help="Source suffix before .jsonl (e.g. 'v2' to load belief_revision.v2.jsonl)")
    ap.add_argument("--mode", default="multimodal", choices=["multimodal", "text-only"],
                    help="Value to write into _meta.mode")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--image-dir", default="outputs/images_v2",
                    help="Path used to synthesize image_path when the source jsonl has no source_path "
                         "yet (e.g. images still generating). Default: outputs/images_v2.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Decide which input files to process
    if args.taxonomy:
        files = []
        for tax in args.taxonomy:
            fname = f"{tax}.{args.suffix}.jsonl" if args.suffix else f"{tax}.jsonl"
            files.append(CONV_DIR / fname)
    else:
        pattern = f"*.{args.suffix}.jsonl" if args.suffix else "*.jsonl"
        files = sorted(CONV_DIR.glob(pattern))
        # If no suffix, exclude .v2/.kept variants to keep it sane
        if not args.suffix:
            files = [f for f in files
                     if all(s not in f.name for s in (".v2.", ".kept."))]

    total = 0
    for f in files:
        if not f.exists():
            print(f"[skip] missing: {f}")
            continue
        out_path = args.out_dir / (f.stem + ".export.jsonl")
        n = 0
        with f.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                conv = json.loads(line)
                new = convert_one(conv, args.mode, args.image_dir)
                dst.write(json.dumps(new, ensure_ascii=False) + "\n")
                n += 1
        total += n
        print(f"  {f.name} -> {out_path.name}  ({n} convs)")
    print(f"Total: {total} conversations exported to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
