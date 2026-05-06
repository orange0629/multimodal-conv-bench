#!/usr/bin/env python3
"""
Fix conversation.jsonl files in this directory to match the pipeline's
evaluate.py / analysis.ipynb format.

What this fixes:
  turns[].user        → turns[].role
  turns[].turn_id     string, 0-indexed → int, 1-indexed
  turns[].image_id    filename → turns[].image_path (absolute) + image_description: null
  missing scenario_title  (derived from first user question)
  ground_truth        missing reasoning_chain / key_difficulty
  missing _meta

Runs in-place; skips files that are already converted (detected by role field).

Usage:
  python fix_format.py           # fix all data_* subdirs
  python fix_format.py --dry-run # preview without writing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TAXONOMY = "interactive_visual_dialogue"
BASE = Path(__file__).resolve().parent


def _derive_title(turns: list[dict]) -> str:
    """Use the first user turn text as the title (≤80 chars)."""
    for t in turns:
        if t.get("user") == "user" or t.get("role") == "user":
            text = t.get("text", "").split("\n")[0].strip()
            return text[:80]
    return ""


def is_already_fixed(record: dict) -> bool:
    turns = record.get("turns", [])
    return bool(turns) and "role" in turns[0]


def fix_record(record: dict, conv_dir: Path) -> dict:
    new_turns = []
    for t in record.get("turns", []):
        role = "user" if t.get("user") == "user" else "assistant"
        new_turn: dict = {
            "turn_id": int(t["turn_id"]) + 1,
            "role":    role,
            "text":    t.get("text", ""),
        }

        img_id = t.get("image_id")
        if role == "user":
            new_turn["image_description"] = None
            if img_id:
                img_path = conv_dir / img_id
                new_turn["image_path"] = str(img_path.resolve()) if img_path.exists() else None
            else:
                new_turn["image_path"] = None
        else:
            new_turn["image_path"] = None

        new_turns.append(new_turn)

    orig_gt = record.get("ground_truth", {})
    scenario_desc = record.get("scenario_description", "")

    return {
        "scenario_title":       _derive_title(record.get("turns", [])),
        "scenario_description": scenario_desc,
        "turns":                new_turns,
        "ground_truth": {
            "question_type":   orig_gt.get("question_type", "multiple_choice"),
            "answer":          orig_gt.get("answer", ""),
            "reasoning_chain": orig_gt.get("reasoning_chain", ""),
            "key_difficulty":  orig_gt.get("key_difficulty", ""),
        },
        "_meta": {
            "taxonomy":     TAXONOMY,
            "scenario":     scenario_desc,
            "mode":         "multimodal",
            "model":        "",
            "backend":      "",
            "generated_at": "",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Print diffs without writing")
    args = ap.parse_args()

    conv_dirs = sorted(d for d in BASE.iterdir() if d.is_dir())
    fixed = skipped = 0

    for conv_dir in conv_dirs:
        jf = conv_dir / "conversation.jsonl"
        if not jf.exists():
            continue

        record = json.loads(jf.read_text(encoding="utf-8"))

        if is_already_fixed(record):
            skipped += 1
            continue

        new_record = fix_record(record, conv_dir)

        if args.dry_run:
            print(f"\n=== {conv_dir.name} ===")
            print(json.dumps(new_record, indent=2, ensure_ascii=False))
        else:
            jf.write_text(json.dumps(new_record, ensure_ascii=False) + "\n", encoding="utf-8")

        fixed += 1

    if args.dry_run:
        print(f"\n[dry-run] would fix {fixed}, skip {skipped}")
    else:
        print(f"Fixed {fixed} files, skipped {skipped} already-converted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
