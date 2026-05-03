#!/usr/bin/env python3
"""
Image Generation Pipeline
==========================
Turns image_description fields in synthesized conversations into real images,
using Gemini's image generation model via Vertex AI.

Sequential coherence: when generating the image for turn N, all images
generated for turns 1..N-1 in the same conversation are passed as context,
so the model maintains visual consistency across the conversation.

Input:  output/<taxonomy>/*.jsonl   (from synthesize.py)
Output: output_images/<taxonomy>/<conv_id>/turn_<N>.png
        output_images/<taxonomy>/<conv_id>/conversation.jsonl  (updated record with image paths)

Usage
-----
  python pipeline/gen_images.py --input-dir output/incremental_state_tracking
  python pipeline/gen_images.py --input-dir output/ --taxonomy all
  python pipeline/gen_images.py --input-dir output/ --limit 10  # first 10 conversations
"""

import argparse
import json
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

VERTEX_PROJECT = "project-baba0887-0dd6-4603-91b"
GEMINI_MODEL   = "gemini-3.1-flash-image-preview"

# ─── Gemini client ────────────────────────────────────────────────────────────

def make_client() -> genai.Client:
    return genai.Client(vertexai=True, project=VERTEX_PROJECT)

# ─── Image generation ─────────────────────────────────────────────────────────

def generate_image(
    client: genai.Client,
    description: str,
    prior_images: list[Image.Image],
    retries: int = 3,
    retry_delay: float = 5.0,
) -> Image.Image | None:
    """
    Generate one image from a description, using prior conversation images as context.
    Prior images are passed in order so the model maintains visual consistency.
    """
    if prior_images:
        prompt = (
            "The following images are from an ongoing visual conversation, shown in order. "
            "Generate the NEXT image in this conversation that matches the description below. "
            "Maintain visual consistency (same scene, style, entities) with the previous images.\n\n"
            f"Description: {description}"
        )
        contents: list = list(prior_images) + [prompt]
    else:
        prompt = (
            "Generate a photorealistic image that matches this description exactly.\n\n"
            f"Description: {description}"
        )
        contents = [prompt]

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
            )
            for part in response.parts:
                if part.inline_data is not None:
                    return part.as_image()
            print(f"  [warn] no image in response (attempt {attempt})", file=sys.stderr)
        except Exception as e:
            print(f"  [retry {attempt}/{retries}] {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(retry_delay * attempt)

    return None

# ─── Per-conversation processing ──────────────────────────────────────────────

def process_conversation(
    client: genai.Client,
    record: dict,
    out_dir: Path,
    conv_id: str,
) -> dict:
    """
    Generate images for all turns with image_description in one conversation.
    Returns the updated record with image_path fields added.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    turns = record.get("turns", [])
    prior_images: list[Image.Image] = []

    for turn in turns:
        desc = turn.get("image_description")
        if not desc:
            turn["image_path"] = None
            continue

        turn_id  = turn["turn_id"]
        img_path = out_dir / f"turn_{turn_id:02d}.png"

        print(f"    turn {turn_id}: generating image ({len(prior_images)} prior) ...",
              file=sys.stderr)

        img = generate_image(client, desc, prior_images)

        if img is not None:
            img.save(img_path)
            turn["image_path"] = str(img_path)
            prior_images.append(img)
        else:
            print(f"    [warn] turn {turn_id}: image generation failed", file=sys.stderr)
            turn["image_path"] = None

    return record

# ─── JSONL processing ─────────────────────────────────────────────────────────

def process_jsonl(
    client: genai.Client,
    jsonl_path: Path,
    output_root: Path,
    limit: int | None = None,
):
    taxonomy = jsonl_path.parent.name
    stem     = jsonl_path.stem
    done = 0

    with open(jsonl_path) as f:
        lines = [l for l in f if l.strip() and '"_error"' not in l]

    if limit:
        lines = lines[:limit]

    print(f"\n[{taxonomy}] {len(lines)} conversations from {jsonl_path.name}", file=sys.stderr)

    for idx, line in enumerate(lines):
        record  = json.loads(line)
        conv_id = f"{stem}_{idx:04d}"
        out_dir = output_root / taxonomy / conv_id

        # Skip if already done
        conv_jsonl = out_dir / "conversation.jsonl"
        if conv_jsonl.exists():
            print(f"  [{idx+1}/{len(lines)}] {conv_id}: already done, skipping",
                  file=sys.stderr)
            done += 1
            continue

        print(f"  [{idx+1}/{len(lines)}] {conv_id}", file=sys.stderr)
        updated = process_conversation(client, record, out_dir, conv_id)

        with open(conv_jsonl, "w") as f:
            f.write(json.dumps(updated, ensure_ascii=False) + "\n")

        done += 1

    print(f"[{taxonomy}] done: {done}/{len(lines)}", file=sys.stderr)

# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate images for synthesized benchmark conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input-dir", required=True,
                   help="Directory containing taxonomy subdirs with JSONL files "
                        "(i.e. the output/ dir from synthesize.py), "
                        "or a specific taxonomy subdir.")
    p.add_argument("--output-dir", default="output_images",
                   help="Root dir for generated images. Default: output_images/")
    p.add_argument("--taxonomy", default=None,
                   help="Only process this taxonomy subdir (default: all found).")
    p.add_argument("--limit", type=int, default=None,
                   help="Max conversations to process per JSONL file (for testing).")
    return p.parse_args()


def main():
    args       = parse_args()
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    client     = make_client()

    # Collect JSONL files to process
    if input_dir.suffix == ".jsonl":
        jsonl_files = [input_dir]
    elif args.taxonomy:
        jsonl_files = sorted((input_dir / args.taxonomy).glob("*.jsonl"))
    else:
        jsonl_files = sorted(input_dir.rglob("*.jsonl"))

    # Exclude already-output conversation.jsonl files
    jsonl_files = [f for f in jsonl_files if f.name != "conversation.jsonl"]

    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[config] model={GEMINI_MODEL}  files={len(jsonl_files)}", file=sys.stderr)

    for jsonl_path in jsonl_files:
        process_jsonl(client, jsonl_path, output_dir, limit=args.limit)

    print("\nDone.")


if __name__ == "__main__":
    main()
