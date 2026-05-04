"""
Generate images for every ImageRef.description in the conversation JSONL files.

Reads:   outputs/conversations/<taxonomy>.jsonl
Writes:  outputs/images/<conversation_id>/<image_id>.png
Updates: each ImageRef.source_path in-place to point at the saved image.

Providers:
  - gemini   (default; gemini-2.5-flash-image; free tier ~500 images/day)
  - openai   (gpt-image-1, paid)

Daily quota:
  Tracked in outputs/images/.gemini_quota.json. Defaults to a hard cap of 450
  per UTC day (leaves a small safety margin under Gemini's 500/day free limit).
  Override with --daily-limit. The script refuses to start more jobs once the
  remaining quota would be exhausted, and persists the counter atomically.

Run:
  python scripts/generate_images.py --taxonomy belief_revision --dry-run
  python scripts/generate_images.py --taxonomy belief_revision --limit 5
  python scripts/generate_images.py --all --concurrency 4
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("t2i")

ROOT = Path(__file__).resolve().parent.parent
CONV_DIR = ROOT / "outputs" / "conversations"
IMG_ROOT = ROOT / "outputs" / "images"  # mutable default; overridable via --image-dir

PROMPT_PREFIX = (
    "Photorealistic image, neutral lighting, no text overlays, no watermarks. "
    "Render exactly the following scene: "
)

CONTEXT_PROMPT_SUFFIX = (
    "\n\nThis is the next image in a multi-image dialogue. Maintain strict visual consistency "
    "with the preceding image(s): same objects, same characters, same setting, same style, "
    "same lighting tone, same image quality. Only the angle / framing / state should change "
    "as described above; everything else must match the prior images."
)


# ----------------------------------------------------------------------------
# Provider implementations
# ----------------------------------------------------------------------------

class T2IProvider:
    name = "base"
    def generate(self, prompt: str, out_path: Path,
                 context_image_paths: list[Path] | None = None) -> None:
        raise NotImplementedError


class OpenAIProvider(T2IProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-image-1", size: str = "1024x1024"):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DIALOGVIS_T2I_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.size = size

    def generate(self, prompt: str, out_path: Path,
                 context_image_paths: list[Path] | None = None) -> None:
        # OpenAI gpt-image-1 doesn't take prior images via .generate; ignore context.
        resp = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size=self.size,
            n=1,
        )
        b64 = resp.data[0].b64_json
        out_path.write_bytes(base64.b64decode(b64))


class GeminiProvider(T2IProvider):
    """Gemini 2.5 Flash Image (a.k.a. 'nano-banana'). Free tier ~500/day."""
    name = "gemini"

    def __init__(self, model: str = "gemini-3.1-flash-image-preview"):
        from google import genai
        from google.genai import types
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self._types = types

    def generate(self, prompt: str, out_path: Path,
                 context_image_paths: list[Path] | None = None) -> None:
        contents = _build_contents(prompt, context_image_paths, self._types)
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )
        for cand in resp.candidates or []:
            for part in cand.content.parts:
                inline = getattr(part, "inline_data", None)
                if inline and inline.data:
                    out_path.write_bytes(inline.data)
                    return
        raise RuntimeError("Gemini returned no image part")


def _build_contents(prompt: str, context_image_paths: list[Path] | None, types_mod):
    """Construct a multimodal contents list: prior images followed by the prompt."""
    if not context_image_paths:
        return [prompt]
    parts = []
    for p in context_image_paths:
        parts.append(types_mod.Part.from_bytes(
            data=p.read_bytes(),
            mime_type="image/png",
        ))
    parts.append(prompt)
    return parts


class VertexProvider(T2IProvider):
    """Gemini 2.5 Flash Image via Vertex AI (GCP project credits, separate from AI Studio prepay)."""
    name = "vertex"

    def __init__(self, model: str = "gemini-3.1-flash-image-preview"):
        from google import genai
        from google.genai import types
        project = os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        if not project:
            raise RuntimeError(
                "VERTEX_PROJECT (or GOOGLE_CLOUD_PROJECT) not set. "
                "Also ensure ADC is configured (`gcloud auth application-default login`)."
            )
        self.client = genai.Client(vertexai=True, project=project, location=location)
        self.model = model
        self._types = types

    def generate(self, prompt: str, out_path: Path,
                 context_image_paths: list[Path] | None = None) -> None:
        contents = _build_contents(prompt, context_image_paths, self._types)
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        for cand in resp.candidates or []:
            for part in cand.content.parts:
                inline = getattr(part, "inline_data", None)
                if inline and inline.data:
                    out_path.write_bytes(inline.data)
                    return
        raise RuntimeError("Vertex returned no image part")


def make_provider(name: str) -> T2IProvider:
    if name == "openai":
        return OpenAIProvider()
    if name == "gemini":
        return GeminiProvider()
    if name == "vertex":
        return VertexProvider()
    raise ValueError(f"Unknown provider: {name}")


# ----------------------------------------------------------------------------
# Daily quota tracker (UTC)
# ----------------------------------------------------------------------------

class DailyQuota:
    """Thread-safe persistent UTC-day counter. Refuses calls past the limit."""

    def __init__(self, path: Path, limit: int):
        self.path = path
        self.limit = limit
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._date, self._used = self._load()

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load(self) -> tuple[str, int]:
        today = self._today()
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                if d.get("date") == today:
                    return today, int(d.get("used", 0))
            except Exception:
                pass
        return today, 0

    def _save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"date": self._date, "used": self._used, "limit": self.limit}))
        tmp.replace(self.path)

    def remaining(self) -> int:
        with self._lock:
            today = self._today()
            if today != self._date:
                self._date, self._used = today, 0
                self._save()
            return max(0, self.limit - self._used)

    def reserve(self) -> bool:
        """Atomically claim 1 unit of quota. Returns False if exhausted."""
        with self._lock:
            today = self._today()
            if today != self._date:
                self._date, self._used = today, 0
            if self._used >= self.limit:
                return False
            self._used += 1
            self._save()
            return True

    def refund(self) -> None:
        """Give back a reservation when a call fails (don't burn quota on errors)."""
        with self._lock:
            self._used = max(0, self._used - 1)
            self._save()


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

@dataclass
class Job:
    conv_id: str
    image_id: str
    description: str
    out_path: Path
    line_no: int
    file: Path


@dataclass
class ConvGroup:
    """All images in a single conversation, ordered by turn. Some may already exist on disk."""
    conv_id: str
    file: Path
    line_no: int
    scenario: str
    # Each entry: (image_id, description, out_path, already_exists)
    images: list[tuple[str, str, Path, bool]]


def collect_conv_groups(jsonl_files: list[Path]) -> tuple[list[ConvGroup], dict]:
    """Walk conversations, return per-conv ordered image lists."""
    groups: list[ConvGroup] = []
    file_lines: dict[Path, list[str]] = {}
    for f in jsonl_files:
        lines = f.read_text(encoding="utf-8").splitlines()
        file_lines[f] = lines
        for i, raw in enumerate(lines):
            if not raw.strip():
                continue
            conv = json.loads(raw)
            conv_id = conv["id"]
            images = []
            for t in sorted(conv["turns"], key=lambda x: x["turn_id"]):
                img = t.get("image")
                if not img:
                    continue
                out_path = IMG_ROOT / conv_id / f"{img['id']}.png"
                images.append((img["id"], img["description"], out_path, out_path.exists()))
                if out_path.exists():
                    img["source_path"] = str(out_path.relative_to(ROOT))
            file_lines[f][i] = json.dumps(conv, ensure_ascii=False)
            if images:
                groups.append(ConvGroup(
                    conv_id=conv_id,
                    file=f,
                    line_no=i,
                    scenario=conv.get("scenario", ""),
                    images=images,
                ))
    return groups, file_lines


def collect_jobs(jsonl_files: list[Path], skip_existing: bool) -> tuple[list[Job], dict]:
    """Walk all conversations and collect images that need to be generated."""
    jobs: list[Job] = []
    file_lines: dict[Path, list[str]] = {}
    for f in jsonl_files:
        lines = f.read_text(encoding="utf-8").splitlines()
        file_lines[f] = lines
        for i, raw in enumerate(lines):
            if not raw.strip():
                continue
            conv = json.loads(raw)
            conv_id = conv["id"]
            for t in conv["turns"]:
                img = t.get("image")
                if not img:
                    continue
                out_path = IMG_ROOT / conv_id / f"{img['id']}.png"
                if skip_existing and out_path.exists():
                    img["source_path"] = str(out_path.relative_to(ROOT))
                    continue
                jobs.append(Job(
                    conv_id=conv_id,
                    image_id=img["id"],
                    description=img["description"],
                    out_path=out_path,
                    line_no=i,
                    file=f,
                ))
            # rewrite line with potentially-updated source_paths
            file_lines[f][i] = json.dumps(conv, ensure_ascii=False)
    return jobs, file_lines


class RateLimiter:
    """Simple token-bucket throttle: at most `rpm` calls per 60s, across threads."""
    def __init__(self, rpm: int):
        self.interval = 60.0 / max(1, rpm)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(self._next_at, now)
            self._next_at = slot + self.interval
            sleep_for = slot - now
        if sleep_for > 0:
            time.sleep(sleep_for)


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


def run_job(provider: T2IProvider, job: Job, quota: DailyQuota | None,
            limiter: RateLimiter | None, max_retries: int = 5) -> tuple[Job, bool, str]:
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = PROMPT_PREFIX + job.description
    if quota is not None and not quota.reserve():
        return job, False, "QUOTA_EXHAUSTED"
    msg = ""
    for attempt in range(1, max_retries + 1):
        if limiter is not None:
            limiter.wait()
        try:
            provider.generate(prompt, job.out_path)
            return job, True, ""
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if _is_rate_limit(e):
                # Long backoff for rate limits (Vertex/AI Studio per-minute quota).
                delay = min(60.0 * attempt, 240.0)
                log.warning("Rate-limited (%d/%d) for %s/%s — sleeping %.0fs",
                            attempt, max_retries, job.conv_id, job.image_id, delay)
            else:
                delay = min(2 ** attempt, 30)
                log.warning("Attempt %d/%d failed for %s/%s: %s", attempt, max_retries,
                            job.conv_id, job.image_id, msg)
            time.sleep(delay)
    if quota is not None:
        quota.refund()
    return job, False, msg


def _seed_first_image(group: ConvGroup, seed_from: Path | None) -> list[tuple[str, str, Path, bool]]:
    """
    If seed_from is given, copy the first image of this conv from seed_from to
    its target location (if not already present there). Returns the (possibly
    updated) image specs with refreshed `exists` flags.
    """
    import shutil
    images = list(group.images)
    if not seed_from or not images:
        return images
    image_id, desc, out_path, exists = images[0]
    if exists:
        return images  # already in dest; nothing to do
    src = seed_from / group.conv_id / f"{image_id}.png"
    if src.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out_path)
        log.info("Seeded %s/%s from %s", group.conv_id, image_id, seed_from)
        images[0] = (image_id, desc, out_path, True)
    return images


def run_conv_group(provider: T2IProvider, group: ConvGroup, quota: DailyQuota | None,
                   limiter: RateLimiter | None, skip_existing: bool,
                   seed_from: Path | None = None,
                   max_retries: int = 5) -> tuple[ConvGroup, int, int, list[str]]:
    """
    Process one conversation's images in order, passing prior images as context.
    Returns (group, succeeded, failed, generated_image_ids).
    """
    succeeded = failed = 0
    generated_ids: list[str] = []
    prior_paths: list[Path] = []

    images = _seed_first_image(group, seed_from)

    for image_id, description, out_path, exists in images:
        # If image exists and we're skipping existing, just include it as context for later turns.
        if skip_existing and exists:
            prior_paths.append(out_path)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = PROMPT_PREFIX + description
        if prior_paths:
            prompt = prompt + CONTEXT_PROMPT_SUFFIX

        if quota is not None and not quota.reserve():
            failed += 1
            log.error("QUOTA_EXHAUSTED at %s/%s", group.conv_id, image_id)
            break  # quota is exhausted for the day; stop this group

        msg = ""
        ok = False
        for attempt in range(1, max_retries + 1):
            if limiter is not None:
                limiter.wait()
            try:
                provider.generate(prompt, out_path, context_image_paths=prior_paths or None)
                ok = True
                break
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if _is_rate_limit(e):
                    delay = min(60.0 * attempt, 240.0)
                    log.warning("Rate-limited (%d/%d) for %s/%s — sleeping %.0fs",
                                attempt, max_retries, group.conv_id, image_id, delay)
                else:
                    delay = min(2 ** attempt, 30)
                    log.warning("Attempt %d/%d failed for %s/%s: %s", attempt, max_retries,
                                group.conv_id, image_id, msg)
                time.sleep(delay)

        if ok:
            succeeded += 1
            generated_ids.append(image_id)
            prior_paths.append(out_path)
        else:
            failed += 1
            if quota is not None:
                quota.refund()
            log.error("FAILED %s/%s: %s", group.conv_id, image_id, msg)
            # Still continue with this conv — but later images will lack THIS one as context.
    return group, succeeded, failed, generated_ids


def write_back(file_lines: dict[Path, list[str]], jsonl_out: Path | None = None) -> None:
    """Persist updated source_paths to disk. If jsonl_out is set, write there instead."""
    if jsonl_out is not None:
        jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        # If multiple input files were collected, concatenate; usually single file in practice.
        with jsonl_out.open("w", encoding="utf-8") as f:
            for path, lines in file_lines.items():
                for line in lines:
                    if line.strip():
                        f.write(line + "\n")
        log.info("Wrote updated JSONL to %s", jsonl_out)
        return
    for f, lines in file_lines.items():
        tmp = f.with_suffix(f.suffix + ".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(f)


def update_source_paths(jsonl_files: list[Path]) -> None:
    """Re-scan and rewrite source_path fields based on existing image files."""
    for f in jsonl_files:
        lines = f.read_text(encoding="utf-8").splitlines()
        out_lines = []
        for raw in lines:
            if not raw.strip():
                out_lines.append(raw)
                continue
            conv = json.loads(raw)
            conv_id = conv["id"]
            for t in conv["turns"]:
                img = t.get("image")
                if not img:
                    continue
                p = IMG_ROOT / conv_id / f"{img['id']}.png"
                if p.exists():
                    img["source_path"] = str(p.relative_to(ROOT))
            out_lines.append(json.dumps(conv, ensure_ascii=False))
        tmp = f.with_suffix(f.suffix + ".tmp")
        tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        tmp.replace(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", help="Process only this taxonomy file (e.g. belief_revision)")
    ap.add_argument("--all", action="store_true", help="Process every taxonomy file under outputs/conversations/")
    ap.add_argument("--provider", default="gemini", choices=["openai", "gemini", "vertex"])
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="Cap total images to generate (for smoke testing)")
    ap.add_argument("--max-convs", type=int, default=None,
                    help="(--with-context) cap to the first N conversations after sorting; combine with --offset for slicing")
    ap.add_argument("--offset", type=int, default=0,
                    help="(--with-context) skip the first N conversations before processing")
    ap.add_argument("--rpm", type=int, default=10,
                    help="Cap requests-per-minute across all workers (default 10; Vertex new-account RPM is often 10).")
    ap.add_argument("--daily-limit", type=int, default=450,
                    help="Hard cap on Gemini calls per UTC day (default 450, leaves margin under 500/day free tier).")
    ap.add_argument("--dry-run", action="store_true", help="List jobs, don't call API")
    ap.add_argument("--no-skip", action="store_true", help="Regenerate images even if PNG exists")
    ap.add_argument("--with-context", action="store_true",
                    help="Generate per-conversation in turn order, passing prior images as visual context "
                         "for consistency. Within a conv it is sequential; concurrency parallelizes across convs.")
    ap.add_argument("--image-dir", type=Path, default=None,
                    help="Output directory for generated PNGs (default: outputs/images).")
    ap.add_argument("--seed-from", type=Path, default=None,
                    help="At the start of each conversation, copy the FIRST image (lowest turn_id) "
                         "from this dir into --image-dir if missing. Combine with --with-context to "
                         "keep the first image fixed and re-generate the rest with consistency.")
    ap.add_argument("--jsonl-out", type=Path, default=None,
                    help="Optional path to write updated JSONL with new source_path values; "
                         "if unset, source paths are written back to the input JSONL in-place.")
    args = ap.parse_args()

    # Allow overriding the global image dir.
    global IMG_ROOT
    if args.image_dir:
        IMG_ROOT = args.image_dir.resolve()
        log.info("Image output dir: %s", IMG_ROOT)
    IMG_ROOT.mkdir(parents=True, exist_ok=True)

    if not (args.taxonomy or args.all):
        ap.error("Pass --taxonomy <name> or --all")

    if args.all:
        files = sorted(CONV_DIR.glob("*.jsonl"))
    else:
        f = CONV_DIR / f"{args.taxonomy}.jsonl"
        if not f.exists():
            ap.error(f"No such file: {f}")
        files = [f]

    skip_existing = not args.no_skip

    if args.with_context:
        groups, file_lines = collect_conv_groups(files)
        # Slice by offset/max-convs FIRST (against the full conv list, so positions are stable),
        # then drop convs that have nothing to generate.
        if args.offset:
            groups = groups[args.offset:]
        if args.max_convs is not None:
            groups = groups[: args.max_convs]
        if skip_existing:
            groups = [g for g in groups if any(not exists for _, _, _, exists in g.images)]
        pending_count = sum(
            sum(1 for _, _, _, exists in g.images if (not skip_existing) or (not exists))
            for g in groups
        )
        # Recompute pending count after slicing
        pending_count = sum(
            sum(1 for _, _, _, exists in g.images if (not skip_existing) or (not exists))
            for g in groups
        )
        log.info("Found %d conversations with %d images to generate across %d files (offset=%d max_convs=%s)",
                 len(groups), pending_count, len(files), args.offset, args.max_convs)
        if args.limit:
            # Limit by total images, truncating conv list.
            kept, count = [], 0
            for g in groups:
                kept.append(g)
                count += sum(1 for _, _, _, exists in g.images if (not skip_existing) or (not exists))
                if count >= args.limit:
                    break
            groups = kept
            log.info("Capped to %d conversations", len(groups))
        if args.dry_run:
            for g in groups[:10]:
                print(f"  conv={g.conv_id[:8]}  scenario={g.scenario[:60]}")
                for image_id, desc, _, exists in g.images:
                    mark = "[exists]" if exists else "[gen]   "
                    print(f"    {mark} {image_id}: {desc[:70]}")
            print(f"... ({len(groups)} convs total)")
            return
        if not groups:
            log.info("Nothing to do.")
            write_back(file_lines, jsonl_out=args.jsonl_out)
            return
    else:
        jobs, file_lines = collect_jobs(files, skip_existing=skip_existing)
        log.info("Found %d images to generate across %d files", len(jobs), len(files))

        if args.limit:
            jobs = jobs[: args.limit]
            log.info("Capped to %d jobs", len(jobs))

        if args.dry_run:
            for j in jobs[:10]:
                print(f"  {j.conv_id}/{j.image_id}: {j.description[:80]}")
            print(f"... ({len(jobs)} total)")
            return

        if not jobs:
            log.info("Nothing to do.")
            write_back(file_lines, jsonl_out=args.jsonl_out)
            return

    provider = make_provider(args.provider)
    log.info("Provider: %s", provider.name)

    quota: DailyQuota | None = None
    if args.provider in ("gemini", "vertex"):
        quota = DailyQuota(IMG_ROOT / ".gemini_quota.json", limit=args.daily_limit)
        remaining = quota.remaining()
        log.info("Gemini daily quota: %d/%d remaining today (UTC).", remaining, args.daily_limit)
        if remaining <= 0:
            log.error("Daily quota exhausted. Try again after UTC midnight.")
            return
        if args.with_context:
            pending = sum(
                sum(1 for _, _, _, exists in g.images if (not skip_existing) or (not exists))
                for g in groups
            )
            if pending > remaining:
                log.warning("Have %d pending images but only %d quota left today.", pending, remaining)
        else:
            if len(jobs) > remaining:
                log.warning("Have %d jobs but only %d quota left today — will stop early; rerun tomorrow.", len(jobs), remaining)

    limiter = RateLimiter(args.rpm) if args.rpm > 0 else None
    if limiter:
        log.info("Throttling to %d RPM (one request every %.1fs).", args.rpm, 60.0 / args.rpm)

    succeeded = failed = quota_stop = 0
    if args.with_context:
        total_imgs = sum(
            sum(1 for _, _, _, exists in g.images if (not skip_existing) or (not exists))
            for g in groups
        )
        completed_imgs = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            seed_from = args.seed_from.resolve() if args.seed_from else None
            futures = {pool.submit(run_conv_group, provider, g, quota, limiter, skip_existing,
                                   seed_from): g for g in groups}
            for done_idx, fut in enumerate(as_completed(futures), 1):
                group, g_ok, g_fail, generated_ids = fut.result()
                succeeded += g_ok
                failed += g_fail
                completed_imgs += g_ok + g_fail
                # Update file lines with new source_paths.
                lines = file_lines[group.file]
                conv = json.loads(lines[group.line_no])
                for t in conv["turns"]:
                    img = t.get("image")
                    if img and img["id"] in generated_ids:
                        p = IMG_ROOT / group.conv_id / f"{img['id']}.png"
                        img["source_path"] = str(p.relative_to(ROOT))
                lines[group.line_no] = json.dumps(conv, ensure_ascii=False)
                if done_idx % 5 == 0 or done_idx == len(futures):
                    log.info("Convs done: %d/%d  imgs: ok=%d fail=%d (%d/%d)",
                             done_idx, len(futures), succeeded, failed,
                             completed_imgs, total_imgs)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(run_job, provider, j, quota, limiter): j for j in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                job, ok, err = fut.result()
                if ok:
                    succeeded += 1
                    lines = file_lines[job.file]
                    conv = json.loads(lines[job.line_no])
                    for t in conv["turns"]:
                        if t.get("image") and t["image"]["id"] == job.image_id:
                            t["image"]["source_path"] = str(job.out_path.relative_to(ROOT))
                            break
                    lines[job.line_no] = json.dumps(conv, ensure_ascii=False)
                else:
                    if err == "QUOTA_EXHAUSTED":
                        quota_stop += 1
                    else:
                        failed += 1
                        log.error("FAILED %s/%s: %s", job.conv_id, job.image_id, err)
                if i % 10 == 0:
                    log.info("Progress: %d/%d (ok=%d, fail=%d, quota_skipped=%d)",
                             i, len(jobs), succeeded, failed, quota_stop)

    write_back(file_lines, jsonl_out=args.jsonl_out)
    log.info("Done. ok=%d failed=%d quota_skipped=%d images_dir=%s",
             succeeded, failed, quota_stop, IMG_ROOT)
    if quota is not None:
        log.info("Gemini quota now: %d remaining today.", quota.remaining())


if __name__ == "__main__":
    sys.exit(main())
