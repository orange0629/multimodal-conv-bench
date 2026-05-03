#!/usr/bin/env python3
"""
Conversation Synthesis Pipeline
================================
Generates multi-turn, multi-modal benchmark conversations for each taxonomy.

Backends
--------
  --backend vllm     vLLM library (from vllm import LLM) — default
  --backend openai   OpenAI API (requires OPENAI_API_KEY)

Model shortcuts (--model)
--------------------------
  gemma    →  google/gemma-4-31B-it
  qwen     →  Qwen/Qwen3.5-27B
  <path>   →  any local path or full HuggingFace model ID

Modes
-----
  --mode text-only       LLM writes image_description fields (no real images needed)
  --mode with-images     Pair with real seed images from --seed-image-dir

Usage examples
--------------
  # Quick debug: one sample, Qwen, text-only
  python pipeline/synthesize.py \\
      --taxonomy ist \\
      --scenario "A cluttered desk being gradually tidied up" \\
      --n 1 --backend vllm --model qwen

  # All taxonomies with Gemma, from config
  python pipeline/synthesize.py \\
      --config configs/scenarios.yaml --taxonomy all \\
      --n 3 --backend vllm --model gemma

  # Use a local cached model path
  python pipeline/synthesize.py \\
      --taxonomy br --scenario "..." \\
      --model /work/hdd/bfuj/lzhang49/huggingface/Qwen3.5-27B
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from prompts import (TAXONOMIES, TAXONOMY_ALIASES, build_messages,
                     build_scenario_gen_messages)

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ─── Model shorthands & per-model configs ────────────────────────────────────

MODEL_SHORTHANDS = {
    "gemma": "google/gemma-4-31B-it",
    "qwen":  "Qwen/Qwen3.5-27B",
}

# Per-model defaults; looked up by resolved model ID (or longest prefix match)
MODEL_CONFIGS = {
    "google/gemma-4-31B-it": {
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_tokens": 4096},
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "Qwen/Qwen3.5-27B": {
        "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                     "presence_penalty": 1.5, "max_tokens": 4096},
        "chat_template_kwargs": {"enable_thinking": False},
    },
}

# Settings applied when --memory-efficient is passed.
# Needed for large models (Qwen3.5-27B / Gemma-31B) on 2×A40 (80 GB total):
#   - dtype=bfloat16       : halves weight memory vs fp32
#   - gpu_memory_utilization=0.90 : leave 10% headroom
#   - max_model_len=8192   : caps KV cache (default 262K context → OOM)
#   - max_num_seqs=64      : shrinks profiling forward pass activation memory
#   - enforce_eager=True   : disables CUDA graph pre-allocation (~2-4 GB)
MEMORY_EFFICIENT_VLLM = {
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.90,
    "max_model_len": 8192,
    "max_num_seqs": 64,
    "enforce_eager": True,
}

_FALLBACK_CONFIG = {
    "sampling": {"temperature": 0.9, "top_p": 0.95, "max_tokens": 4096},
    "chat_template_kwargs": {},
}


def resolve_model(model_arg: str) -> str:
    """Expand shorthand (gemma/qwen) to full model ID."""
    return MODEL_SHORTHANDS.get(model_arg, model_arg)


def get_model_config(model_id: str) -> dict:
    """Return sampling + chat_template_kwargs for a model ID."""
    if model_id in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_id]
    # fuzzy match: allow local paths that contain a known model name
    for key, cfg in MODEL_CONFIGS.items():
        if key.split("/")[-1].lower() in model_id.lower():
            return cfg
    return _FALLBACK_CONFIG

# ─── Model loading ────────────────────────────────────────────────────────────

def load_vllm(model_id: str, tensor_parallel_size: int = 1, memory_efficient: bool = False):
    from vllm import LLM
    kwargs: dict = {"tensor_parallel_size": tensor_parallel_size}
    if memory_efficient:
        kwargs.update(MEMORY_EFFICIENT_VLLM)
        print(f"[vllm] Loading {model_id} (tp={tensor_parallel_size}, memory-efficient) ...",
              file=sys.stderr)
    else:
        print(f"[vllm] Loading {model_id} (tp={tensor_parallel_size}) ...", file=sys.stderr)
    llm = LLM(model=model_id, **kwargs)
    print("[vllm] Model ready.", file=sys.stderr)
    return llm, llm.get_tokenizer()

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _parse_json(text: str) -> dict:
    """Strip markdown fences and common model JSON quirks, then parse."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])
    # Strip trailing commas before ] or } — common model output artifact
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return json.loads(text)

# ─── Backend implementations ──────────────────────────────────────────────────

def generate_vllm_batch(llm, tok, model_id: str,
                        messages_list: list[list[dict]], temperature: float) -> list[str]:
    """Submit all prompts in one vLLM call for maximum throughput."""
    from vllm import SamplingParams
    cfg = get_model_config(model_id)
    sp  = {**cfg["sampling"], "temperature": temperature}

    prompts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                **cfg["chat_template_kwargs"])
        for m in messages_list
    ]
    print(f"[vllm] submitting batch of {len(prompts)} prompts ...", file=sys.stderr)
    outputs = llm.generate(prompts, SamplingParams(**sp))
    return [_strip_thinking(o.outputs[0].text.strip()) for o in outputs]


def generate_openai_batch(model_id: str, messages_list: list[list[dict]],
                          temperature: float) -> list[str]:
    """Sequential OpenAI calls (no native batch API)."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    results = []
    for messages in messages_list:
        resp = client.chat.completions.create(
            model=model_id, messages=messages, temperature=temperature,
            response_format={"type": "json_object"},
        )
        results.append(resp.choices[0].message.content.strip())
    return results

# ─── Core generation ──────────────────────────────────────────────────────────

def _make_meta(taxonomy, scenario, mode, model_id, backend):
    return {
        "taxonomy": taxonomy, "scenario": scenario, "mode": mode,
        "model": model_id, "backend": backend,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _parse_raw(raw: str, taxonomy: str, scenario: str,
               mode: str, model_id: str, backend: str) -> dict:
    parsed = _parse_json(raw)
    parsed["_meta"] = _make_meta(taxonomy, scenario, mode, model_id, backend)
    return parsed


def run_batch(
    backend: str,
    model_id: str,
    taxonomy: str,
    scenarios: list[str],
    n: int,
    mode: str,
    output_dir: Path,
    temperature: float,
    llm=None,
    tok=None,
    verbose: bool = False,
    max_retries: int = 3,
) -> Path:
    tax_key  = TAXONOMY_ALIASES.get(taxonomy, taxonomy)
    out_dir  = output_dir / tax_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"

    # Build every (scenario, repeat) job
    jobs = [(s, i) for s in scenarios for i in range(n)]
    all_messages = [
        build_messages(tax_key, scenario, mode)
        for scenario, _ in jobs
    ]

    print(f"[{tax_key}] {len(jobs)} conversations to generate", file=sys.stderr)

    # Generate — one big batch for vllm, sequential for openai
    if backend == "vllm":
        raws = generate_vllm_batch(llm, tok, model_id, all_messages, temperature)
    else:
        raws = generate_openai_batch(model_id, all_messages, temperature)

    # Parse results; retry individually on JSON failure
    with open(out_file, "w") as f:
        for idx, ((scenario, i), raw) in enumerate(zip(jobs, raws)):
            try:
                result = _parse_raw(raw, tax_key, scenario, mode, model_id, backend)
            except json.JSONDecodeError as e:
                # Retry this one item sequentially
                print(f"  [retry] item {idx} JSON error: {e} — retrying ...", file=sys.stderr)
                result = None
                for attempt in range(1, max_retries + 1):
                    try:
                        if backend == "vllm":
                            r = generate_vllm_batch(llm, tok, model_id,
                                                    [all_messages[idx]], temperature)[0]
                        else:
                            r = generate_openai_batch(model_id,
                                                      [all_messages[idx]], temperature)[0]
                        result = _parse_raw(r, tax_key, scenario, mode, model_id, backend)
                        break
                    except Exception as e2:
                        print(f"    attempt {attempt}/{max_retries}: {e2}", file=sys.stderr)
                if result is None:
                    result = {"_error": f"failed after {max_retries} retries",
                              "_meta": _make_meta(tax_key, scenario, mode, model_id, backend)}

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            if verbose:
                print(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"[{tax_key}] wrote {len(jobs)} records → {out_file}", file=sys.stderr)
    return out_file

# ─── Config loading ────────────────────────────────────────────────────────────

def load_scenarios_from_config(config_path: str, taxonomy: str) -> dict[str, list[str]]:
    """Load from YAML (static config) or a directory of generated JSONL files."""
    path = Path(config_path)

    # Directory of generated JSONL files (one per taxonomy)
    if path.is_dir():
        tax_key = TAXONOMY_ALIASES.get(taxonomy, taxonomy)
        result: dict[str, list[str]] = {}
        targets = list(TAXONOMIES.keys()) if taxonomy == "all" else [tax_key]
        for tax in targets:
            jsonl = path / f"{tax}.jsonl"
            if jsonl.exists():
                descriptions = []
                with open(jsonl) as f:
                    for line in f:
                        obj = json.loads(line)
                        descriptions.append(obj["description"])
                result[tax] = descriptions
            else:
                print(f"  [warn] no scenario file for '{tax}' in {path}", file=sys.stderr)
        return result

    # YAML static config
    with open(path) as f:
        cfg = yaml.safe_load(f)
    tax_key = TAXONOMY_ALIASES.get(taxonomy, taxonomy)
    if taxonomy == "all":
        return {k: v.get("scenarios", []) for k, v in cfg.items() if k in TAXONOMIES}
    elif tax_key in cfg:
        return {tax_key: cfg[tax_key].get("scenarios", [])}
    else:
        raise ValueError(f"Taxonomy '{taxonomy}' not in config {config_path}")

# ─── Scenario generation (two-layer) ──────────────────────────────────────────

DEFAULT_SCENARIO_DIR = Path(__file__).parent.parent / "configs" / "generated"


def _call_llm(backend: str, model_id: str, messages: list[dict],
              temperature: float, llm=None, tok=None) -> str:
    """Single LLM call — uses pre-loaded llm/tok for vllm, OpenAI client otherwise."""
    if backend == "vllm":
        return generate_vllm_batch(llm, tok, model_id, [messages], temperature)[0]
    else:
        return generate_openai_batch(model_id, [messages], temperature)[0]


def _gen_scenarios_for_taxonomy(
    backend: str, model_id: str, taxonomy: str,
    n_themes: int, n_per_theme: int,
    temperature: float, out_dir: Path,
    llm=None, tok=None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{taxonomy}.jsonl"

    print(f"\n[gen-scenarios] {taxonomy}: {n_themes} themes × {n_per_theme} = "
          f"{n_themes * n_per_theme} scenarios", file=sys.stderr)

    # Layer 1: themes (single prompt)
    messages = build_scenario_gen_messages(taxonomy=taxonomy, layer=1,
                                           n_themes=n_themes, used_domains=[])
    print("  [layer1] generating themes ...", file=sys.stderr)
    raw    = _call_llm(backend, model_id, messages, temperature, llm, tok)
    themes = _parse_json(_strip_thinking(raw))
    if not isinstance(themes, list):
        raise ValueError(f"Layer 1 returned non-list: {raw[:200]}")
    print(f"  [layer1] got {len(themes)} themes", file=sys.stderr)

    # Layer 2: batch all theme prompts together for vllm
    layer2_messages = [
        build_scenario_gen_messages(
            taxonomy=taxonomy, layer=2, n_per_theme=n_per_theme,
            theme=theme, theme_id=theme_id,
            existing_scenarios=[],   # diversity handled by layer1 themes
        )
        for theme_id, theme in enumerate(themes)
    ]

    if backend == "vllm":
        print(f"  [layer2] batching {len(layer2_messages)} theme prompts ...", file=sys.stderr)
        layer2_raws = generate_vllm_batch(llm, tok, model_id, layer2_messages, temperature)
    else:
        layer2_raws = [
            _call_llm(backend, model_id, m, temperature, llm, tok)
            for m in layer2_messages
        ]

    # Parse layer 2 results
    all_scenarios: list[dict] = []
    for theme_id, (theme, raw) in enumerate(zip(themes, layer2_raws)):
        print(f"  [layer2] theme {theme_id:02d} '{theme['theme']}' ...", file=sys.stderr)
        try:
            scenarios = _parse_json(_strip_thinking(raw))
            if not isinstance(scenarios, list):
                raise ValueError(f"non-list: {raw[:200]}")
        except Exception as e:
            print(f"    [error] {e}", file=sys.stderr)
            continue

        for idx, s in enumerate(scenarios):
            s["scenario_id"] = f"{taxonomy}_{theme_id:02d}_{idx:02d}"
            s["theme"] = theme["theme"]
            s["domain"] = theme["domain"]
            all_scenarios.append(s)
        print(f"    → {len(scenarios)} scenarios ({len(all_scenarios)} total)", file=sys.stderr)

    with open(out_file, "w") as f:
        for s in all_scenarios:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"  wrote {len(all_scenarios)} → {out_file}", file=sys.stderr)
    return out_file


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Synthesize multi-turn multi-modal benchmark conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--taxonomy", "-t", default="all",
                   help="Taxonomy name, alias, or 'all'. "
                        f"Keys: {list(TAXONOMIES)}.  Aliases: {list(TAXONOMY_ALIASES)}")
    p.add_argument("--scenario", "-s", default=None,
                   help="Scenario string (single run; overrides --config).")
    p.add_argument("--config", "-c", default=None,
                   help="YAML file or directory of generated JSONL files.")
    p.add_argument("--n", "-n", type=int, default=1,
                   help="Samples per scenario. Default: 1")
    p.add_argument("--mode", choices=["text-only", "with-images"], default="text-only")
    p.add_argument("--seed-image-dir", default=None)

    p.add_argument("--backend", choices=["vllm", "openai"], default="vllm")
    p.add_argument("--model", "-m", default="qwen",
                   help=f"Model shorthand (gemma|qwen) or full ID. Shorthands: {MODEL_SHORTHANDS}")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor parallel size for vLLM. Default: 1")
    p.add_argument("--memory-efficient", action="store_true",
                   help="Enable memory-saving vLLM options (dtype=bfloat16, "
                        "max_model_len=8192, enforce_eager, etc.). "
                        "Use when loading large models on 2×A40.")

    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--verbose", "-v", action="store_true")

    # Scenario generation config (auto-runs on first use, cached after)
    p.add_argument("--n-themes", type=int, default=10,
                   help="Layer-1 themes per taxonomy (used only when generating). Default: 10")
    p.add_argument("--n-per-theme", type=int, default=10,
                   help="Layer-2 scenarios per theme (used only when generating). Default: 10")
    p.add_argument("--scenario-dir", default=str(DEFAULT_SCENARIO_DIR),
                   help=f"Cache dir for generated scenario JSONL files. Default: {DEFAULT_SCENARIO_DIR}")
    p.add_argument("--regen", action="store_true",
                   help="Force scenario regeneration even if cache exists.")
    return p.parse_args()


def _ensure_scenarios(args, model_id: str, eff_temp: float, targets: list[str],
                      llm=None, tok=None) -> dict[str, list[str]]:
    """Return scenario descriptions per taxonomy, generating missing ones on the fly."""
    scenario_dir = Path(args.scenario_dir)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    taxonomy_scenarios: dict[str, list[str]] = {}

    for taxonomy in targets:
        cache_file = scenario_dir / f"{taxonomy}.jsonl"

        if cache_file.exists() and not args.regen:
            print(f"[scenarios] {taxonomy}: using cache ({cache_file})", file=sys.stderr)
        else:
            print(f"[scenarios] {taxonomy}: cache missing — generating ...", file=sys.stderr)
            try:
                _gen_scenarios_for_taxonomy(
                    args.backend, model_id, taxonomy,
                    n_themes=args.n_themes,
                    n_per_theme=args.n_per_theme,
                    temperature=eff_temp,
                    out_dir=scenario_dir,
                    llm=llm, tok=tok,
                )
            except Exception as e:
                print(f"[error] scenario generation failed for '{taxonomy}': {e}", file=sys.stderr)
                continue

        descriptions = []
        with open(cache_file) as f:
            for line in f:
                obj = json.loads(line)
                descriptions.append(obj["description"])
        taxonomy_scenarios[taxonomy] = descriptions

    return taxonomy_scenarios


def main():
    args     = parse_args()
    model_id = resolve_model(args.model)
    eff_temp = args.temperature if args.temperature is not None \
               else get_model_config(model_id)["sampling"]["temperature"]

    print(f"[config] backend={args.backend}  model={model_id}  "
          f"tp={args.tp}  mode={args.mode}", file=sys.stderr)

    # ── Load model once upfront ────────────────────────────────────────────────
    llm, tok = (load_vllm(model_id, args.tp, args.memory_efficient) if args.backend == "vllm"
                else (None, None))

    tax_key = TAXONOMY_ALIASES.get(args.taxonomy, args.taxonomy)
    targets = list(TAXONOMIES.keys()) if args.taxonomy == "all" else [tax_key]

    # Resolve scenarios — inline string skips generation entirely
    if args.scenario:
        taxonomy_scenarios = {t: [args.scenario] for t in targets}
    elif args.config:
        taxonomy_scenarios = load_scenarios_from_config(args.config, args.taxonomy)
    else:
        taxonomy_scenarios = _ensure_scenarios(args, model_id, eff_temp, targets,
                                               llm=llm, tok=tok)

    out_dir = Path(args.output_dir)
    all_outputs = []
    for taxonomy, scenarios in taxonomy_scenarios.items():
        if not scenarios:
            print(f"[skip] no scenarios for '{taxonomy}'", file=sys.stderr)
            continue
        out_file = run_batch(
            args.backend, model_id, taxonomy, scenarios,
            n=args.n,
            mode=args.mode,
            output_dir=out_dir,
            temperature=eff_temp,
            llm=llm, tok=tok,
            verbose=args.verbose,
        )
        all_outputs.append(str(out_file))

    print("\nOutput files:")
    for p_str in all_outputs:
        print(f"  {p_str}")


if __name__ == "__main__":
    main()
