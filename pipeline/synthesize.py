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

from prompts import TAXONOMIES, TAXONOMY_ALIASES, build_messages

DEFAULT_N_TURNS  = 4
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

# ─── Model cache (loaded once per process) ────────────────────────────────────

_vllm_cache: dict = {}   # model_id -> LLM instance


def _load_vllm(model_id: str, tensor_parallel_size: int = 1):
    if model_id not in _vllm_cache:
        from vllm import LLM
        print(f"[vllm] Loading {model_id} (tp={tensor_parallel_size}) ...", file=sys.stderr)
        _vllm_cache[model_id] = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
        )
        print("[vllm] Model ready.", file=sys.stderr)
    return _vllm_cache[model_id]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _parse_json(text: str) -> dict:
    """Strip optional markdown fences then parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])
    return json.loads(text)

# ─── Backend implementations ──────────────────────────────────────────────────

def _generate_vllm(model_id: str, messages: list[dict],
                   temperature: float, tensor_parallel_size: int) -> str:
    from vllm import SamplingParams

    llm   = _load_vllm(model_id, tensor_parallel_size)
    tok   = llm.get_tokenizer()
    cfg   = get_model_config(model_id)
    sp    = {**cfg["sampling"], "temperature": temperature}

    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **cfg["chat_template_kwargs"],
    )

    params  = SamplingParams(**sp)
    outputs = llm.generate([prompt], params)
    raw     = outputs[0].outputs[0].text.strip()
    return _strip_thinking(raw)


def _generate_openai(model_id: str, messages: list[dict],
                     temperature: float, **_) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content.strip()


BACKENDS = {
    "vllm":   _generate_vllm,
    "openai": _generate_openai,
}

# ─── Core generation ──────────────────────────────────────────────────────────

def generate_one(
    backend: str,
    model_id: str,
    taxonomy: str,
    scenario: str,
    n_turns: int,
    mode: str,
    seed_image_description: str | None = None,
    temperature: float | None = None,
    tensor_parallel_size: int = 1,
    max_retries: int = 3,
) -> dict:
    taxonomy  = TAXONOMY_ALIASES.get(taxonomy, taxonomy)
    messages  = build_messages(taxonomy, scenario, n_turns, mode, seed_image_description)
    gen_fn    = BACKENDS[backend]
    eff_temp  = temperature if temperature is not None \
                else get_model_config(model_id)["sampling"]["temperature"]

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = gen_fn(model_id, messages, eff_temp, tensor_parallel_size=tensor_parallel_size)
        except Exception as e:
            last_err = e
            print(f"  [retry {attempt}/{max_retries}] generation error: {e}", file=sys.stderr)
            continue
        try:
            parsed = _parse_json(raw)
        except json.JSONDecodeError as e:
            last_err = e
            print(f"  [retry {attempt}/{max_retries}] JSON parse error: {e}\n"
                  f"  Raw (first 400): {raw[:400]}", file=sys.stderr)
            continue

        parsed["_meta"] = {
            "taxonomy":  taxonomy,
            "scenario":  scenario,
            "mode":      mode,
            "model":     model_id,
            "backend":   backend,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        return parsed

    raise RuntimeError(f"Failed after {max_retries} attempts. Last: {last_err}")

# ─── Batch runner ─────────────────────────────────────────────────────────────

def run_batch(
    backend: str,
    model_id: str,
    taxonomy: str,
    scenarios: list[str],
    n: int,
    n_turns: int,
    mode: str,
    output_dir: Path,
    temperature: float | None = None,
    tensor_parallel_size: int = 1,
    verbose: bool = False,
) -> Path:
    tax_key = TAXONOMY_ALIASES.get(taxonomy, taxonomy)
    out_dir = output_dir / tax_key
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file  = out_dir / f"{timestamp}.jsonl"
    total     = len(scenarios) * n
    done      = 0

    with open(out_file, "w") as f:
        for scenario in scenarios:
            for i in range(n):
                done += 1
                print(f"[{done}/{total}] {tax_key} | sample {i+1} | '{scenario[:60]}'",
                      file=sys.stderr)
                try:
                    result = generate_one(
                        backend, model_id, tax_key, scenario,
                        n_turns, mode,
                        temperature=temperature,
                        tensor_parallel_size=tensor_parallel_size,
                    )
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    if verbose:
                        print(json.dumps(result, indent=2, ensure_ascii=False))
                except Exception as e:
                    print(f"  [error] {e}", file=sys.stderr)
                    f.write(json.dumps({
                        "_error": str(e),
                        "_meta": {"taxonomy": tax_key, "scenario": scenario, "sample": i},
                    }) + "\n")
                    f.flush()

    print(f"\n[done] {total} records → {out_file}", file=sys.stderr)
    return out_file

# ─── Config loading ────────────────────────────────────────────────────────────

def load_scenarios_from_config(config_path: str, taxonomy: str) -> dict[str, list[str]]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    tax_key = TAXONOMY_ALIASES.get(taxonomy, taxonomy)
    if taxonomy == "all":
        return {k: v.get("scenarios", []) for k, v in cfg.items() if k in TAXONOMIES}
    elif tax_key in cfg:
        return {tax_key: cfg[tax_key].get("scenarios", [])}
    else:
        raise ValueError(f"Taxonomy '{taxonomy}' not in config {config_path}")

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
                   help="YAML config with scenarios (see configs/scenarios.yaml).")
    p.add_argument("--n", "-n", type=int, default=1,
                   help="Samples per scenario. Default: 1")
    p.add_argument("--n-turns", type=int, default=DEFAULT_N_TURNS,
                   help=f"User+assistant turn pairs per conversation. Default: {DEFAULT_N_TURNS}")
    p.add_argument("--mode", choices=["text-only", "with-images"], default="text-only")
    p.add_argument("--seed-image-dir", default=None)

    p.add_argument("--backend", choices=["vllm", "openai"], default="vllm",
                   help="Inference backend. Default: vllm")
    p.add_argument("--model", "-m", default="qwen",
                   help="Model shorthand (gemma | qwen) or full model ID / local path. "
                        f"Shorthands: {MODEL_SHORTHANDS}")
    p.add_argument("--temperature", type=float, default=None,
                   help="Override sampling temperature (uses per-model default if omitted).")
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor parallel size for vLLM (number of GPUs). Default: 1")

    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    model_id = resolve_model(args.model)
    out_dir  = Path(args.output_dir)

    print(f"[config] backend={args.backend}  model={model_id}  "
          f"tp={args.tp}  mode={args.mode}", file=sys.stderr)

    # Resolve taxonomy → scenarios
    if args.scenario:
        tax_key = TAXONOMY_ALIASES.get(args.taxonomy, args.taxonomy)
        if args.taxonomy == "all":
            taxonomy_scenarios = {k: [args.scenario] for k in TAXONOMIES}
        else:
            taxonomy_scenarios = {tax_key: [args.scenario]}
    elif args.config:
        taxonomy_scenarios = load_scenarios_from_config(args.config, args.taxonomy)
    else:
        default_cfg = Path(__file__).parent.parent / "configs" / "scenarios.yaml"
        if default_cfg.exists():
            print(f"[info] No --scenario/--config; using {default_cfg}", file=sys.stderr)
            taxonomy_scenarios = load_scenarios_from_config(str(default_cfg), args.taxonomy)
        else:
            print("Error: provide --scenario, --config, or populate configs/scenarios.yaml",
                  file=sys.stderr)
            sys.exit(1)

    all_outputs = []
    for taxonomy, scenarios in taxonomy_scenarios.items():
        if not scenarios:
            print(f"[skip] No scenarios for '{taxonomy}'", file=sys.stderr)
            continue
        out_file = run_batch(
            args.backend, model_id, taxonomy, scenarios,
            n=args.n,
            n_turns=args.n_turns,
            mode=args.mode,
            output_dir=out_dir,
            temperature=args.temperature,
            tensor_parallel_size=args.tp,
            verbose=args.verbose,
        )
        all_outputs.append(str(out_file))

    print("\nOutput files:")
    for p_str in all_outputs:
        print(f"  {p_str}")


if __name__ == "__main__":
    main()
