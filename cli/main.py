"""CLI for the DialogVis two-stage data generation pipeline."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(
    name="dialogvis",
    help=(
        "DialogVis: two-stage multi-turn multimodal benchmark generation.\n\n"
        "Stage 1 (blueprint):  dialogvis blueprint ...\n"
        "Stage 2 (instantiate): dialogvis instantiate ...\n"
        "Both stages:          dialogvis run ..."
    ),
    add_completion=False,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _make_model_config(model, provider, api_base):
    from dialogvis.config import ModelConfig
    cfg = ModelConfig()
    if model:
        cfg.model = model
    if provider:
        cfg.provider = provider   # type: ignore[assignment]
    if api_base:
        cfg.api_base_url = api_base
    return cfg


# ---------------------------------------------------------------------------
# Shared options (reused across commands)
# ---------------------------------------------------------------------------

_model_opt    = typer.Option(None, "--model", "-m", help="Model name (overrides DIALOGVIS_MODEL)")
_provider_opt = typer.Option(None, "--provider", help="anthropic | openai | vllm")
_api_base_opt = typer.Option(None, "--api-base", help="API base URL (openai/vllm providers)")
_output_opt   = typer.Option(Path("outputs"), "--output-dir", "-o")
_concur_opt   = typer.Option(4, "--concurrency", "-c", min=1, max=32)
_batch_size_opt = typer.Option(10, "--batch-size", "-b", min=1, max=50, help="Blueprints to generate per LLM call")
_verbose_opt  = typer.Option(False, "--verbose", "-v")
_resume_opt   = typer.Option(True, "--resume/--no-resume", help="Skip already-completed items")


# ---------------------------------------------------------------------------
# `dialogvis blueprint` — Stage 1, single job
# ---------------------------------------------------------------------------

@app.command("blueprint")
def cmd_blueprint(
    taxonomy: str = typer.Option(..., "--taxonomy", "-t"),
    count: int = typer.Option(10, "--count", "-n", min=1, max=50, help="Number of diverse blueprints to generate"),
    hints: Optional[str] = typer.Option(None, "--hints", help="Extra instructions for the LLM"),
    output_dir: Path = _output_opt,
    model: Optional[str] = _model_opt,
    provider: Optional[str] = _provider_opt,
    api_base: Optional[str] = _api_base_opt,
    verbose: bool = _verbose_opt,
) -> None:
    """[Stage 1] Generate diverse blueprints for a taxonomy (LLM invents all scenarios)."""
    _setup_logging(verbose)
    from dialogvis.generator import BlueprintGenerator, GenerationError
    from dialogvis.io.writer import OutputWriter
    from dialogvis.models import GenerationJob

    taxonomy_enum = _parse_taxonomy(taxonomy)
    job = GenerationJob(taxonomy=taxonomy_enum, count=count, hints=hints)
    gen = BlueprintGenerator(_make_model_config(model, provider, api_base))
    writer = OutputWriter(output_dir)

    typer.echo(f"Generating {count} blueprints  [{taxonomy_enum.value}]")
    try:
        blueprints = gen.generate(job)
    except GenerationError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1)

    for bp in blueprints:
        writer.write_blueprint(bp)
    typer.echo(f"Saved {len(blueprints)} blueprints → {output_dir}/blueprints/")


# ---------------------------------------------------------------------------
# `dialogvis instantiate` — Stage 2, single blueprint
# ---------------------------------------------------------------------------

@app.command("instantiate")
def cmd_instantiate(
    blueprint_file: Path = typer.Argument(..., help="Path to a blueprint JSON file"),
    output_dir: Path = _output_opt,
    model: Optional[str] = _model_opt,
    provider: Optional[str] = _provider_opt,
    api_base: Optional[str] = _api_base_opt,
    verbose: bool = _verbose_opt,
) -> None:
    """[Stage 2] Instantiate a Blueprint JSON file into a full Conversation."""
    _setup_logging(verbose)
    from dialogvis.generator import DialogueInstantiator, GenerationError
    from dialogvis.io.writer import OutputWriter
    from dialogvis.models import Blueprint

    bp = Blueprint.model_validate_json(blueprint_file.read_text())
    inst = DialogueInstantiator(_make_model_config(model, provider, api_base))
    writer = OutputWriter(output_dir)

    typer.echo(f"Instantiating blueprint {bp.id}  [{bp.taxonomy.value}]")
    try:
        conv = inst.instantiate(bp)
    except GenerationError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1)

    writer.write_conversation(conv)
    typer.echo(f"Saved → {output_dir}/conversations/{conv.id}.json")
    typer.echo(f"  Turns: {len(conv.turns)}  Images: {conv.num_images}")


# ---------------------------------------------------------------------------
# `dialogvis run` — both stages, batch
# ---------------------------------------------------------------------------

@app.command("run")
def cmd_run(
    jobs_file: Path = typer.Argument(..., help="JSONL file of GenerationJob specs"),
    output_dir: Path = _output_opt,
    model: Optional[str] = _model_opt,
    provider: Optional[str] = _provider_opt,
    api_base: Optional[str] = _api_base_opt,
    concurrency: int = _concur_opt,
    resume: bool = _resume_opt,
    taxonomy: Optional[str] = typer.Option(None, "--taxonomy", "-t", help="Only process jobs with this taxonomy"),
    verbose: bool = _verbose_opt,
) -> None:
    """[Both stages] Generate blueprints then instantiate conversations from a job spec file."""
    _setup_logging(verbose)
    from dialogvis.config import PipelineConfig
    from dialogvis.pipeline import BatchPipeline

    _check_jobs_file(jobs_file)
    pipeline = BatchPipeline(
        _make_model_config(model, provider, api_base),
        PipelineConfig(output_dir=output_dir, concurrency=concurrency, resume=resume),
    )
    jobs = _filter_jobs(BatchPipeline.load_jobs(jobs_file), taxonomy)
    typer.echo(f"Loaded {len(jobs)} jobs. Running full pipeline...")

    bp_stats, conv_stats = pipeline.run_full(jobs)
    _print_stats("Blueprint stage", bp_stats)
    _print_stats("Conversation stage", conv_stats)
    if bp_stats.failed or conv_stats.failed:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# `dialogvis run-blueprints` — Stage 1 only, batch
# ---------------------------------------------------------------------------

@app.command("run-blueprints")
def cmd_run_blueprints(
    jobs_file: Path = typer.Argument(...),
    output_dir: Path = _output_opt,
    model: Optional[str] = _model_opt,
    provider: Optional[str] = _provider_opt,
    api_base: Optional[str] = _api_base_opt,
    concurrency: int = _concur_opt,
    batch_size: int = _batch_size_opt,
    resume: bool = _resume_opt,
    taxonomy: Optional[str] = typer.Option(None, "--taxonomy", "-t", help="Only process jobs with this taxonomy"),
    verbose: bool = _verbose_opt,
) -> None:
    """[Stage 1 only] Generate blueprints from a job spec file (no dialogue instantiation)."""
    _setup_logging(verbose)
    from dialogvis.config import PipelineConfig
    from dialogvis.pipeline import BatchPipeline

    _check_jobs_file(jobs_file)
    pipeline = BatchPipeline(
        _make_model_config(model, provider, api_base),
        PipelineConfig(output_dir=output_dir, concurrency=concurrency, resume=resume, blueprint_batch_size=batch_size),
    )
    jobs = _filter_jobs(BatchPipeline.load_jobs(jobs_file), taxonomy)
    typer.echo(f"Loaded {len(jobs)} jobs. Generating blueprints (batch_size={batch_size})...")
    stats = pipeline.run_blueprint_stage(jobs)
    _print_stats("Blueprint stage", stats)
    typer.echo(f"\nBlueprints saved to: {output_dir}/blueprints/")
    typer.echo("Review them, then run `dialogvis run-conversations` to instantiate.")
    if stats.failed:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# `dialogvis run-conversations` — Stage 2 only, batch
# ---------------------------------------------------------------------------

@app.command("run-conversations")
def cmd_run_conversations(
    output_dir: Path = _output_opt,
    model: Optional[str] = _model_opt,
    provider: Optional[str] = _provider_opt,
    api_base: Optional[str] = _api_base_opt,
    concurrency: int = _concur_opt,
    resume: bool = _resume_opt,
    taxonomy: Optional[str] = typer.Option(None, "--taxonomy", "-t", help="Only instantiate blueprints with this taxonomy"),
    verbose: bool = _verbose_opt,
) -> None:
    """[Stage 2 only] Instantiate all saved blueprints into conversations."""
    _setup_logging(verbose)
    from dialogvis.config import PipelineConfig
    from dialogvis.pipeline import BatchPipeline

    pipeline = BatchPipeline(
        _make_model_config(model, provider, api_base),
        PipelineConfig(output_dir=output_dir, concurrency=concurrency, resume=resume),
    )
    blueprints = pipeline._writer.load_blueprints()
    blueprints = _filter_blueprints(blueprints, taxonomy)
    stats = pipeline.run_instantiation_stage(blueprints)
    _print_stats("Conversation stage", stats)
    if stats.failed:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# `dialogvis validate`
# ---------------------------------------------------------------------------

@app.command("validate")
def cmd_validate(
    input_dir: Path = typer.Option(Path("outputs"), "--input-dir", "-i"),
    stage: str = typer.Option("both", "--stage", help="blueprints | conversations | both"),
    verbose: bool = _verbose_opt,
) -> None:
    """Validate saved blueprints and/or conversations."""
    _setup_logging(verbose)
    from dialogvis.models import Blueprint, Conversation

    errors = 0
    if stage in ("blueprints", "both"):
        errors += _validate_dir(input_dir / "blueprints", Blueprint, "Blueprint")
    if stage in ("conversations", "both"):
        errors += _validate_dir(input_dir / "conversations", Conversation, "Conversation")
    if errors:
        raise typer.Exit(1)


def _validate_dir(directory: Path, model_cls, label: str) -> int:
    if not directory.exists():
        typer.echo(f"No {label.lower()} directory: {directory}")
        return 0
    files = list(directory.glob("*.json"))
    typer.echo(f"Validating {len(files)} {label} files...")
    errors = 0
    for path in files:
        try:
            model_cls.model_validate_json(path.read_text())
        except Exception as exc:
            typer.echo(f"  INVALID {path.name}: {exc}", err=True)
            errors += 1
    ok = len(files) - errors
    typer.echo(f"  {ok} valid, {errors} invalid\n")
    return errors


# ---------------------------------------------------------------------------
# `dialogvis list-taxonomies`
# ---------------------------------------------------------------------------

@app.command("list-taxonomies")
def cmd_list_taxonomies() -> None:
    """List all available taxonomy types."""
    from dialogvis.models import TaxonomyType
    for t in TaxonomyType:
        typer.echo(f"  {t.value}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_taxonomy(value: str):
    from dialogvis.models import TaxonomyType
    try:
        return TaxonomyType(value)
    except ValueError:
        typer.echo(
            f"ERROR: Unknown taxonomy '{value}'.\nValid values:\n"
            + "\n".join(f"  {t.value}" for t in TaxonomyType),
            err=True,
        )
        raise typer.Exit(1)


def _filter_jobs(jobs: list, taxonomy: Optional[str]) -> list:
    if not taxonomy:
        return jobs
    from dialogvis.models import TaxonomyType
    t = _parse_taxonomy(taxonomy)
    filtered = [j for j in jobs if j.taxonomy == t]
    typer.echo(f"Filtered to {len(filtered)}/{len(jobs)} jobs with taxonomy '{t.value}'")
    return filtered


def _filter_blueprints(blueprints: list, taxonomy: Optional[str]) -> list:
    if not taxonomy:
        return blueprints
    t = _parse_taxonomy(taxonomy)
    filtered = [b for b in blueprints if b.taxonomy == t]
    typer.echo(f"Filtered to {len(filtered)}/{len(blueprints)} blueprints with taxonomy '{t.value}'")
    return filtered


def _check_jobs_file(path: Path) -> None:
    if not path.exists():
        typer.echo(f"ERROR: Jobs file not found: {path}", err=True)
        raise typer.Exit(1)


def _print_stats(label: str, stats) -> None:
    typer.echo(
        f"{label}: {stats.succeeded} succeeded, {stats.failed} failed, {stats.skipped} skipped"
    )
    if stats.failed_ids:
        typer.echo(f"  Failed IDs: {stats.failed_ids}", err=True)


if __name__ == "__main__":
    app()
