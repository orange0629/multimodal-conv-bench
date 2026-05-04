"""
Two-stage batch pipeline:
  Stage 1 — generate blueprints from job specs
  Stage 2 — instantiate blueprints into conversations

Stages can be run independently, allowing human review between them.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from dialogvis.config import ModelConfig, PipelineConfig
from dialogvis.generator import BlueprintGenerator, DialogueInstantiator, GenerationError
from dialogvis.io.writer import OutputWriter
from dialogvis.models import Blueprint, GenerationJob

logger = logging.getLogger(__name__)


@dataclass
class StageStats:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    failed_ids: list[str] = field(default_factory=list)


class BatchPipeline:

    def __init__(self, model_config: ModelConfig, pipeline_config: PipelineConfig):
        self._model_cfg = model_config
        self._pipeline_cfg = pipeline_config
        self._writer = OutputWriter(pipeline_config.output_dir)
        self._bp_generator = BlueprintGenerator(model_config)
        self._instantiator = DialogueInstantiator(model_config)

    # ------------------------------------------------------------------
    # Stage 1: Blueprint generation
    # ------------------------------------------------------------------

    def run_blueprint_stage(self, jobs: list[GenerationJob]) -> StageStats:
        """Generate blueprints from job specs using batch inference."""
        stats = StageStats(total=len(jobs))
        completed = self._writer.completed_blueprint_ids() if self._pipeline_cfg.resume else set()

        pending = [j for j in jobs if j.job_id not in completed]
        stats.skipped = len(jobs) - len(pending)

        if not pending:
            logger.info("All %d blueprint jobs already completed.", stats.total)
            return stats

        batch_size = self._pipeline_cfg.blueprint_batch_size
        logger.info(
            "Generating blueprints using batch inference (batch_size=%d per LLM call, concurrency=%d)...",
            batch_size, self._pipeline_cfg.concurrency
        )
        
        # Process in batches
        concurrency = self._pipeline_cfg.concurrency
        for i in range(0, len(pending), concurrency):
            batch = pending[i:i + concurrency]
            logger.info(
                f"Processing batch {i//concurrency + 1}/{(len(pending)-1)//concurrency + 1} ({len(batch)} jobs)"
            )
            
            try:
                # Use batch generation with batch_size parameter
                results = self._bp_generator.generate_batch(batch, batch_size=batch_size)
                
                # Write all generated blueprints
                for job in batch:
                    blueprints = results.get(job.job_id, [])
                    if blueprints:
                        self._writer.write_blueprints_batch(blueprints, job.job_id)
                        stats.succeeded += 1
                    else:
                        stats.failed += 1
                        stats.failed_ids.append(job.job_id)
                        logger.error(f"No blueprints generated for job {job.job_id}")
                        
            except Exception as exc:
                logger.error(f"Batch processing failed: {exc}")
                for job in batch:
                    stats.failed += 1
                    stats.failed_ids.append(job.job_id)
                    
        return stats

    def _generate_one_blueprint(self, job: GenerationJob) -> None:
        try:
            blueprints = self._bp_generator.generate(job)
            for bp in blueprints:
                self._writer.write_blueprint(bp)
        except GenerationError as exc:
            self._writer.record_failure(job.job_id, "blueprint", exc)
            raise

    # ------------------------------------------------------------------
    # Stage 2: Dialogue instantiation
    # ------------------------------------------------------------------

    def run_instantiation_stage(self, blueprints: list[Blueprint] | None = None) -> StageStats:
        """Instantiate blueprints into conversations.

        If blueprints is None, loads all saved blueprints from the output directory.
        """
        if blueprints is None:
            blueprints = self._writer.load_blueprints()

        stats = StageStats(total=len(blueprints))
        completed = self._writer.completed_conversation_ids() if self._pipeline_cfg.resume else set()

        pending = [bp for bp in blueprints if bp.id not in completed]
        stats.skipped = len(blueprints) - len(pending)

        if not pending:
            logger.info("All %d instantiation jobs already completed.", stats.total)
            return stats

        logger.info("Instantiating %d blueprints (concurrency=%d)...", len(pending), self._pipeline_cfg.concurrency)
        self._run_parallel(
            items=pending,
            fn=self._instantiate_one,
            stats=stats,
            desc="Conversations",
        )
        return stats

    def _instantiate_one(self, blueprint: Blueprint) -> None:
        try:
            conv = self._instantiator.instantiate(blueprint)
            self._writer.write_conversation(conv)
        except GenerationError as exc:
            self._writer.record_failure(blueprint.id, "instantiation", exc)
            raise

    # ------------------------------------------------------------------
    # Full pipeline (both stages)
    # ------------------------------------------------------------------

    def run_full(self, jobs: list[GenerationJob]) -> tuple[StageStats, StageStats]:
        """Run both stages end-to-end."""
        bp_stats = self.run_blueprint_stage(jobs)
        blueprints = self._writer.load_blueprints()
        conv_stats = self.run_instantiation_stage(blueprints)
        return bp_stats, conv_stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_parallel(self, items: list, fn, stats: StageStats, desc: str) -> None:
        with ThreadPoolExecutor(max_workers=self._pipeline_cfg.concurrency) as pool:
            future_to_item = {pool.submit(fn, item): item for item in items}
            with tqdm(total=len(items), desc=desc, unit="item") as pbar:
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    item_id = getattr(item, "job_id", None) or getattr(item, "id", "?")
                    try:
                        future.result()
                        stats.succeeded += 1
                    except Exception as exc:
                        stats.failed += 1
                        stats.failed_ids.append(item_id)
                        logger.error("%s %s failed: %s", desc, item_id, exc)
                    finally:
                        pbar.update(1)

    # ------------------------------------------------------------------
    @staticmethod
    def load_jobs(jobs_file: Path) -> list[GenerationJob]:
        import json
        jobs = []
        with jobs_file.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    jobs.append(GenerationJob.model_validate(json.loads(line)))
                except Exception as exc:
                    logger.warning("Skipping malformed job at line %d: %s", i + 1, exc)
        return jobs
