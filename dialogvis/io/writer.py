"""Writes blueprints and conversations to disk."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from dialogvis.models import Blueprint, Conversation, GenerationJob

logger = logging.getLogger(__name__)


class OutputWriter:
    """
    Produces:
      output_dir/blueprints/<taxonomy>.jsonl — blueprints per taxonomy
      output_dir/blueprints.jsonl            — all blueprints
      output_dir/conversations/<taxonomy>.jsonl — conversations per taxonomy
      output_dir/conversations.jsonl         — all conversations
      output_dir/failed/<job_id>.txt         — raw LLM response on failure
    """

    def __init__(self, output_dir: Path):
        self._dir = output_dir
        self._bp_dir   = output_dir / "blueprints"
        self._conv_dir = output_dir / "conversations"
        self._failed_dir = output_dir / "failed"
        self._bp_jsonl   = output_dir / "blueprints.jsonl"
        self._conv_jsonl = output_dir / "conversations.jsonl"

        for d in (self._bp_dir, self._conv_dir, self._failed_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def completed_blueprint_ids(self) -> set[str]:
        return {p.stem for p in self._bp_dir.glob("*.json")}

    def completed_conversation_ids(self) -> set[str]:
        """Return IDs of all completed conversations from global JSONL."""
        if not self._conv_jsonl.exists():
            return set()
        ids = set()
        with self._conv_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    ids.add(data["id"])
                except Exception:
                    continue
        return ids

    # ------------------------------------------------------------------
    def write_blueprint(self, bp: Blueprint) -> None:
        path = self._bp_dir / f"{bp.id}.json"
        path.write_text(bp.model_dump_json(indent=2), encoding="utf-8")
        with self._bp_jsonl.open("a", encoding="utf-8") as f:
            f.write(bp.model_dump_json() + "\n")
        logger.info("Blueprint %s  [%s / %s]", bp.id, bp.taxonomy.value, bp.scenario)
    
    def write_blueprints_batch(self, blueprints: list[Blueprint], job_id: str) -> None:
        """Write multiple blueprints from a single job to taxonomy-based JSONL files."""
        if not blueprints:
            return
            
        # Group by taxonomy and write to taxonomy-specific JSONL files
        taxonomy = blueprints[0].taxonomy.value
        taxonomy_file = self._bp_dir / f"{taxonomy}.jsonl"
        
        with taxonomy_file.open("a", encoding="utf-8") as f:
            for bp in blueprints:
                f.write(bp.model_dump_json() + "\n")
        
        # Also write to global JSONL
        with self._bp_jsonl.open("a", encoding="utf-8") as f:
            for bp in blueprints:
                f.write(bp.model_dump_json() + "\n")
                logger.info("Blueprint %s  [%s / %s]", bp.id, bp.taxonomy.value, bp.scenario)
        
        logger.info("Wrote %d blueprints to %s", len(blueprints), taxonomy_file.name)

    def write_conversation(self, conv: Conversation) -> None:
        """Write conversation to taxonomy-specific JSONL and global JSONL."""
        # Write to taxonomy-specific JSONL
        taxonomy = conv.taxonomy.value
        taxonomy_file = self._conv_dir / f"{taxonomy}.jsonl"
        
        with taxonomy_file.open("a", encoding="utf-8") as f:
            f.write(conv.model_dump_json() + "\n")
        
        # Write to global JSONL
        with self._conv_jsonl.open("a", encoding="utf-8") as f:
            f.write(conv.model_dump_json() + "\n")
        
        logger.info(
            "Conversation %s  [%s / %s turns, %s images]",
            conv.id, conv.taxonomy.value, len(conv.turns), conv.num_images,
        )

    # ------------------------------------------------------------------
    def record_failure(self, job_id: str, label: str, exc: Exception, raw: str = "") -> None:
        path = self._failed_dir / f"{job_id}_{label}.txt"
        content = f"Job/Blueprint ID: {job_id}\nStage: {label}\nError: {exc}\n\n--- Raw Response ---\n{raw}\n"
        path.write_text(content, encoding="utf-8")
        logger.error("Failed [%s] %s: %s", label, job_id, exc)

    # ------------------------------------------------------------------
    def load_blueprints(self) -> list[Blueprint]:
        """Load all saved blueprints from the global JSONL file."""
        blueprints = []
        if not self._bp_jsonl.exists():
            logger.warning("No blueprints file found at %s", self._bp_jsonl)
            return blueprints
            
        with self._bp_jsonl.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    blueprints.append(Blueprint.model_validate_json(line))
                except Exception as exc:
                    logger.warning("Could not load blueprint at line %d: %s", i, exc)
        
        logger.info("Loaded %d blueprints from %s", len(blueprints), self._bp_jsonl.name)
        return blueprints
