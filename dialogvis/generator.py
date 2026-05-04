"""
Two-stage generation pipeline:
  Stage 1 — BlueprintGenerator:   (taxonomy, count) → list[Blueprint]
  Stage 2 — DialogueInstantiator: Blueprint → Conversation
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from dialogvis.config import ModelConfig
from dialogvis.llm.client import LLMClient
from dialogvis.models import Blueprint, Conversation, GenerationJob
from dialogvis.taxonomy import build_batch_blueprint_prompt, DIALOGUE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_JSON_RETRY_HINT = (
    "IMPORTANT: Your previous response could not be parsed as valid JSON. "
    "Output ONLY a raw JSON array — no markdown fences, no commentary, "
    "no text before [ or after the closing ]."
)


class GenerationError(Exception):
    pass


# ===========================================================================
# Stage 1: Blueprint Generator
# One LLM call → list of N diverse blueprints
# ===========================================================================

class BlueprintGenerator:

    def __init__(self, model_config: ModelConfig):
        self._config = model_config
        self._client = LLMClient(model_config)

    def generate_batch(self, jobs: list[GenerationJob], batch_size: int | None = None) -> dict[str, list[Blueprint]]:
        """
        Generate blueprints for multiple jobs in a single batch request.
        
        Args:
            jobs: List of generation jobs
            batch_size: Number of blueprints to generate per LLM call (None = use job.count)
        
        Returns:
            Dict mapping job_id to list of blueprints
        """
        if not jobs:
            return {}
        
        results = {}
        
        # For each job, handle batch_size logic
        for job in jobs:
            if batch_size is None or batch_size >= job.count:
                # Single call per job - can batch these together
                pass
            else:
                # Multiple calls needed per job - process sequentially
                results[job.job_id] = self.generate(job, batch_size=batch_size)
                continue
        
        # Batch the single-call jobs
        single_call_jobs = [j for j in jobs if j.job_id not in results]
        if not single_call_jobs:
            return results
            
        # Build requests for jobs that need single calls
        requests = []
        for job in single_call_jobs:
            count = batch_size if batch_size else job.count
            prompt = build_batch_blueprint_prompt(job.taxonomy, count)
            user_content = self._build_user_content(prompt, job.hints)
            requests.append(("", user_content, ""))
        
        # Send batch request
        logger.info(f"Sending batch request for {len(single_call_jobs)} jobs")
        responses = self._client.complete_batch(requests)
        
        # Parse responses
        for job, raw in zip(single_call_jobs, responses):
            try:
                blueprints = self._parse_blueprints(raw, job)
                results[job.job_id] = blueprints
                logger.info(
                    "Generated %d blueprints for %s (job %s)",
                    len(blueprints), job.taxonomy.value, job.job_id,
                )
            except Exception as exc:
                logger.error(f"Failed to parse blueprints for job {job.job_id}: {exc}")
                results[job.job_id] = []
                
        return results

    def generate(self, job: GenerationJob, batch_size: int | None = None) -> list[Blueprint]:
        """
        Generate `job.count` diverse blueprints for the given taxonomy.
        
        Args:
            job: The generation job specification
            batch_size: If provided, generate this many blueprints per LLM call,
                       repeating until job.count is reached. This ensures diversity
                       within each batch while allowing larger total counts.
        """
        if batch_size is None or batch_size >= job.count:
            # Single LLM call for all blueprints
            return self._generate_single_batch(job, job.count)
        
        # Multiple LLM calls, each generating batch_size blueprints
        all_blueprints = []
        remaining = job.count
        batch_num = 0
        
        while remaining > 0:
            current_batch_size = min(batch_size, remaining)
            batch_num += 1
            
            logger.info(
                "Generating batch %d for job %s (%d blueprints, %d remaining)",
                batch_num, job.job_id, current_batch_size, remaining - current_batch_size
            )
            
            # Create a temporary job for this batch
            batch_job = GenerationJob(
                job_id=f"{job.job_id}_batch{batch_num}",
                taxonomy=job.taxonomy,
                count=current_batch_size,
                hints=job.hints
            )
            
            batch_blueprints = self._generate_single_batch(batch_job, current_batch_size)
            
            # Update metadata to reference the original job
            for bp in batch_blueprints:
                bp.metadata["original_job_id"] = job.job_id
                bp.metadata["batch_number"] = batch_num
            
            all_blueprints.extend(batch_blueprints)
            remaining -= len(batch_blueprints)
        
        logger.info(
            "Generated total %d blueprints for %s (job %s) across %d batches",
            len(all_blueprints), job.taxonomy.value, job.job_id, batch_num
        )
        return all_blueprints
    
    def _generate_single_batch(self, job: GenerationJob, count: int) -> list[Blueprint]:
        """Generate a single batch of blueprints with retry logic."""
        prompt = build_batch_blueprint_prompt(job.taxonomy, count)
        user_content = self._build_user_content(prompt, job.hints)

        last_exc: Exception | None = None
        last_raw: str = ""
        for attempt in range(1, self._config.max_parse_retries + 2):
            extra = _JSON_RETRY_HINT if attempt > 1 else ""
            try:
                # No separate system prompt — the full task description is the user message
                raw = self._client.complete("", user_content, extra_system=extra)
                last_raw = raw
                
                # DEBUG: Log response details
                logger.debug(f"Attempt {attempt}: Raw LLM response (first 500 chars): {raw[:500]}")
                logger.debug(f"Attempt {attempt}: Raw LLM response (last 200 chars): {raw[-200:]}")
                logger.debug(f"Attempt {attempt}: Raw LLM response length: {len(raw)} chars")
                
                # Check if it starts with [ or {
                stripped = raw.strip()
                if stripped:
                    logger.debug(f"Attempt {attempt}: Response starts with: '{stripped[0]}', ends with: '{stripped[-1]}'")
                    # Count occurrences of "id": to estimate number of blueprints
                    id_count = raw.count('"id":')
                    logger.debug(f"Attempt {attempt}: Number of '\"id\":' occurrences in response: {id_count}")
                
                blueprints = self._parse_blueprints(raw, job)
                logger.info(
                    "Generated %d blueprints for %s (job %s)",
                    len(blueprints), job.taxonomy.value, job.job_id,
                )
                return blueprints
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_exc = exc
                logger.warning(
                    "Blueprint parse attempt %d/%d failed for job %s: %s",
                    attempt, self._config.max_parse_retries + 1, job.job_id, exc,
                )
        
        # Save failed response for debugging
        if last_raw:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='outputs/failed') as f:
                f.write(f"Job: {job.job_id}\nCount requested: {count}\n\n")
                f.write(f"Last exception: {last_exc}\n\n")
                f.write("=" * 80 + "\n")
                f.write(last_raw)
                logger.error(f"Saved failed response to: {f.name}")

        raise GenerationError(
            f"Failed to generate blueprints for job {job.job_id} "
            f"after {self._config.max_parse_retries + 1} attempts"
        ) from last_exc

    def _build_user_content(self, prompt: str, hints: str | None) -> list[dict[str, Any]]:
        text = prompt
        if hints:
            text += f"\n\nAdditional instructions: {hints}"
        # The full prompt IS the user message (no separate system prompt for this stage)
        return [{"type": "text", "text": text}]

    def _parse_blueprints(self, raw: str, job: GenerationJob) -> list[Blueprint]:
        data = _extract_json_array(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")

        blueprints = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logger.warning("Skipping non-dict item at index %d", i)
                continue
            # Assign fresh UUIDs (LLM outputs integer ids)
            item["id"] = str(uuid.uuid4())
            item["taxonomy"] = job.taxonomy.value
            try:
                bp = Blueprint.model_validate(item)
                bp.metadata = {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "model": self._config.model,
                    "provider": self._config.provider,
                    "job_id": job.job_id,
                    "batch_index": i,
                }
                blueprints.append(bp)
            except ValidationError as exc:
                logger.warning("Skipping invalid blueprint at index %d: %s", i, exc)

        if not blueprints:
            raise ValueError("No valid blueprints could be parsed from LLM response")
        return blueprints


# ===========================================================================
# Stage 2: Dialogue Instantiator
# ===========================================================================

class DialogueInstantiator:

    def __init__(self, model_config: ModelConfig):
        self._config = model_config
        self._client = LLMClient(model_config)

    def instantiate(self, blueprint: Blueprint) -> Conversation:
        user_content = self._build_user_content(blueprint)
        last_exc: Exception | None = None

        for attempt in range(1, self._config.max_parse_retries + 2):
            extra = (
                "IMPORTANT: Your previous response could not be parsed as valid JSON. "
                "Output ONLY the raw JSON object." if attempt > 1 else ""
            )
            try:
                raw = self._client.complete(DIALOGUE_SYSTEM_PROMPT, user_content, extra_system=extra)
                conv = self._parse_conversation(raw, blueprint)
                return conv
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_exc = exc
                logger.warning(
                    "Dialogue parse attempt %d/%d failed for blueprint %s: %s",
                    attempt, self._config.max_parse_retries + 1, blueprint.id, exc,
                )

        raise GenerationError(
            f"Failed to instantiate dialogue for blueprint {blueprint.id} "
            f"after {self._config.max_parse_retries + 1} attempts"
        ) from last_exc

    def _build_user_content(self, blueprint: Blueprint) -> list[dict[str, Any]]:
        text = (
            "Instantiate the following blueprint into a multi-turn conversation.\n\n"
            "Blueprint:\n"
            + blueprint.model_dump_json(indent=2)
            + "\n\nOutput ONLY the raw JSON conversation object."
        )
        return [{"type": "text", "text": text}]

    def _parse_conversation(self, raw: str, blueprint: Blueprint) -> Conversation:
        data = _extract_json(raw)
        # Always link conversation id to its blueprint so resume + dedupe work.
        data["id"] = blueprint.id
        data["blueprint_id"] = blueprint.id
        data["taxonomy"] = blueprint.taxonomy.value
        conv = Conversation.model_validate(data)
        conv.metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": self._config.model,
            "provider": self._config.provider,
            "blueprint_id": blueprint.id,
        }
        return conv


# ===========================================================================
# JSON helpers
# ===========================================================================

def _extract_json_array(raw: str) -> list:
    """Extract a JSON array from LLM output, stripping markdown fences."""
    text = _strip_fences(raw)
    try:
        parsed = json.loads(text)
        # If it's already a list, return it
        if isinstance(parsed, list):
            return parsed
        # If it's a dict, try to extract the array from common wrapper keys
        if isinstance(parsed, dict):
            logger.debug(f"Got dict with keys: {list(parsed.keys())}")
            # Try common keys that might contain the blueprints array
            for key in ['blueprints', 'data', 'items', 'results', 'output']:
                if key in parsed and isinstance(parsed[key], list):
                    logger.info(f"Extracting array from dict key: {key}")
                    return parsed[key]
            # If dict has only one key with a list value, use that
            if len(parsed) == 1:
                value = next(iter(parsed.values()))
                if isinstance(value, list):
                    key_name = next(iter(parsed.keys()))
                    logger.info(f"Extracting array from single dict key: {key_name}")
                    return value
            # Check if this looks like a single blueprint (has expected blueprint keys)
            blueprint_keys = {'id', 'taxonomy', 'scenario', 'visual_sequence', 'ground_truth'}
            if blueprint_keys.issubset(set(parsed.keys())):
                logger.info("Detected single blueprint dict, wrapping in array")
                return [parsed]
            # Log the actual keys to help debug
            logger.warning(f"Dict keys don't match expected patterns. Keys: {list(parsed.keys())[:5]}")
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
    except json.JSONDecodeError as e:
        # Try extracting the outermost [...] block
        logger.debug(f"JSON decode error: {e}, trying regex extraction")
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in LLM response")
        return json.loads(match.group())


def _extract_json(raw: str) -> dict:
    """Extract a JSON object from LLM output."""
    text = _strip_fences(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response")
        return json.loads(match.group())


def _strip_fences(text: str) -> str:
    text = text.strip()
    # Remove markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Remove <think> tags (some models output reasoning in these)
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def _is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False
