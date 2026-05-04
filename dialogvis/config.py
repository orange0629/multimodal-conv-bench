from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DIALOGVIS_", env_file=".env", extra="ignore")

    provider: Literal["anthropic", "openai", "vllm"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    api_base_url: str = "https://api.openai.com/v1"  # used for openai/vllm providers
    api_key: str = ""                                  # reads DIALOGVIS_API_KEY or provider key
    temperature: float = 0.8
    max_tokens: int = 4096
    timeout: float = 180.0
    max_retries: int = 3
    max_parse_retries: int = 2   # extra retries on JSON parse failure
    use_json_mode: bool = True   # set False if vLLM rejects response_format=json_object


class PipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DIALOGVIS_", env_file=".env", extra="ignore")

    output_dir: Path = Path("outputs")
    concurrency: int = 4
    resume: bool = True
    blueprint_batch_size: int = 10  # number of blueprints to generate per LLM call
