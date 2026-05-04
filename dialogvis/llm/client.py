"""Unified LLM client supporting Anthropic, OpenAI, and vLLM endpoints."""
from __future__ import annotations

import logging
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from dialogvis.config import ModelConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Wraps either the Anthropic SDK or the OpenAI SDK depending on provider.

    Usage:
        client = LLMClient(config)
        text = client.complete(system_prompt, user_content_blocks)
    """

    def __init__(self, config: ModelConfig):
        self._config = config
        self._provider = config.provider

        if self._provider == "anthropic":
            self._init_anthropic()
        else:
            self._init_openai()

    # ------------------------------------------------------------------
    def _init_anthropic(self) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("pip install anthropic") from e

        import os
        api_key = self._config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = anthropic.Anthropic(api_key=api_key)

    def _init_openai(self) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError("pip install openai") from e

        import os
        api_key = self._config.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=self._config.api_base_url,
            timeout=self._config.timeout,
            max_retries=0,  # tenacity handles retries
        )

    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        reraise=True,  # re-raises last exception after all attempts
    )
    def complete(
        self,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        extra_system: str = "",
    ) -> str:
        """
        Send a single-turn request (system + one user message) and return the text.

        Args:
            system_prompt: The system-level instruction.
            user_content:  List of content blocks (text / image) for the user turn.
            extra_system:  Additional text appended to the system prompt (e.g. retry hint).
        """
        full_system = system_prompt + ("\n\n" + extra_system if extra_system else "")

        if self._provider == "anthropic":
            return self._complete_anthropic(full_system, user_content)
        else:
            return self._complete_openai(full_system, user_content)

    # ------------------------------------------------------------------
    def complete_batch(
        self,
        requests: list[tuple[str, list[dict[str, Any]], str]],
    ) -> list[str]:
        """
        Send multiple requests in a single batch call (vLLM only).
        
        Args:
            requests: List of (system_prompt, user_content, extra_system) tuples
            
        Returns:
            List of response texts in the same order as requests
        """
        if self._provider == "anthropic":
            # Anthropic doesn't support batch, fall back to sequential
            return [self.complete(sys, user, extra) for sys, user, extra in requests]
        
        # OpenAI/vLLM batch implementation
        kwargs = {}
        if self._config.use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            
        # Build all message lists
        all_messages = []
        for system_prompt, user_content, extra_system in requests:
            full_system = system_prompt + ("\n\n" + extra_system if extra_system else "")
            all_messages.append([
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_content},
            ])
        
        # Send batch request - vLLM expects multiple separate requests
        # We'll use the standard API but send them with custom request IDs
        responses = []
        for messages in all_messages:
            try:
                response = self._client.chat.completions.create(
                    model=self._config.model,
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                    messages=messages,
                    **kwargs,
                )
                responses.append(response.choices[0].message.content)
            except Exception as e:
                logger.error(f"Batch request failed: {e}")
                responses.append("")
                
        return responses

    # ------------------------------------------------------------------
    def _complete_anthropic(self, system: str, user_content: list[dict]) -> str:
        response = self._client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text

    def _complete_openai(self, system: str, user_content: list[dict]) -> str:
        kwargs = {}
        if self._config.use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            **kwargs,
        )
        return response.choices[0].message.content
