
"""
LLMClient: Swappable wrapper for Anthropic, Google Gemini and Groq.
Reads LLM_PROVIDER, LLM_MODEL and API_KEY from settings.
"""

from __future__ import annotations

from src.config.settings import settings
from src.observability.logger import logger


class LLMConfigurationError(Exception):
    pass


class LLMClient:

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.provider = (provider or settings.llm_provider).lower()
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.api_key

        if not self.api_key:
            raise LLMConfigurationError(
                "No API_KEY configured. Set API_KEY in your .env file."
            )

        if self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self.api_key
            )

        elif self.provider in ("gemini", "google"):
            from google import genai

            self._client = genai.Client(
                api_key=self.api_key
            )

        elif self.provider == "groq":
            from groq import Groq

            self._client = Groq(
                api_key=self.api_key
            )

        else:
            raise LLMConfigurationError(
                f"Unsupported provider: {self.provider}. "
                "Supported providers: anthropic, gemini, groq."
            )

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:

        max_tokens = (
            max_tokens
            if max_tokens is not None
            else settings.llm_max_tokens
        )

        temperature = (
            temperature
            if temperature is not None
            else settings.llm_temperature
        )

        try:

            if self.provider == "anthropic":

                kwargs = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                }

                if system:
                    kwargs["system"] = system

                response = self._client.messages.create(
                    **kwargs
                )

                text_blocks = [
                    block.text
                    for block in response.content
                    if getattr(block, "type", "") == "text"
                ]

                return "".join(text_blocks).strip()

            elif self.provider in ("gemini", "google"):

                from google.genai import types

                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=system if system else None,
                )

                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )

                return (response.text or "").strip()

            elif self.provider == "groq":

                messages = []

                if system:
                    messages.append({
                        "role": "system",
                        "content": system,
                    })

                messages.append({
                    "role": "user",
                    "content": prompt,
                })

                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                return (
                    response.choices[0].message.content or ""
                ).strip()

        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise
