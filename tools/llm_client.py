"""OpenRouter LLM client.

Usage::

    from tools.llm_client import get_llm_client

    client = get_llm_client(
        provider="openrouter",
        api_key="sk-or-...",
    )
    response = client.chat(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.1,
    )

OpenRouter is the only provider supported in the public release. Its
OpenAI-compatible API exposes 200+ frontier and open-weight VLMs
(GPT, Claude, Gemini, Qwen-VL, Llama-Vision, ...), which is why
PCFBench can evaluate the same 41-model cohort through a single
provider without per-vendor SDK plumbing.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from typing import Any, Dict, List, Optional

import requests


def _b64_data_url(path: str) -> str:
    """Convert image file to base64 data URL."""
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


class BaseLLMClient:
    """Base class for OpenAI-compatible chat clients."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: int = 120,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_s = timeout_s

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Send chat completion request."""
        raise NotImplementedError

    def vision_chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: List[str],
        temperature: float = 0.7,
    ) -> str:
        """Send vision chat request with images."""
        raise NotImplementedError


class OpenRouterClient(BaseLLMClient):
    """OpenRouter API client (https://openrouter.ai/docs).

    Routes calls to 200+ frontier and open-weight chat / vision models
    behind a single OpenAI-compatible endpoint. The constructor reads
    ``OPENROUTER_API_KEY`` from the environment when ``api_key`` is not
    explicitly supplied.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: int = 120,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY")

        super().__init__(api_key, base_url, timeout_s)
        self.chat_path = "/chat/completions"
        self.site_url = site_url or os.environ.get("OPENROUTER_SITE_URL", "")
        self.app_name = app_name or os.environ.get("OPENROUTER_APP_NAME", "PCFBench")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Any:
        url = f"{self.base_url}{self.chat_path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def vision_chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: List[str],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """OpenRouter vision uses OpenAI-compatible content arrays."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for p in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": _b64_data_url(p)},
            })
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        response = self.chat(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"]


def get_llm_client(
    provider: str = "openrouter",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_s: int = 120,
    **kwargs: Any,
) -> BaseLLMClient:
    """Factory returning an :class:`OpenRouterClient`.

    The ``provider`` argument is accepted for forward compatibility but
    must equal ``"openrouter"`` in the public release; passing any other
    value raises :class:`ValueError`.
    """
    if provider.lower() != "openrouter":
        raise ValueError(
            f"Unsupported provider {provider!r}. The public release only "
            "supports 'openrouter'; route your model of choice through it."
        )
    return OpenRouterClient(
        api_key=api_key,
        base_url=base_url or "https://openrouter.ai/api/v1",
        timeout_s=timeout_s,
        site_url=kwargs.get("site_url"),
        app_name=kwargs.get("app_name"),
    )
