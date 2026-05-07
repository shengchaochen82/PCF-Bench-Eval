"""Unified provider wrapper around ``tools.llm_client``.

Adds:
* round-robin multi-key rotation (one persistent client per key, no per-call
  construction overhead)
* simple sync ``query`` API consumed by legacy tasks
* optional async ``aquery`` for new-style task implementations
"""
from __future__ import annotations

import itertools
import os
import threading
from typing import Any, List, Optional

from pcfbench.config import EvalConfig

try:  # legacy import
    from tools.llm_client import get_llm_client
except ImportError as exc:
    raise RuntimeError(
        "Cannot import tools.llm_client; ensure tools/ is on PYTHONPATH."
    ) from exc


class ProviderClient:
    """Sync provider wrapper with multi-key round-robin.

    Resolves API keys from explicit ``api_keys`` argument, then from
    environment variables listed in ``EvalConfig.api_keys_env``, then from
    provider-default lookup in ``tools.env_utils.get_api_keys``.
    """

    def __init__(
        self,
        config: EvalConfig,
        api_keys: Optional[List[str]] = None,
    ) -> None:
        self.config = config
        self.model = config.model

        keys = self._resolve_keys(api_keys, config)
        if not keys:
            raise RuntimeError(
                f"No API keys for provider '{config.provider.name}'. "
                f"Set one of {config.api_keys_env or '<provider default>'}."
            )

        self._keys = keys
        self._clients = [
            get_llm_client(
                provider=config.provider.name,
                api_key=k,
                base_url=config.provider.base_url,
                timeout_s=config.provider.timeout_s,
            )
            for k in keys
        ]
        self._cycle = itertools.cycle(range(len(self._clients)))
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_keys(
        explicit: Optional[List[str]],
        config: EvalConfig,
    ) -> List[str]:
        if explicit:
            return [k for k in explicit if k]
        if config.api_keys_env:
            out: List[str] = []
            for var in config.api_keys_env:
                v = os.environ.get(var, "").strip()
                if v:
                    out.append(v)
            if out:
                return out
        # Fallback to provider-default lookup (env_utils.get_api_keys handles
        # SILICONFLOW_API_KEY_*, YUNWU_API_KEY_* etc.).
        from tools.env_utils import get_api_keys
        return list(get_api_keys(config.provider.name) or [])

    @property
    def n_keys(self) -> int:
        return len(self._clients)

    # ── sync API ──────────────────────────────────────────────────────────

    def _next_client(self) -> Any:
        with self._lock:
            idx = next(self._cycle)
        return self._clients[idx]

    def query(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        system_prompt: str = "You are an expert in photonic crystal fibers.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Single sync call with round-robin key rotation + retry."""
        import time

        T = temperature if temperature is not None else self.config.runner.temperature
        M = max_tokens if max_tokens is not None else self.config.runner.max_tokens
        retries = self.config.provider.max_retries

        last_err: Optional[BaseException] = None
        for attempt in range(retries + 1):
            client = self._next_client()
            try:
                if image_path:
                    raw = client.vision_chat(
                        model=self.model,
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        image_paths=[image_path],
                        temperature=T,
                        max_tokens=M,
                    )
                else:
                    raw = client.chat(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=T,
                        max_tokens=M,
                    )
                return _extract_text(raw)
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Provider call failed after {retries + 1} attempts: {last_err}")


def _extract_text(response: Any) -> str:
    """Best-effort string extraction from heterogeneous provider returns."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if content:
                return content
        message = response.get("message")
        if isinstance(message, dict):
            return message.get("content", str(response))
    return str(response)
