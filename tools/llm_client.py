"""
Unified LLM Client supporting multiple providers:
- SiliconFlow
- Zhipu AI (智谱AI)
- OpenRouter

Usage:
    from tools.llm_client import get_llm_client
    
    client = get_llm_client(
        provider="zhipuai",
        api_key="your_api_key",
        base_url=None  # Optional, uses default
    )
    
    response = client.chat(
        model="glm-4-plus",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7
    )
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
    """Base class for LLM clients."""
    
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


class SiliconFlowClient(BaseLLMClient):
    """SiliconFlow API client."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.siliconflow.cn/v1",
        timeout_s: int = 120,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("SILICONFLOW_API_KEY")
        if not api_key:
            raise RuntimeError("Missing SILICONFLOW_API_KEY")
        
        super().__init__(api_key, base_url, timeout_s)
        self.chat_path = "/chat/completions"
    
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
        payload = {
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
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for p in image_paths:
            content.append({"type": "image_url", "image_url": {"url": _b64_data_url(p)}})
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        response = self.chat(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return response["choices"][0]["message"]["content"]


class ZhipuAIClient(BaseLLMClient):
    """Zhipu AI (智谱AI) API client.
    
    Docs: https://open.bigmodel.cn/dev/api
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        timeout_s: int = 120,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("ZHIPUAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing ZHIPUAI_API_KEY")
        
        super().__init__(api_key, base_url, timeout_s)
        self.chat_path = "/chat/completions"
    
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
        payload = {
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
        """Zhipu AI vision API uses same format as OpenAI."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for p in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": _b64_data_url(p)}
            })
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        response = self.chat(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return response["choices"][0]["message"]["content"]


class OpenRouterClient(BaseLLMClient):
    """OpenRouter API client.
    
    Docs: https://openrouter.ai/docs
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
        self.app_name = app_name or os.environ.get("OPENROUTER_APP_NAME", "PCF-Grounding")
    
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
        
        payload = {
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
        max_tokens: Optional[int] = None    
    ) -> str:
        """OpenRouter vision API uses OpenAI-compatible format."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for p in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": _b64_data_url(p)}
            })
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        response = self.chat(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return response["choices"][0]["message"]["content"]


class YunwuClient(BaseLLMClient):
    """Yunwu.ai API client (OpenAI-compatible aggregator).

    Yunwu.ai provides an OpenAI-compatible endpoint that proxies GPT, Claude,
    Gemini, Qwen, and other models under a unified API.

    Docs: https://yunwu.ai
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://yunwu.ai/v1",
        timeout_s: int = 120,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("YUNWU_API_KEY") or os.environ.get("YUNWU_API_KEY_1")
        if not api_key:
            raise RuntimeError("Missing YUNWU_API_KEY (or YUNWU_API_KEY_1)")

        super().__init__(api_key, base_url, timeout_s)
        self.chat_path = "/chat/completions"

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
        """Yunwu.ai uses the same OpenAI-compatible vision format."""
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
    provider: str = "siliconflow",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_s: int = 120,
    **kwargs: Any,
) -> BaseLLMClient:
    """
    Factory function to get LLM client for specified provider.

    Args:
        provider: One of "siliconflow", "zhipuai", "openrouter", "yunwu"
        api_key: API key (if None, reads from environment)
        base_url: Base URL (if None, uses default)
        timeout_s: Request timeout in seconds
        **kwargs: Additional provider-specific arguments

    Returns:
        LLM client instance

    Example:
        >>> client = get_llm_client("yunwu", api_key="sk-...")
        >>> response = client.chat("gpt-4o-mini", messages=[...])
    """
    provider = provider.lower()

    if provider == "siliconflow":
        return SiliconFlowClient(
            api_key=api_key,
            base_url=base_url or "https://api.siliconflow.cn/v1",
            timeout_s=timeout_s,
        )
    elif provider == "zhipuai":
        return ZhipuAIClient(
            api_key=api_key,
            base_url=base_url or "https://open.bigmodel.cn/api/paas/v4",
            timeout_s=timeout_s,
        )
    elif provider == "openrouter":
        return OpenRouterClient(
            api_key=api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
            timeout_s=timeout_s,
            site_url=kwargs.get("site_url"),
            app_name=kwargs.get("app_name"),
        )
    elif provider == "yunwu":
        return YunwuClient(
            api_key=api_key,
            base_url=base_url or "https://yunwu.ai/v1",
            timeout_s=timeout_s,
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            "Choose from: siliconflow, zhipuai, openrouter, yunwu"
        )
