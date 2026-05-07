"""Environment configuration utilities.

Provides centralized .env file loading with fallback to environment variables.
Supports multi-key API key pools for high-throughput parallel processing.

Usage:
    from tools.env_utils import get_api_key, get_api_keys, get_config

    # Single key
    api_key = get_api_key("SILICONFLOW_API_KEY")

    # Multi-key pool
    api_keys = get_api_keys("SILICONFLOW")
    
    # Config value with default
    workers = get_config("DEFAULT_NUM_WORKERS", default=4, as_int=True)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _find_env_file() -> Optional[Path]:
    """Find .env file starting from current directory and walking up."""
    # Try current directory first
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        return env_path
    
    # Walk up directory tree
    current = Path.cwd()
    for _ in range(5):  # Max 5 levels up
        env_path = current / ".env"
        if env_path.exists():
            return env_path
        parent = current.parent
        if parent == current:
            break
        current = parent
    
    return None


def _load_env_file(env_path: Optional[Path] = None) -> Dict[str, str]:
    """Load .env file into dictionary."""
    if env_path is None:
        env_path = _find_env_file()
    
    if env_path is None or not env_path.exists():
        return {}
    
    env_vars = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            
            # Parse KEY=VALUE
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                env_vars[key] = value
    
    return env_vars


# Cache loaded env vars
_ENV_CACHE: Optional[Dict[str, str]] = None


def _get_env_vars() -> Dict[str, str]:
    """Get cached env vars (from .env file + os.environ)."""
    global _ENV_CACHE
    
    if _ENV_CACHE is None:
        # Load from .env file
        file_vars = _load_env_file()
        
        # Merge with os.environ (os.environ takes precedence)
        _ENV_CACHE = {**file_vars, **dict(os.environ)}
    
    return _ENV_CACHE


def reload_env() -> None:
    """Reload environment from .env file."""
    global _ENV_CACHE
    _ENV_CACHE = None
    _get_env_vars()


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with .env file support.
    
    Args:
        key: Environment variable name
        default: Default value if not found
    
    Returns:
        Value from .env file or os.environ, or default
    """
    return _get_env_vars().get(key, default)


def get_api_key(provider: str, key_name: Optional[str] = None) -> Optional[str]:
    """Get API key for a provider.
    
    Looks for keys in this order:
    1. Explicit key_name argument
    2. {PROVIDER}_API_KEY (e.g., SILICONFLOW_API_KEY)
    3. {PROVIDER}_API_KEY_1 (first numbered key as fallback)
    
    Args:
        provider: Provider name (siliconflow, openrouter, zhipuai, openai)
        key_name: Explicit key name to look for
    
    Returns:
        API key or None
    """
    provider_upper = provider.upper().replace("-", "_")
    
    # Try explicit key name first
    if key_name:
        key = get_env(key_name)
        if key and not key.startswith("your_"):
            return key
    
    # Try PROVIDER_API_KEY
    key = get_env(f"{provider_upper}_API_KEY")
    if key and not key.startswith("your_"):
        return key
    
    # Fallback: try first numbered key
    key = get_env(f"{provider_upper}_API_KEY_1")
    if key and not key.startswith("your_"):
        return key
    
    return None


def get_api_keys(provider: str) -> List[str]:
    """Get multiple API keys for a provider (for round-robin load balancing).
    
    Looks for keys in this order:
    1. {PROVIDER}_API_KEYS (comma-separated)
    2. {PROVIDER}_API_KEY_1, {PROVIDER}_API_KEY_2, etc.
    3. {PROVIDER}_API_KEY (single key)
    
    Args:
        provider: Provider name
    
    Returns:
        List of API keys (may be empty)
    """
    provider_upper = provider.upper().replace("-", "_")
    keys: List[str] = []
    
    # Try comma-separated list
    keys_str = get_env(f"{provider_upper}_API_KEYS")
    if keys_str and not keys_str.startswith("your_"):
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        keys = [k for k in keys if not k.startswith("your_")]
        if keys:
            return keys
    
    # Try numbered keys (KEY_1, KEY_2, ...)
    for i in range(1, 20):
        key = get_env(f"{provider_upper}_API_KEY_{i}")
        if key and not key.startswith("your_"):
            keys.append(key)
    
    if keys:
        return keys
    
    # Fall back to single key
    single_key = get_api_key(provider)
    if single_key:
        return [single_key]
    
    return []


def get_config(
    key: str,
    default: Any = None,
    as_int: bool = False,
    as_float: bool = False,
    as_bool: bool = False,
) -> Any:
    """Get configuration value with type conversion.
    
    Args:
        key: Configuration key
        default: Default value
        as_int: Convert to integer
        as_float: Convert to float
        as_bool: Convert to boolean
    
    Returns:
        Configuration value with appropriate type
    """
    value = get_env(key, default)
    
    if value is None:
        return default
    
    if as_bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)
    
    if as_int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    if as_float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    return value


def get_rpm_limit(provider: str) -> int:
    """Get rate limit (requests per minute) for a provider.
    
    Args:
        provider: Provider name
    
    Returns:
        RPM limit (default 60)
    """
    provider_upper = provider.upper().replace("-", "_")
    
    # Try provider-specific limit
    rpm = get_config(f"{provider_upper}_RPM_LIMIT", default=None, as_int=True)
    if rpm:
        return rpm
    
    # Fall back to default
    return get_config("DEFAULT_RPM_LIMIT", default=60, as_int=True)


def get_num_workers(provider: Optional[str] = None) -> int:
    """Get recommended number of parallel workers.
    
    Args:
        provider: Optional provider name for provider-specific settings
    
    Returns:
        Number of workers
    """
    if provider:
        keys = get_api_keys(provider)
        if len(keys) > 1:
            # With multiple keys, use more workers
            return min(len(keys) * 2, 10)
    
    return get_config("DEFAULT_NUM_WORKERS", default=4, as_int=True)


def get_cpu_workers() -> int:
    """Get number of workers for CPU-bound tasks."""
    return get_config("DEFAULT_CPU_WORKERS", default=8, as_int=True)


def mask_api_key(key: str, show_length: int = 8) -> str:
    """Mask API key for safe logging.
    
    Args:
        key: API key
        show_length: Number of characters to show
    
    Returns:
        Masked key (e.g., "sk-xxxx...xxxx")
    """
    if not key:
        return "(none)"
    
    if len(key) <= show_length * 2:
        return key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
    
    return f"{key[:show_length]}...{key[-show_length:]}"


def print_env_status() -> None:
    """Print status of loaded API keys."""
    from rich.console import Console
    console = Console()
    
    console.print("\n[bold]Environment Configuration[/bold]")
    console.print("=" * 50)
    
    providers = ["yunwu", "siliconflow", "zhipuai", "openrouter", "openai"]
    
    for provider in providers:
        keys = get_api_keys(provider)
        if keys:
            masked = [mask_api_key(k) for k in keys]
            key_str = ", ".join(masked)
            console.print(f"  {provider.upper()}: {key_str} ({len(keys)} key(s))")
        else:
            console.print(f"  {provider.upper()}: [dim](not configured)[/dim]")
    
    # Show other config
    workers = get_num_workers()
    rpm = get_config("DEFAULT_RPM_LIMIT", default=60)
    console.print(f"\n  DEFAULT_NUM_WORKERS: {workers}")
    console.print(f"  DEFAULT_RPM_LIMIT: {rpm}")
    console.print("")


if __name__ == "__main__":
    # Test the module
    print_env_status()
