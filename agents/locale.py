"""Locale-aware prompt loader.

Usage in mode modules:
    from agents.locale import load_prompts
    prompts = load_prompts("agents.retrieve.languages")
    # then use prompts.QUERY_INSTRUCTIONS, prompts.build_query_prompt(), etc.
"""
from __future__ import annotations

import importlib
from types import ModuleType

from config.app_settings import app_settings

_cache: dict[str, ModuleType] = {}


def load_prompts(module_path: str) -> ModuleType:
    """Load the language-specific prompts module for the configured APP_LANGUAGE.

    Args:
        module_path: Dotted path to the languages package, e.g. "agents.retrieve.languages"

    Returns:
        The language module (e.g. agents.retrieve.languages.nb)
    """
    lang = app_settings.language
    key = f"{module_path}.{lang}"
    if key not in _cache:
        try:
            _cache[key] = importlib.import_module(key)
        except ModuleNotFoundError:
            raise ValueError(
                f"No prompts module found for language '{lang}' at {key}. "
                f"Create {key.replace('.', '/')}.py to add support."
            )
    return _cache[key]
