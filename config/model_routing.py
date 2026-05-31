"""Model routing configuration for agent modes.

Defines which LLM model to use for each agent mode at each quality level.
"""
from __future__ import annotations

from pathlib import Path

import yaml

_YAML_PATH = Path(__file__).resolve().parent / "models.yaml"

with _YAML_PATH.open() as _f:
    _cfg = yaml.safe_load(_f)

# mode_name -> {"fast": model_name, "thorough": model_name}
MODE_MODELS: dict[str, dict[str, str]] = _cfg["modes"]

SUMMARY_MODEL: str = _cfg.get("summary_model", "gpt-5-nano")
