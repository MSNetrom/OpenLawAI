#!/usr/bin/env python
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

base = Path(__file__).parent
env_file = base / ".env"
if env_file.exists():
    load_dotenv(env_file)


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError("Couldn't import Django. Are you sure it's installed and available on your PYTHONPATH?") from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
