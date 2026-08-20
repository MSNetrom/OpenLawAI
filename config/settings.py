from __future__ import annotations

import os
import warnings
from pathlib import Path
from django.core.exceptions import ImproperlyConfigured

warnings.filterwarnings("ignore", message="StreamingHttpResponse must consume synchronous iterators")


def _require_env(name: str) -> str:
    """Require an environment variable, crash if missing."""
    value = os.environ.get(name)
    if not value:
        raise ImproperlyConfigured(f"{name} environment variable must be set")
    return value


def _require_env_list(name: str) -> list[str]:
    """Require a comma-separated environment variable, crash if missing."""
    raw = os.environ.get(name, "")
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ImproperlyConfigured(f"{name} environment variable must be set (comma-separated list)")
    return values


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        stripped = value.strip()
        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
            stripped = stripped[1:-1]
        os.environ.setdefault(key.strip(), stripped)


BASE_DIR = Path(__file__).resolve().parent.parent
_load_env_file(BASE_DIR / ".env")


# SECURITY: Default DEBUG to False (safe default for production)
DEBUG = os.environ.get("DJANGO_DEBUG", "false").lower() in {"1", "true", "yes"}

# SECURITY: Require SECRET_KEY in production
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY")
if not SECRET_KEY:
    if DEBUG:
        SECRET_KEY = "dev-secret-key-not-for-production"
    else:
        raise ImproperlyConfigured(
            "DJANGO_SECRET_KEY environment variable must be set in production. "
            "Generate one with: python -c \"from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())\""
        )
ALLOWED_HOSTS: list[str] = _require_env_list("DJANGO_ALLOWED_HOSTS")
CSRF_TRUSTED_ORIGINS: list[str] = _require_env_list("DJANGO_CSRF_TRUSTED_ORIGINS")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "adrf",
    "legaldb",
    "rest_framework",
    "chatdb",
    "accounts",
    "usage",
]

MIDDLEWARE = [
    "config.middleware.AsyncDBConnectionMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "config.middleware.AsyncWhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]


DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

DJANGO_LOG_LEVEL = os.environ.get("DJANGO_LOG_LEVEL", "INFO").upper()

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": DJANGO_LOG_LEVEL,
    },
    "loggers": {
        "chat_manager": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "chatdb": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "chatdb.views": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
}

# Session security
SESSION_COOKIE_SECURE = not DEBUG
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_AGE = 60 * 60 * 24 * 14  # 14 days
CSRF_COOKIE_SECURE = not DEBUG

# Allow 100MB uploads plus multipart overhead; reject larger bodies before the view reads them.
DATA_UPLOAD_MAX_MEMORY_SIZE = 110 * 1024 * 1024

# Redis configuration (used for chat streaming, locks, and caching)
REDIS_URL = _require_env("REDIS_URL")
REDIS_KEY_PREFIX = os.environ.get("REDIS_KEY_PREFIX", "openlawai")


# Trust X-Forwarded-Proto when running behind a reverse proxy
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
