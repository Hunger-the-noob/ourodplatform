from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """极简 .env 读取（KEY=VALUE），避免为一个配置项引入额外依赖。

    已存在的环境变量优先，方便命令行临时覆盖。
    """
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(BASE_DIR / ".env")

SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY",
    "dev-only-odplatform-secret-key-change-before-production",
)
DEBUG = os.getenv("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "defect",
    "assistant",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "corsheaders.middleware.CorsMiddleware",
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
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "UserAttributeSimilarityValidator"
        )
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "MinimumLengthValidator"
        )
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "CommonPasswordValidator"
        )
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "NumericPasswordValidator"
        )
    },
]

LANGUAGE_CODE = "zh-hans"
TIME_ZONE = "Asia/Shanghai"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "frontend" / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

CORS_ALLOW_ALL_ORIGINS = DEBUG
CORS_ALLOWED_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]
CSRF_TRUSTED_ORIGINS = CORS_ALLOWED_ORIGINS

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.MultiPartParser",
        "rest_framework.parsers.FormParser",
    ],
}

DATA_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024

# =============================================================================
# 智能助手 / 视频分析（LLM + 本地 YOLO）
# 所有键都可用环境变量或 apps/web-backend/.env 覆盖，样例见 .env.example
# =============================================================================
ASSISTANT = {
    "llm": {
        "provider": os.getenv("LLM_PROVIDER", "DeepSeek"),
        "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        "api_key": (
            os.getenv("LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY", "")
        ),
        "model": os.getenv("LLM_MODEL", "deepseek-chat"),
        "timeout": float(os.getenv("LLM_TIMEOUT", "120")),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.3")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1600")),
    },
    "video": {
        # 跑 YOLO 的 Python 解释器；留空则自动探测（当前进程 → conda 环境 → 常见安装位置）
        "yolo_python": os.getenv("ODP_YOLO_PYTHON", ""),
        # 指定权重；留空则按 自训练产物 → 预训练权重 → fallback 自动选择
        "model": os.getenv("ODP_VIDEO_MODEL", ""),
        "fallback_model": os.getenv("ODP_VIDEO_FALLBACK_MODEL", "yolo11n.pt"),
        "device": os.getenv("ODP_VIDEO_DEVICE", "cpu"),
        "conf": float(os.getenv("ODP_VIDEO_CONF", "0.25")),
        "iou": float(os.getenv("ODP_VIDEO_IOU", "0.7")),
        "imgsz": int(os.getenv("ODP_VIDEO_IMGSZ", "640")),
        "max_frames": int(os.getenv("ODP_VIDEO_MAX_FRAMES", "24")),
        "max_analysis_frames": int(os.getenv("ODP_VIDEO_MAX_ANALYSIS_FRAMES", "60")),
        "max_keyframes": int(os.getenv("ODP_VIDEO_MAX_KEYFRAMES", "6")),
        "max_upload_mb": int(os.getenv("ODP_VIDEO_MAX_UPLOAD_MB", "200")),
        "time_budget_seconds": float(os.getenv("ODP_VIDEO_TIME_BUDGET", "600")),
    },
}

