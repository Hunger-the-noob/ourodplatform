"""大模型客户端：兼容 OpenAI Chat Completions 协议的任意服务。

默认按 DeepSeek 配置（https://api.deepseek.com/v1 + deepseek-chat），
只要把 ``LLM_BASE_URL`` / ``LLM_API_KEY`` / ``LLM_MODEL`` 换成其他家
（OpenAI、通义千问 DashScope 兼容模式、智谱、Kimi、本地 Ollama / vLLM）也能直接用。

设计要点：
- 只用 ``requests``，不依赖官方 SDK，换供应商无需改代码；
- 所有异常统一转成中文 ``LLMError``，前端可以直接展示；
- 未配置 API Key 时 ``configured`` 为 False，调用方改用本地规则解释。
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import requests
from django.conf import settings


class LLMError(RuntimeError):
    """大模型调用失败（含配置缺失、网络异常、接口返回错误）。"""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    provider: str
    timeout: float = 120.0
    temperature: float = 0.3
    max_tokens: int = 1600

    @classmethod
    def from_settings(cls) -> LLMConfig:
        raw: Mapping[str, Any] = (getattr(settings, "ASSISTANT", {}) or {}).get("llm", {}) or {}
        return cls(
            base_url=str(raw.get("base_url") or "").rstrip("/"),
            api_key=str(raw.get("api_key") or "").strip(),
            model=str(raw.get("model") or "").strip(),
            provider=str(raw.get("provider") or "OpenAI 兼容服务").strip(),
            timeout=float(raw.get("timeout") or 120.0),
            temperature=float(raw.get("temperature") or 0.3),
            max_tokens=int(raw.get("max_tokens") or 1600),
        )

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    @property
    def label(self) -> str:
        if not self.configured:
            return "未配置（本地规则解释）"
        return f"{self.provider} · {self.model}"

    def public(self) -> dict[str, Any]:
        """给前端展示的脱敏信息（绝不下发 api_key）。"""
        return {
            "configured": self.configured,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "label": self.label,
        }


def _endpoint(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    detail = ""
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            detail = str(error.get("message") or "")
        elif error:
            detail = str(error)
        detail = detail or str(payload.get("message") or "")
    detail = detail.strip()
    if len(detail) > 180:
        detail = detail[:180] + "…"

    status = response.status_code
    if status in (401, 403):
        hint = "鉴权失败：请检查 LLM_API_KEY 是否有效、是否有该模型权限。"
    elif status == 404:
        hint = "接口路径不存在：请确认 LLM_BASE_URL 是否包含 /v1 且服务支持 /chat/completions。"
    elif status == 429:
        hint = "请求过于频繁或额度不足，请稍后重试。"
    elif status >= 500:
        hint = "大模型服务暂时不可用，请稍后重试。"
    else:
        hint = "大模型接口返回错误。"
    return f"{hint}（HTTP {status}）{('：' + detail) if detail else ''}"


def chat(
    messages: Sequence[Mapping[str, str]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    config: LLMConfig | None = None,
) -> str:
    """调用 Chat Completions，返回助手回复文本。

    Raises:
        LLMError: 未配置、网络异常、超时或接口返回非 2xx。
    """
    cfg = config or LLMConfig.from_settings()
    if not cfg.configured:
        raise LLMError("尚未配置大模型：请设置 LLM_API_KEY（以及可选的 LLM_BASE_URL / LLM_MODEL）。")

    payload = {
        "model": cfg.model,
        "messages": [dict(item) for item in messages],
        "temperature": cfg.temperature if temperature is None else float(temperature),
        "max_tokens": cfg.max_tokens if max_tokens is None else int(max_tokens),
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    request_timeout = float(timeout or cfg.timeout)
    url = _endpoint(cfg.base_url)

    started = time.perf_counter()
    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=(10, request_timeout),
        )
    except requests.Timeout as exc:
        raise LLMError(f"大模型响应超时（>{request_timeout:.0f}s）：{url}") from exc
    except requests.RequestException as exc:
        raise LLMError(
            f"无法连接大模型服务：{url}。请检查网络、代理或 LLM_BASE_URL 配置。（{exc.__class__.__name__}）"
        ) from exc

    if response.status_code >= 400:
        raise LLMError(_error_message(response))

    try:
        data = response.json()
    except ValueError as exc:
        raise LLMError("大模型返回的不是合法 JSON，请检查服务地址是否为 OpenAI 兼容接口。") from exc

    choices = data.get("choices") or []
    if not choices:
        raise LLMError(f"大模型返回为空（耗时 {time.perf_counter() - started:.1f}s）。")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):  # 少数兼容实现返回分段内容
        content = "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict)
        )
    content = (content or "").strip()
    if not content:
        raise LLMError("大模型返回了空内容，请更换模型或降低 max_tokens 后重试。")
    return content


def status() -> dict[str, Any]:
    cfg = LLMConfig.from_settings()
    return cfg.public()
