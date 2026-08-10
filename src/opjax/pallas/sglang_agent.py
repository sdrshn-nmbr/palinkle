"""mini-swe-agent over an SGLang OpenAI-compatible endpoint."""

from __future__ import annotations

from typing import Any

from minisweagent.models.litellm_model import LitellmModel


class SGLangEndpointModel(LitellmModel):
    """The stock mini-swe tool protocol with reproducible SGLang metadata."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_id: str,
        model_revision: str,
        runtime_revision: str,
        precision: str,
        seed: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        proxy_headers: dict[str, str] | None = None,
        reasoning_effort: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/v1"):
            endpoint += "/v1"
        model_kwargs: dict[str, Any] = {
            "api_base": endpoint,
            "api_key": api_key,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "timeout": 900,
        }
        if proxy_headers:
            model_kwargs["extra_headers"] = dict(proxy_headers)
        if reasoning_effort is not None:
            model_kwargs["reasoning_effort"] = reasoning_effort
        if chat_template_kwargs:
            model_kwargs["extra_body"] = {
                "chat_template_kwargs": dict(chat_template_kwargs)
            }
        super().__init__(
            model_name=f"openai/{model_id}",
            model_kwargs=model_kwargs,
            cost_tracking="ignore_errors",
        )
        self.endpoint = endpoint
        self.model_id = model_id
        self.model_revision = model_revision
        self.runtime_revision = runtime_revision
        self.precision = precision
        self.seed = seed
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "model": {
                    "provider": "sglang_openai",
                    "model_name": self.model_id,
                    "model_revision": self.model_revision,
                    "runtime_revision": self.runtime_revision,
                    "precision": self.precision,
                    "endpoint": self.endpoint,
                    "seed": self.seed,
                    "sampling": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "max_tokens": self.max_tokens,
                        "reasoning_effort": self.reasoning_effort,
                        "chat_template_kwargs": self.chat_template_kwargs,
                    },
                }
            }
        }
