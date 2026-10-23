from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .types import GenerationResult


class BackendError(RuntimeError):
    pass


class LlamaCppBackend:
    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1",
                 model: str = "local", timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, messages: list[dict[str, str]], *, seed: int = 20261023,
                 max_tokens: int = 64, temperature: float = 0.7,
                 top_p: float = 0.8) -> GenerationResult:
        body = json.dumps({
            "model": self.model, "messages": messages,
            "temperature": temperature, "top_p": top_p,
            "top_k": 20, "max_tokens": max_tokens, "seed": seed,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise BackendError(f"local llama.cpp request failed: {exc}") from exc
        elapsed = time.perf_counter() - start
        try:
            text = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError("local server returned an unexpected response") from exc
        usage = payload.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        return GenerationResult(
            text=text, prompt_tokens=prompt_tokens or len(" ".join(m["content"] for m in messages).split()),
            completion_tokens=completion_tokens or len(text.split()),
            elapsed_seconds=elapsed, backend_id=f"llama.cpp:{self.model}",
            stop_reason=str(payload.get("choices", [{}])[0].get("finish_reason", "unknown")),
        )

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url.rsplit("/v1", 1)[0] + "/health",
                                        timeout=5) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError):
            return False
