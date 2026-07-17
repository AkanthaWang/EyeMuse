from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Optional
from urllib import error, request


class LLMClientError(RuntimeError):
    """Raised when the configured LLM endpoint cannot be used."""


@dataclass(frozen=True)
class LLMConfig:
    model_name: str
    model_url: str
    api_key: str

    @property
    def is_configured(self) -> bool:
        return bool(self.model_name and self.model_url and self.api_key)

    @property
    def request_url(self) -> str:
        url = self.model_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        if url == "https://open.bigmodel.cn/api/paas/v4":
            return f"{url}/chat/completions"
        return url


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Model connection settings should follow the repo .env on each app start,
        # even if the parent shell still has older values cached in its environment.
        if key in {"MODEL_NAME", "MODEL_URL", "MODEL_APIKEY"}:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


def load_llm_config(env_path: Optional[Path] = None) -> LLMConfig:
    root = Path(__file__).resolve().parents[4]
    _load_env_file(env_path or root / ".env")
    return LLMConfig(
        model_name=os.getenv("MODEL_NAME", "").strip(),
        model_url=os.getenv("MODEL_URL", "").strip(),
        api_key=os.getenv("MODEL_APIKEY", "").strip(),
    )


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None, *, timeout_seconds: float = 30.0) -> None:
        self._config = config or load_llm_config()
        self._timeout_seconds = timeout_seconds

    @property
    def configured(self) -> bool:
        return self._config.is_configured

    def generate_reply(
        self,
        *,
        user_text: str,
        conversation_items: Iterable[dict[str, str]],
        context_summary: str,
    ) -> str:
        return "".join(
            self.stream_reply(
                user_text=user_text,
                conversation_items=conversation_items,
                context_summary=context_summary,
            )
        ).strip()

    def stream_reply(
        self,
        *,
        user_text: str,
        conversation_items: Iterable[dict[str, str]],
        context_summary: str,
    ) -> Iterator[str]:
        if not self.configured:
            raise LLMClientError("MODEL_NAME, MODEL_URL, MODEL_APIKEY must all be set in .env.")

        payload = {
            "model": self._config.model_name,
            "messages": self._build_messages(
                user_text=user_text,
                conversation_items=conversation_items,
                context_summary=context_summary,
            ),
            "temperature": 0.7,
            "stream": True,
        }

        try:
            with request.urlopen(self._build_request(payload), timeout=self._timeout_seconds) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                yielded = False
                for event_payload in self._iter_stream_events(response, charset):
                    if event_payload == "[DONE]":
                        break

                    try:
                        parsed = json.loads(event_payload)
                    except json.JSONDecodeError:
                        continue

                    delta = self._extract_stream_delta(parsed)
                    if delta:
                        yielded = True
                        yield delta

                if not yielded:
                    raise LLMClientError("LLM stream did not include assistant content.")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise LLMClientError(f"LLM request could not reach server: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMClientError("LLM request timed out.") from exc

    def _build_request(self, payload: dict) -> request.Request:
        body = json.dumps(payload).encode("utf-8")
        return request.Request(
            self._config.request_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._config.api_key}",
                "Accept": "text/event-stream, application/json",
            },
            method="POST",
        )

    def _build_messages(
        self,
        *,
        user_text: str,
        conversation_items: Iterable[dict[str, str]],
        context_summary: str,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are EyeMuse, a warm and supportive desktop companion. "
                    "Reply in concise natural Chinese unless the user clearly asks for another language. "
                    "Use the camera/context summary when relevant, but do not claim to perceive anything you were not given."
                ),
            },
            {
                "role": "system",
                "content": f"Current local context: {context_summary}",
            },
        ]

        role_map = {
            "user": "user",
            "eyeMuse": "assistant",
            "system": "system",
        }
        for item in conversation_items:
            role = role_map.get(item.get("role", ""), "user")
            text = item.get("text", "").strip()
            if not text:
                continue
            messages.append({"role": role, "content": text})

        if not messages or messages[-1]["role"] != "user" or messages[-1]["content"] != user_text:
            messages.append({"role": "user", "content": user_text})
        return messages

    @staticmethod
    def _extract_content(payload: dict) -> str:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()

        output_text = payload.get("output_text")
        if isinstance(output_text, str):
            return output_text.strip()

        output = payload.get("output")
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content_items = item.get("content", [])
                if not isinstance(content_items, list):
                    continue
                for content_item in content_items:
                    if isinstance(content_item, dict) and content_item.get("type") in {"output_text", "text"}:
                        text = content_item.get("text")
                        if isinstance(text, str):
                            chunks.append(text.strip())
            if chunks:
                return "\n".join(chunk for chunk in chunks if chunk)

        return ""

    @staticmethod
    def _iter_stream_events(response, charset: str) -> Iterator[str]:
        data_lines: list[str] = []
        while True:
            raw_line = response.readline()
            if not raw_line:
                if data_lines:
                    yield "\n".join(data_lines)
                break

            line = raw_line.decode(charset, errors="replace").strip()
            if not line:
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines.clear()
                continue

            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif line.startswith("{"):
                yield line

    @staticmethod
    def _extract_stream_delta(payload: dict) -> str:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta", {})
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text")
                            if isinstance(text, str):
                                parts.append(text)
                    if parts:
                        return "".join(parts)

        event_type = payload.get("type")
        if event_type == "response.output_text.delta":
            delta = payload.get("delta")
            if isinstance(delta, str):
                return delta

        output = payload.get("output")
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content_items = item.get("content", [])
                if not isinstance(content_items, list):
                    continue
                for content_item in content_items:
                    if isinstance(content_item, dict) and content_item.get("type") in {"output_text", "text"}:
                        text = content_item.get("text")
                        if isinstance(text, str):
                            chunks.append(text)
            if chunks:
                return "".join(chunks)

        return ""
