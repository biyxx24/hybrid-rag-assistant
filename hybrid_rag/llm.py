"""A thin wrapper around the OpenAI chat API, plus token accounting and parsing helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import tiktoken

Message = dict[str, str]


@dataclass
class Usage:
    """Token counts accumulated over the calls made through one client."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.calls += 1

    def snapshot(self) -> "Usage":
        return Usage(self.prompt_tokens, self.completion_tokens, self.calls)

    def since(self, earlier: "Usage") -> "Usage":
        """Usage accumulated after ``earlier`` was taken."""
        return Usage(
            self.prompt_tokens - earlier.prompt_tokens,
            self.completion_tokens - earlier.completion_tokens,
            self.calls - earlier.calls,
        )


def get_tokenizer(model: str) -> tiktoken.Encoding:
    """Return the tiktoken encoding for a model name such as ``openai.gpt-4o`` or ``gpt-4o``."""
    bare = model.split(".", 1)[1] if model.startswith("openai.") else model
    try:
        return tiktoken.encoding_for_model(bare)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    return len(get_tokenizer(model).encode(text))


def count_message_tokens(messages: Iterable[Message], model: str = "gpt-4o") -> int:
    """Approximate the input tokens of a chat request: text tokens plus per-message overhead."""
    encoding = get_tokenizer(model)
    total = 3  # priming tokens for the assistant reply
    for message in messages:
        total += 4 + len(encoding.encode(message.get("content", "")))
    return total


class ChatClient:
    """Sends chat requests to an OpenAI-compatible endpoint and records token usage.

    ``task`` labels what a request is for (router, rewrite, sql, answer, ...). The real client
    ignores it; the offline ``FakeChatClient`` uses it to pick a canned behaviour, so both
    classes share one interface.
    """

    def __init__(self, model: str, client: Any | None = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.model = model
        self.usage = Usage()

    def _create(self, **kwargs: Any) -> Any:
        """Call the API; drop optional request fields the endpoint rejects and retry once."""
        from openai import BadRequestError

        optional = [key for key in ("response_format", "stream_options") if key in kwargs]
        try:
            return self.client.chat.completions.create(**kwargs)
        except BadRequestError:
            if not optional:
                raise
            for key in optional:
                kwargs.pop(key)
            return self.client.chat.completions.create(**kwargs)

    def complete(
        self,
        messages: Iterable[Message],
        *,
        task: str = "general",
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._create(**kwargs)
        if response.usage:
            self.usage.add(response.usage.prompt_tokens, response.usage.completion_tokens)
        return (response.choices[0].message.content or "").strip()

    def stream(
        self, messages: Iterable[Message], *, task: str = "answer", temperature: float = 0.2
    ) -> Iterator[str]:
        """Yield the reply piece by piece, as the Streamlit app displays it."""
        messages = list(messages)
        stream = self._create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        produced: list[str] = []
        usage_seen = False
        for chunk in stream:
            if getattr(chunk, "usage", None):
                self.usage.add(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                usage_seen = True
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                produced.append(text)
                yield text
        if not usage_seen:  # the endpoint did not report usage: estimate it with tiktoken
            self.usage.add(
                count_message_tokens(messages, self.model),
                count_tokens("".join(produced), self.model),
            )


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model reply; return ``{}`` when there is none."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def numbered_lines(text: str) -> list[str]:
    """Extract the items of a numbered list such as ``1. ...`` / ``2) ...``."""
    items = re.findall(r"^\s*\d+[.)]\s+(.+?)\s*$", text, flags=re.MULTILINE)
    return [item.strip().strip('"') for item in items if item.strip()]
