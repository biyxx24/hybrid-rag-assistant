"""Offline stand-ins for the embedding model and the chat model.

They let the tests, the evaluation harness and a demo of the UI run without an API key:

* ``HashEmbeddings`` is a deterministic bag-of-words hashing embedding, so lexical overlap
  still produces sensible nearest neighbours.
* ``FakeChatClient`` answers each pipeline task (router, rewrite, hyde, sql, answer ...) with
  simple rules instead of a model. Tests can also script exact replies per task.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Iterable, Iterator

from langchain_core.embeddings import Embeddings

from .llm import Message, Usage, count_message_tokens

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "i", "if", "in", "is", "it", "my", "of", "on", "or", "the", "to", "we", "what", "when",
    "which", "who", "with", "you", "your",
}


class HashEmbeddings(Embeddings):
    """Hash each word into one of ``dim`` buckets; L2-normalise so cosine distance works."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in TOKEN_RE.findall(text.lower()):
            if len(token) < 2 or token in STOPWORDS:
                continue
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


SQL_HINTS = (
    "how many", "count", "total", "top", "most", "best-selling", "best selling", "revenue",
    "sales", "sells", "invoice", "invoices", "customer", "customers", "employee", "employees",
    "average", "which country", "which genre", "spent", "purchased", "highest", "lowest", "list",
)
DOC_HINTS = (
    "policy", "refund", "membership", "support", "price", "prices", "format", "formats",
    "discount", "how do i", "hours", "payment", "founded", "headquartered", "headquarters",
    "plus", "studio", "tier", "student", "students", "currency", "contact", "cancel", "reach",
)
GREETINGS = ("hello", "hi", "hey", "thanks", "thank you", "good morning")


def _last_user_content(messages: list[Message]) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


def _question_from(messages: list[Message]) -> str:
    """Pull the user's question out of a pipeline prompt."""
    content = _last_user_content(messages)
    for pattern in (r"## Question\n(.+)", r"(?:Follow-up question|Question):\s*(.+)"):
        match = re.search(pattern, content)
        if match:
            return match.group(1).strip()
    return content.strip()


def _mentions(text: str, hints: tuple[str, ...]) -> bool:
    """True when one of the hint phrases occurs as whole words (so 'discount' is not 'count')."""
    return any(re.search(r"\b" + re.escape(hint) + r"\b", text) for hint in hints)


class FakeChatClient:
    """Rule-based replacement for :class:`hybrid_rag.llm.ChatClient`; makes no network calls."""

    model = "offline-fake"

    def __init__(self, scripted: dict[str, str | list[str]] | None = None):
        self.usage = Usage()
        self.calls: list[tuple[str, list[Message]]] = []
        self.scripted: dict[str, list[str]] = {
            task: list(reply) if isinstance(reply, list) else [reply]
            for task, reply in (scripted or {}).items()
        }

    # -- the shared interface -------------------------------------------------------------
    def complete(self, messages: Iterable[Message], *, task: str = "general", **_: object) -> str:
        messages = list(messages)
        self.calls.append((task, messages))
        self.usage.add(count_message_tokens(messages), 20)
        if self.scripted.get(task):
            return self.scripted[task].pop(0)
        handler = getattr(self, f"_{task}", self._general)
        if task == "answer" and "## Question" not in _last_user_content(messages):
            handler = self._general  # plain chat turn: no retrieved material to answer from
        return handler(_question_from(messages))

    def stream(self, messages: Iterable[Message], *, task: str = "answer", **_: object) -> Iterator[str]:
        for word in self.complete(messages, task=task).split(" "):
            yield word + " "

    # -- one rule per pipeline task ---------------------------------------------------------
    def _router(self, question: str) -> str:
        lowered = question.lower()
        wants_sql = _mentions(lowered, SQL_HINTS)
        wants_docs = _mentions(lowered, DOC_HINTS)
        if lowered.startswith(GREETINGS) and not (wants_sql or wants_docs):
            route = "chat"
        elif wants_sql and wants_docs:
            route = "both"
        elif wants_sql:
            route = "sql"
        else:
            route = "docs"
        return json.dumps(
            {
                "route": route,
                "reason": "offline keyword heuristic",
                "doc_question": question if route in ("docs", "both") else "",
                "sql_question": question if route in ("sql", "both") else "",
            }
        )

    def _condense(self, question: str) -> str:
        return question

    def _rewrite(self, question: str) -> str:
        core = question.rstrip("?")
        return f"1. {core}?\n2. Details about {core.lower()}\n3. Information regarding {core.lower()}"

    def _hyde(self, question: str) -> str:
        return (
            f"{question.rstrip('?')}: this knowledge-base passage states the applicable policy, "
            "pricing and process in formal terms."
        )

    def _sql(self, question: str) -> str:
        lowered = question.lower()
        if "employee" in lowered or "agent" in lowered:
            return "SELECT FirstName, LastName, Title FROM Employee"
        if "genre" in lowered:
            return (
                "SELECT g.Name AS Genre, SUM(il.Quantity) AS TracksSold\n"
                "FROM InvoiceLine il JOIN Track t ON il.TrackId = t.TrackId\n"
                "JOIN Genre g ON t.GenreId = g.GenreId\n"
                "GROUP BY g.Name ORDER BY TracksSold DESC LIMIT 5"
            )
        if "customer" in lowered or "spent" in lowered:
            return (
                "SELECT c.FirstName, c.LastName, c.Country, ROUND(SUM(i.Total), 2) AS LifetimeSpend\n"
                "FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId\n"
                "GROUP BY c.CustomerId ORDER BY LifetimeSpend DESC LIMIT 5"
            )
        return "SELECT COUNT(*) AS TrackCount FROM Track"

    def _answer(self, question: str) -> str:
        return (
            "(Offline demo) The retrieved material for this question is listed in the trace "
            "below; with an API key configured, the model writes the answer from it [1]."
        )

    def _general(self, question: str) -> str:
        return f"(Offline demo) You said: {question}"
