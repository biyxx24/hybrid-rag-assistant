"""Decide which source(s) a question needs: documents, the database, both, or plain chat."""

from __future__ import annotations

from dataclasses import dataclass

from .llm import Message, extract_json

ROUTES = ("docs", "sql", "both", "chat")

ROUTER_SYSTEM = """You route questions for an assistant that has two sources of truth:
1. "docs": knowledge-base documents: {sources}
2. "sql": a read-only SQLite database with the tables: {tables}

Pick one route:
- "docs" for policies, explanations, how-to questions and anything the documents describe.
- "sql" for counts, rankings, totals and facts about specific records (customers, invoices, tracks, employees...).
- "both" when the answer needs a database fact AND document knowledge together.
- "chat" for greetings, small talk, or questions unrelated to either source.

Reply with JSON only:
{{"route": "docs|sql|both|chat", "reason": "<one short sentence>", "doc_question": "<question for the documents or empty>", "sql_question": "<question for the database or empty>"}}
For "both", decompose the question so that each sub-question can be answered by one source alone."""


@dataclass
class RouteDecision:
    route: str
    reason: str
    doc_question: str = ""
    sql_question: str = ""


def route_question(llm, question: str, sources: list[str], tables: list[str]) -> RouteDecision:
    messages: list[Message] = [
        {
            "role": "system",
            "content": ROUTER_SYSTEM.format(
                sources=", ".join(sources) or "(none)", tables=", ".join(tables) or "(none)"
            ),
        },
        {"role": "user", "content": f"Question: {question}"},
    ]
    data = extract_json(llm.complete(messages, task="router", json_mode=True))
    route = str(data.get("route", "")).strip().lower()
    if route not in ROUTES:
        return RouteDecision("docs", "router reply could not be parsed; defaulting to documents", question, "")
    doc_question = str(data.get("doc_question") or "").strip()
    sql_question = str(data.get("sql_question") or "").strip()
    if route in ("docs", "both") and not doc_question:
        doc_question = question
    if route in ("sql", "both") and not sql_question:
        sql_question = question
    return RouteDecision(route, str(data.get("reason", "")).strip(), doc_question, sql_question)
