"""Query transformations run before retrieval.

* ``condense_question`` turns a follow-up ("and for students?") into a standalone question
  using the conversation history, so retrieval does not depend on the chat context.
* ``rewrite_queries`` produces alternative phrasings (multi-query retrieval).
* ``hypothetical_document`` implements HyDE: embed a hypothetical answer passage instead of
  the question, because passages resemble passages more than questions do.
"""

from __future__ import annotations

from .llm import Message, numbered_lines

CONDENSE_SYSTEM = (
    "You rewrite the latest follow-up question of a conversation as a standalone question that "
    "can be understood without the conversation. Keep the user's language and intent, resolve "
    "pronouns and references, and add nothing new. Return only the rewritten question."
)

REWRITE_SYSTEM = (
    "You are an expert at rewriting search queries to improve information retrieval. Generate "
    "alternative phrasings that express the same intent with different wording (synonyms, "
    "formal and everyday terms). Keep names and numbers unchanged; add no new claims. Return a "
    "numbered list with one complete question per line and nothing else."
)

HYDE_SYSTEM = (
    "Write a short passage (60-100 words) that could appear in a company knowledge base and "
    "that answers the question. Use formal, specific wording, but invent no numbers, names or "
    "citations: the passage is only used as search text. Return the passage only."
)


def condense_question(llm, question: str, history: list[Message], max_turns: int = 6) -> str:
    """Rewrite ``question`` as a standalone question given the recent ``history``."""
    if not history:
        return question
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-max_turns:])
    messages = [
        {"role": "system", "content": CONDENSE_SYSTEM},
        {
            "role": "user",
            "content": f"Conversation so far:\n{transcript}\n\nFollow-up question: {question}\n\n"
            "Standalone question:",
        },
    ]
    rewritten = llm.complete(messages, task="condense").strip().strip('"')
    return rewritten or question


def rewrite_queries(llm, question: str, n: int = 3) -> list[str]:
    """Return up to ``n`` alternative phrasings of ``question``."""
    if n <= 0:
        return []
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": f"Generate exactly {n} alternative phrasings.\n\nQuestion: {question}"},
    ]
    return numbered_lines(llm.complete(messages, task="rewrite", temperature=0.3))[:n]


def hypothetical_document(llm, question: str) -> str:
    """Return a hypothetical knowledge-base passage answering ``question`` (HyDE)."""
    messages = [
        {"role": "system", "content": HYDE_SYSTEM},
        {"role": "user", "content": f"Question: {question}"},
    ]
    return llm.complete(messages, task="hyde", temperature=0.3)
