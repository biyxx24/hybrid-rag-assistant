"""The end-to-end pipeline: condense -> route -> transform & retrieve / Text-to-SQL -> answer.

``Pipeline.prepare`` does everything except the final answer call and returns the messages for
it together with a :class:`Trace` of every intermediate step. Callers either stream the answer
(the Streamlit app) or call :meth:`Pipeline.answer` to get it in one go (CLI, tests).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from time import perf_counter

from .config import Settings
from .indexing import IndexInfo, build_index
from .llm import ChatClient, Message, Usage, count_message_tokens
from .retrieval import Hit, format_context, retrieve
from .router import ROUTES, RouteDecision, route_question
from .sql_tool import SQLTool, UnsafeSQL, frame_to_text
from .transforms import condense_question, hypothetical_document, rewrite_queries

ANSWER_SYSTEM = """You are the knowledge assistant for {name}. Answer the user's question using ONLY the material provided in the message: document excerpts and/or a database query with its result.
Rules:
- Cite document excerpts by number, like [1] or [2][3], right after the sentence they support.
- When you use the database result, say in plain words what the query measured (for example "by number of tracks sold") and mention any limit or assumption in it.
- If the material does not contain the answer, say so instead of guessing.
- Be concise and answer in the language of the question."""

CHAT_SYSTEM = """You are the assistant for {name}, a digital music store. Reply briefly and helpfully. For questions about store policies or sales data, invite the user to ask them directly so you can look them up in the documents or the database."""


@dataclass
class Options:
    route: str = "auto"  # auto | docs | sql | both | chat
    k: int = 4  # chunks kept after fusion
    rewrites: int = 3  # alternative phrasings generated for multi-query retrieval
    use_hyde: bool = True
    max_history_turns: int = 6


@dataclass
class Trace:
    """Everything the pipeline did for one question, for display and for tests."""

    question: str
    standalone_question: str = ""
    route: str = ""
    route_reason: str = ""
    doc_question: str = ""
    sql_question: str = ""
    queries: list[dict] = field(default_factory=list)  # [{"label", "text"}]
    hyde_document: str = ""
    hits: list[dict] = field(default_factory=list)
    sql: str = ""
    sql_columns: list[str] = field(default_factory=list)
    sql_rows: list[dict] = field(default_factory=list)
    sql_row_count: int = 0
    sql_error: str = ""
    context_tokens: int = 0  # size of the final answer request
    usage: dict = field(default_factory=dict)  # tokens spent on this question, all calls
    elapsed_seconds: float = 0.0
    # bookkeeping, not exported
    _started: float = field(default=0.0, repr=False)
    _usage_before: Usage | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        data = asdict(self)
        return {key: value for key, value in data.items() if not key.startswith("_")}


@dataclass
class Prepared:
    messages: list[Message]
    trace: Trace


class Pipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        llm=None,
        store=None,
        index_info: IndexInfo | None = None,
        sql_tool: SQLTool | None = None,
    ):
        self.settings = settings or Settings()
        if llm is None:
            if self.settings.fake:
                from .fakes import FakeChatClient

                llm = FakeChatClient()
            else:
                llm = ChatClient(self.settings.chat_model)
        self.llm = llm
        if store is None:
            store, index_info = build_index(self.settings)
        self.store = store
        self.index_info = index_info
        self.sql = sql_tool or SQLTool(self.settings.db_path)

    # -- building blocks ----------------------------------------------------------------------
    def retrieve_docs(
        self, question: str, options: Options
    ) -> tuple[list[tuple[str, str]], str, list[Hit]]:
        """Transform ``question`` into search queries and return (queries, hyde_doc, fused hits)."""
        queries = [("original", question)]
        for number, rewritten in enumerate(rewrite_queries(self.llm, question, options.rewrites), 1):
            queries.append((f"rewrite {number}", rewritten))
        hyde_doc = ""
        if options.use_hyde:
            hyde_doc = hypothetical_document(self.llm, question)
            queries.append(("hyde", hyde_doc))
        return queries, hyde_doc, retrieve(self.store, queries, options.k)

    def query_database(self, question: str, trace: Trace) -> str:
        """Generate, guard and run a query; fill the trace; return the text for the prompt."""
        try:
            sql = self.sql.generate_sql(self.llm, question)
            trace.sql = sql
            frame = self.sql.run(sql)
        except (UnsafeSQL, Exception) as exc:  # noqa: BLE001 - surface any failure to the user
            trace.sql_error = f"{type(exc).__name__}: {exc}"
            return f"The database query could not be run: {trace.sql_error}"
        trace.sql_columns = [str(column) for column in frame.columns]
        trace.sql_rows = json.loads(frame.head(50).to_json(orient="records"))
        trace.sql_row_count = len(frame)
        return f"```sql\n{sql}\n```\nResult ({len(frame)} rows):\n{frame_to_text(frame)}"

    def answer_messages(
        self, route: str, question: str, history: list[Message], context: str, sql_section: str,
        options: Options,
    ) -> list[Message]:
        recent = [m for m in history if m["role"] in ("user", "assistant")][-options.max_history_turns:]
        if route == "chat":
            system = CHAT_SYSTEM.format(name=self.settings.assistant_name)
            return [{"role": "system", "content": system}, *recent, {"role": "user", "content": question}]
        parts = [f"## Question\n{question}"]
        if route in ("docs", "both"):
            parts.append("## Document excerpts\n" + (context or "(no document excerpts were retrieved)"))
        if route in ("sql", "both"):
            parts.append("## Database\n" + sql_section)
        system = ANSWER_SYSTEM.format(name=self.settings.assistant_name)
        return [{"role": "system", "content": system}, *recent, {"role": "user", "content": "\n\n".join(parts)}]

    # -- the pipeline -------------------------------------------------------------------------
    def prepare(self, question: str, history: list[Message] | None = None, options: Options | None = None) -> Prepared:
        options = options or Options()
        history = list(history or [])
        if options.route not in ("auto", *ROUTES):
            raise ValueError(f"Unknown route {options.route!r}")
        trace = Trace(question=question, _started=perf_counter(), _usage_before=self.llm.usage.snapshot())

        standalone = condense_question(self.llm, question, history, options.max_history_turns)
        trace.standalone_question = standalone

        if options.route == "auto":
            sources = self.index_info.sources if self.index_info else []
            decision = route_question(self.llm, standalone, sources, self.sql.tables)
        else:
            decision = RouteDecision(options.route, "route selected manually", standalone, standalone)
        trace.route, trace.route_reason = decision.route, decision.reason

        context, sql_section = "", ""
        if decision.route in ("docs", "both"):
            trace.doc_question = decision.doc_question or standalone
            queries, hyde_doc, hits = self.retrieve_docs(trace.doc_question, options)
            trace.queries = [{"label": label, "text": text} for label, text in queries]
            trace.hyde_document = hyde_doc
            trace.hits = [hit.to_dict() for hit in hits]
            context = format_context(hits)
        if decision.route in ("sql", "both"):
            trace.sql_question = decision.sql_question or standalone
            sql_section = self.query_database(trace.sql_question, trace)

        messages = self.answer_messages(decision.route, question, history, context, sql_section, options)
        trace.context_tokens = count_message_tokens(messages, self.settings.chat_model)
        self.finalize(trace)
        return Prepared(messages, trace)

    def finalize(self, trace: Trace) -> Trace:
        """Refresh the timing and token totals (call again after streaming the answer)."""
        if trace._usage_before is not None:
            trace.usage = asdict(self.llm.usage.since(trace._usage_before))
        trace.elapsed_seconds = round(perf_counter() - trace._started, 2)
        return trace

    def answer(self, question: str, history: list[Message] | None = None, options: Options | None = None) -> tuple[str, Trace]:
        prepared = self.prepare(question, history, options)
        text = self.llm.complete(prepared.messages, task="answer", temperature=0.2)
        self.finalize(prepared.trace)
        return text, prepared.trace
