"""Streamlit chat interface for the Hybrid RAG Assistant.

Run from the project root:  streamlit run app.py
Every answer comes with an expandable trace: route decision, transformed queries, retrieved
chunks with their fusion scores, the generated SQL and its rows, and the tokens spent.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from hybrid_rag.indexing import build_index
from hybrid_rag.pipeline import Options, Pipeline

EXAMPLES = [
    "How long do I have to request a refund?",
    "Which genre sells the most tracks?",
    "Which customers qualify for the complimentary Studio tier?",
    "Who is the support agent for customers in Brazil, and when can I reach them?",
]

st.set_page_config(page_title="Hybrid RAG Assistant", page_icon="🎧", layout="wide")


@st.cache_resource(show_spinner="Loading the document index and the database…")
def load_pipeline() -> Pipeline:
    return Pipeline()


pipeline = load_pipeline()
settings = pipeline.settings

# ---------------------------------------------------------------- sidebar: settings & status
with st.sidebar:
    st.header("Settings")
    if settings.fake:
        st.warning("Offline demo mode (RAG_FAKE=1): no API calls are made.", icon="🔌")
    else:
        st.caption(f"Chat model: `{settings.chat_model}`  \nEmbeddings: `{settings.embed_model}`")
    route = st.selectbox(
        "Route",
        ["auto", "docs", "sql", "both", "chat"],
        help="auto lets the model decide whether a question needs the documents, the database, both, or neither.",
    )
    k = st.slider("Chunks kept after fusion (k)", 1, 10, 4)
    rewrites = st.slider("Query rewrites (multi-query)", 0, 5, 3)
    use_hyde = st.toggle("HyDE: search with a hypothetical answer", value=True)
    show_trace = st.toggle("Show the trace under each answer", value=True)

    st.divider()
    info = pipeline.index_info
    if info:
        st.markdown(f"**Index:** {info.n_docs} documents · {info.n_chunks} chunks")
        st.caption("Sources: " + ", ".join(info.sources))
    st.markdown(f"**Database:** {len(pipeline.sql.tables)} tables (read-only)")
    if st.button("Rebuild index"):
        build_index(settings, force=True)
        load_pipeline.clear()
        st.rerun()

    st.divider()
    # The pipeline is cached for the whole server, so count from this browser session's start.
    if "usage_start" not in st.session_state:
        st.session_state.usage_start = pipeline.llm.usage.snapshot()
    usage = pipeline.llm.usage.since(st.session_state.usage_start)
    st.markdown(
        f"**Session tokens:** {usage.total_tokens:,}  \n"
        f"prompt {usage.prompt_tokens:,} · completion {usage.completion_tokens:,} · {usage.calls} calls"
    )
    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.usage_start = pipeline.llm.usage.snapshot()
        st.rerun()

# ---------------------------------------------------------------- main: chat
st.title("🎧 Hybrid RAG Assistant")
st.caption(
    f"Ask about {settings.assistant_name}: policies and guides are answered from the documents, "
    "sales questions from the SQL database, and mixed questions from both. Every answer shows its work."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_trace(trace: dict, expanded: bool = False) -> None:
    usage = trace.get("usage") or {}
    total = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    title = f"🔍 Trace · route: {trace['route']} · {trace['elapsed_seconds']} s · {total:,} tokens · {usage.get('calls', 0)} model calls"
    with st.expander(title, expanded=expanded):
        st.markdown(f"**Route:** `{trace['route']}` — {trace['route_reason']}")
        if trace["standalone_question"] != trace["question"]:
            st.markdown(f"**Standalone question:** {trace['standalone_question']}")
        if trace["route"] == "both":
            st.markdown(
                f"**Decomposed:** documents ← *{trace['doc_question']}*  \n"
                f"database ← *{trace['sql_question']}*"
            )
        if trace["queries"]:
            st.markdown("**Search queries sent to the vector index**")
            st.table(pd.DataFrame(trace["queries"]))
        if trace["hits"]:
            st.markdown(f"**Retrieved chunks after reciprocal rank fusion (top {len(trace['hits'])})**")
            for hit in trace["hits"]:
                st.markdown(
                    f"**[{hit['rank']}] {hit['source']}** · cosine distance {hit['distance']:.3f} · "
                    f"RRF {hit['fused_score']:.4f} · found by: {', '.join(hit['found_by'])}"
                )
                st.text(hit["content"])
        if trace["sql"]:
            st.markdown("**Generated SQL (read-only connection)**")
            st.code(trace["sql"], language="sql")
            if trace["sql_rows"]:
                st.dataframe(pd.DataFrame(trace["sql_rows"]), use_container_width=True)
                st.caption(f"{trace['sql_row_count']} row(s) returned")
            elif not trace["sql_error"]:
                st.caption("The query returned no rows.")
        if trace["sql_error"]:
            st.error(trace["sql_error"])
        st.caption(
            f"Answer request size: {trace['context_tokens']:,} tokens · this question used "
            f"{usage.get('prompt_tokens', 0):,} prompt + {usage.get('completion_tokens', 0):,} completion tokens"
        )


latest = len(st.session_state.messages) - 1
for position, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if show_trace and message.get("trace"):
            render_trace(message["trace"], expanded=position == latest)  # newest trace open

if not st.session_state.messages and st.query_params.get("q"):
    st.session_state.pending_question = st.query_params["q"]  # open the app with ?q=... to start with a question

if not st.session_state.messages:
    st.markdown("**Try one of these:**")
    columns = st.columns(len(EXAMPLES))
    for column, example in zip(columns, EXAMPLES):
        if column.button(example, use_container_width=True):
            st.session_state.pending_question = example
            st.rerun()

question = st.chat_input("Ask a question…") or st.session_state.pop("pending_question", None)

if question:
    with st.chat_message("user"):
        st.markdown(question)
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    options = Options(route=route, k=k, rewrites=rewrites, use_hyde=use_hyde)
    with st.chat_message("assistant"):
        with st.spinner("Routing, transforming the query and retrieving…"):
            prepared = pipeline.prepare(question, history, options)
        answer = st.write_stream(pipeline.llm.stream(prepared.messages))
        trace = pipeline.finalize(prepared.trace).to_dict()
        if show_trace:
            render_trace(trace, expanded=True)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace})
    st.rerun()
