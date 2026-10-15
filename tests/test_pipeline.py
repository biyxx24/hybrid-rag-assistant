import json

from hybrid_rag.fakes import FakeChatClient
from hybrid_rag.pipeline import Options, Pipeline


def test_docs_route_retrieves_and_cites(pipeline):
    answer, trace = pipeline.answer("How long do I have to request a refund?")
    assert trace.route == "docs"
    assert [q["label"] for q in trace.queries] == ["original", "rewrite 1", "rewrite 2", "rewrite 3", "hyde"]
    assert trace.hyde_document and len(trace.hits) == 4
    assert trace.hits[0]["source"] == "refund_policy.md"
    assert trace.sql == "" and trace.sql_error == ""
    assert "[1]" in answer
    assert trace.usage["calls"] == 4  # router + rewrite + hyde + answer; no condense without history
    assert trace.context_tokens > 0 and trace.elapsed_seconds >= 0
    assert "_started" not in trace.to_dict()


def test_sql_route_runs_a_read_only_query(pipeline):
    _, trace = pipeline.answer("Which genre sells the most tracks?")
    assert trace.route == "sql"
    assert trace.sql.startswith("SELECT g.Name AS Genre")
    assert trace.sql_columns == ["Genre", "TracksSold"]
    assert trace.sql_rows[0]["Genre"] == "Rock" and trace.sql_row_count == 5
    assert trace.hits == []


def test_both_route_uses_documents_and_database(pipeline):
    _, trace = pipeline.answer("Which customers spent more than 45 dollars and qualify for the free Studio tier?")
    assert trace.route == "both"
    assert trace.hits and trace.sql_rows
    assert "LifetimeSpend" in trace.sql_columns


def test_chat_route_skips_retrieval(pipeline):
    answer, trace = pipeline.answer("Hello there!")
    assert trace.route == "chat"
    assert trace.hits == [] and trace.sql == ""
    assert "Hello there!" in answer


def test_manual_route_and_options_are_respected(pipeline):
    _, trace = pipeline.answer("refund", options=Options(route="docs", k=2, rewrites=0, use_hyde=False))
    assert trace.route == "docs" and trace.route_reason == "route selected manually"
    assert [q["label"] for q in trace.queries] == ["original"]
    assert len(trace.hits) == 2 and trace.usage["calls"] == 1  # only the answer call


def test_history_is_condensed_into_a_standalone_question(settings, index):
    store, info = index
    llm = FakeChatClient(scripted={"condense": "Is there a student discount on Chinook Plus?"})
    pipeline = Pipeline(settings=settings, llm=llm, store=store, index_info=info)
    history = [
        {"role": "user", "content": "What does Chinook Plus cost?"},
        {"role": "assistant", "content": "Plus is USD 4.99 per month [1]."},
    ]
    _, trace = pipeline.answer("And for students?", history=history)
    assert trace.standalone_question == "Is there a student discount on Chinook Plus?"
    assert trace.route == "docs"
    assert trace.hits[0]["source"] == "membership_program.md"
    answer_messages = llm.calls[-1][1]
    assert answer_messages[1]["content"] == "What does Chinook Plus cost?"  # history precedes the question
    assert answer_messages[-1]["content"].startswith("## Question\nAnd for students?")


def test_unparseable_router_reply_falls_back_to_docs(settings, index):
    store, info = index
    llm = FakeChatClient(scripted={"router": "I am not sure."})
    pipeline = Pipeline(settings=settings, llm=llm, store=store, index_info=info)
    _, trace = pipeline.answer("Where is the company headquartered?")
    assert trace.route == "docs" and "defaulting" in trace.route_reason
    assert trace.hits[0]["source"] == "about_chinook_records.md"


def test_bad_sql_is_reported_not_raised(settings, index):
    store, info = index
    llm = FakeChatClient(scripted={"router": json.dumps({"route": "sql", "reason": "t", "sql_question": "x"}),
                                   "sql": "DROP TABLE Track"})
    pipeline = Pipeline(settings=settings, llm=llm, store=store, index_info=info)
    _, trace = pipeline.answer("Delete everything")
    assert trace.route == "sql" and trace.sql == ""
    assert trace.sql_error.startswith("UnsafeSQL")
    assert "could not be run" in llm.calls[-1][1][-1]["content"]
