import pytest

from hybrid_rag.fakes import FakeChatClient
from hybrid_rag.sql_tool import SQLTool, UnsafeSQL, clean_sql, frame_to_text, guard_sql


def test_clean_sql_strips_fences_and_labels():
    raw = "Here is the query:\n```sql\nSQLQuery: SELECT 1\n```"
    assert clean_sql(raw) == "SELECT 1"


def test_clean_sql_rejects_output_without_select():
    with pytest.raises(UnsafeSQL):
        clean_sql("I cannot answer that.")


def test_guard_adds_limit_and_keeps_existing_limit():
    assert guard_sql("SELECT * FROM Track") == "SELECT * FROM Track\nLIMIT 50"
    assert guard_sql("SELECT * FROM Track LIMIT 3;") == "SELECT * FROM Track LIMIT 3"


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM Track",
        "SELECT 1; DROP TABLE Track",
        "SELECT * FROM Track; PRAGMA writable_schema=1",
        "WITH x AS (SELECT 1) UPDATE Track SET Name = 'x'",
    ],
)
def test_guard_rejects_writes_and_multiple_statements(sql):
    with pytest.raises(UnsafeSQL):
        guard_sql(sql)


def test_sql_tool_reads_schema_and_runs_read_only(settings):
    tool = SQLTool(settings.db_path)
    assert {"Album", "Artist", "Customer", "Employee", "Invoice", "Track"} <= set(tool.tables)
    assert "CREATE TABLE" in tool.schema
    frame = tool.run("SELECT COUNT(*) AS n FROM Employee")
    assert int(frame.iloc[0]["n"]) == 8
    with pytest.raises(Exception):
        tool.run("DELETE FROM Employee")  # the connection is opened with mode=ro


def test_generate_sql_goes_through_the_guard(settings):
    tool = SQLTool(settings.db_path)
    llm = FakeChatClient(scripted={"sql": "```sql\nSELECT Name FROM Genre ORDER BY Name\n```"})
    sql = tool.generate_sql(llm, "List the genres")
    assert sql == "SELECT Name FROM Genre ORDER BY Name\nLIMIT 50"
    assert len(tool.run(sql)) == 25
    assert "CREATE TABLE" in llm.calls[0][1][1]["content"]  # the schema was in the prompt


def test_frame_to_text_caps_rows(settings):
    frame = SQLTool(settings.db_path).run("SELECT TrackId FROM Track LIMIT 30")
    text = frame_to_text(frame, max_rows=5)
    assert "25 more rows" in text
