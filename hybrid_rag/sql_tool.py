"""Read-only Text-to-SQL over a SQLite database.

The model only ever sees the schema and writes a query; the query is cleaned, checked and run
over a read-only connection, so a wrong or malicious query cannot change any record.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

from .llm import Message

FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|TRUNCATE|GRANT)\b",
    re.IGNORECASE,
)

SQL_SYSTEM = (
    "You are a careful {dialect} analyst. Given a database schema and a question, write ONE "
    "syntactically correct {dialect} SELECT query that answers it. Use only the tables and "
    "columns in the schema, join tables through their key columns, name aggregated columns "
    "clearly, order results when a ranking is asked for, and limit results to {limit} rows. "
    "Return only the SQL, without explanation."
)


class UnsafeSQL(ValueError):
    """Raised when a generated query is not a single read-only SELECT statement."""


def clean_sql(text: str) -> str:
    """Pull the SQL out of a model reply (code fences, ``SQLQuery:`` labels, leading prose)."""
    query = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", query, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        query = fenced.group(1).strip()
    query = re.sub(r"^```(?:sql)?\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(r"^SQLQuery:\s*", "", query, flags=re.IGNORECASE)
    start = re.search(r"^\s*(?:SELECT|WITH)\b", query, flags=re.MULTILINE | re.IGNORECASE)
    if start is None:
        raise UnsafeSQL(f"No SELECT or WITH query found in model output: {text!r}")
    return query[start.start():].strip()


def guard_sql(sql: str, default_limit: int = 50) -> str:
    """Allow a single SELECT/WITH statement only; add a LIMIT when the query has none."""
    sql = sql.strip().rstrip(";").strip()
    if ";" in sql:
        raise UnsafeSQL("Only a single statement is allowed.")
    if not re.match(r"^(SELECT|WITH)\b", sql, flags=re.IGNORECASE):
        raise UnsafeSQL("Only SELECT queries are allowed.")
    if FORBIDDEN.search(sql):
        raise UnsafeSQL("The query contains a statement that is not allowed.")
    if not re.search(r"\bLIMIT\s+\d+", sql, flags=re.IGNORECASE):
        sql = f"{sql}\nLIMIT {default_limit}"
    return sql


def frame_to_text(frame: pd.DataFrame, max_rows: int = 20) -> str:
    """Render query results for the answer prompt (capped, so large results stay cheap)."""
    if frame.empty:
        return "(the query returned no rows)"
    text = frame.head(max_rows).to_string(index=False)
    if len(frame) > max_rows:
        text += f"\n... ({len(frame) - max_rows} more rows not shown)"
    return text


class SQLTool:
    def __init__(self, db_path: Path | str, sample_rows: int = 2, default_limit: int = 50):
        db_path = Path(db_path)
        if not db_path.is_file():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")
        # mode=ro opens the file read-only, so generated queries cannot change its records.
        self.uri = f"sqlite:///file:{db_path.resolve().as_posix()}?mode=ro&uri=true"
        self.db = SQLDatabase.from_uri(self.uri, sample_rows_in_table_info=sample_rows)
        self.engine = create_engine(self.uri)
        self.default_limit = default_limit

    @property
    def tables(self) -> list[str]:
        return sorted(self.db.get_usable_table_names())

    @property
    def schema(self) -> str:
        """CREATE TABLE statements plus a few sample rows, as LangChain renders them."""
        return self.db.get_table_info()

    def generate_sql(self, llm, question: str) -> str:
        messages: list[Message] = [
            {
                "role": "system",
                "content": SQL_SYSTEM.format(dialect=self.db.dialect, limit=self.default_limit),
            },
            {
                "role": "user",
                "content": f"Database schema:\n{self.schema}\n\nQuestion: {question}\n\nSQL query:",
            },
        ]
        raw = llm.complete(messages, task="sql")
        return guard_sql(clean_sql(raw), self.default_limit)

    def run(self, sql: str) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.engine)
