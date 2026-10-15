"""Shared fixtures: everything runs offline (RAG_FAKE=1) with a temporary index directory."""

import os

os.environ["RAG_FAKE"] = "1"

import pytest  # noqa: E402

from hybrid_rag.config import Settings  # noqa: E402
from hybrid_rag.indexing import build_index  # noqa: E402
from hybrid_rag.pipeline import Pipeline  # noqa: E402


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    return Settings(fake=True, persist_dir=tmp_path_factory.mktemp("chroma"))


@pytest.fixture(scope="session")
def index(settings):
    return build_index(settings)


@pytest.fixture
def pipeline(settings, index) -> Pipeline:
    store, info = index
    return Pipeline(settings=settings, store=store, index_info=info)
