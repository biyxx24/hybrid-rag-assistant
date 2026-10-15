"""Runtime configuration, read from environment variables and an optional ``.env`` file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable (1/true/yes/on)."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Everything the pipeline needs to know about models, data files and storage."""

    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "openai.gpt-4o"))
    embed_model: str = field(
        default_factory=lambda: os.getenv("EMBED_MODEL", "openai.text-embedding-3-large")
    )
    docs_dir: Path = field(default_factory=lambda: ROOT / os.getenv("DOCS_DIR", "data/docs"))
    db_path: Path = field(default_factory=lambda: ROOT / os.getenv("SQLITE_DB", "data/chinook.db"))
    persist_dir: Path = field(default_factory=lambda: ROOT / os.getenv("CHROMA_DIR", ".chroma"))
    collection: str = "hybrid_rag"
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "600")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))
    assistant_name: str = field(default_factory=lambda: os.getenv("ASSISTANT_NAME", "Chinook Records"))
    # Offline mode: hash-based embeddings and a rule-based stand-in for the chat model.
    fake: bool = field(default_factory=lambda: env_flag("RAG_FAKE"))

    def __post_init__(self) -> None:
        # Keep real and fake indexes apart: their embedding dimensions differ.
        if self.fake and self.persist_dir.name == ".chroma":
            self.persist_dir = self.persist_dir.with_name(".chroma_fake")
