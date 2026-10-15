"""Load the knowledge-base documents, split them into chunks and keep them in a Chroma index.

The index is persisted on disk together with a fingerprint of its inputs (document text,
chunking parameters, embedding model). It is rebuilt only when one of those changes.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings

SUPPORTED_SUFFIXES = {".md", ".txt"}
COLLECTION_METADATA = {"hnsw:space": "cosine"}  # distances in [0, 2]; lower is closer


def chroma_settings(persist_dir: Path) -> ChromaSettings:
    """Client settings for an on-disk index.

    A fresh object per client, because langchain-chroma writes into the one it is given, and
    ``is_persistent`` set explicitly: with custom settings langchain-chroma does not set it, and
    the index would silently live in memory only.
    """
    return ChromaSettings(
        anonymized_telemetry=False, is_persistent=True, persist_directory=str(persist_dir)
    )


@dataclass
class IndexInfo:
    n_docs: int
    n_chunks: int
    rebuilt: bool
    embed_model: str
    sources: list[str]


def load_documents(docs_dir: Path) -> list[Document]:
    """Read every .md / .txt file below ``docs_dir``; ``metadata['source']`` is the file name."""
    docs: list[Document] = []
    paths = sorted(p for p in Path(docs_dir).rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    for path in paths:
        for doc in TextLoader(str(path), encoding="utf-8").load():
            doc.metadata["source"] = path.name
            docs.append(doc)
    if not docs:
        raise FileNotFoundError(f"No .md or .txt documents found in {docs_dir}")
    return docs


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    for position, chunk in enumerate(chunks):
        start = chunk.metadata.get("start_index", position)
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}#{start}"
    return chunks


def fingerprint(docs: list[Document], settings: Settings) -> str:
    digest = hashlib.sha256()
    for doc in docs:
        digest.update(doc.metadata["source"].encode("utf-8"))
        digest.update(doc.page_content.encode("utf-8"))
    digest.update(
        f"{settings.embed_model}|{settings.chunk_size}|{settings.chunk_overlap}|fake={settings.fake}".encode()
    )
    return digest.hexdigest()


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.fake:
        from .fakes import HashEmbeddings

        return HashEmbeddings()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.embed_model)


def open_index(settings: Settings, embeddings: Embeddings | None = None) -> Chroma:
    return Chroma(
        collection_name=settings.collection,
        embedding_function=embeddings or get_embeddings(settings),
        persist_directory=str(settings.persist_dir),
        client_settings=chroma_settings(settings.persist_dir),
        collection_metadata=COLLECTION_METADATA,
    )


def build_index(settings: Settings, force: bool = False) -> tuple[Chroma, IndexInfo]:
    """Return the vector store for ``settings``, rebuilding it when its inputs changed."""
    docs = load_documents(settings.docs_dir)
    current = fingerprint(docs, settings)
    manifest_path = settings.persist_dir / "manifest.json"
    embeddings = get_embeddings(settings)

    if not force and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == current:
            store = open_index(settings, embeddings)
            return store, IndexInfo(
                manifest["n_docs"], manifest["n_chunks"], False, settings.embed_model, manifest["sources"]
            )

    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    settings.persist_dir.mkdir(parents=True, exist_ok=True)
    open_index(settings, embeddings).delete_collection()  # drop stale vectors first
    store = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=settings.collection,
        persist_directory=str(settings.persist_dir),
        client_settings=chroma_settings(settings.persist_dir),
        collection_metadata=COLLECTION_METADATA,
    )
    sources = sorted({doc.metadata["source"] for doc in docs})
    manifest = {
        "fingerprint": current,
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "sources": sources,
        "embed_model": settings.embed_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return store, IndexInfo(len(docs), len(chunks), True, settings.embed_model, sources)
