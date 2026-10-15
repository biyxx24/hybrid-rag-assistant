"""Vector search over the chunk index, multi-query fusion and context formatting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from langchain_chroma import Chroma


@dataclass
class Hit:
    chunk_id: str
    source: str
    content: str
    distance: float  # cosine distance of the closest matching query
    found_by: list[str] = field(default_factory=list)  # labels of the queries that retrieved it
    fused_score: float = 0.0
    rank: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def search(store: Chroma, query: str, k: int) -> list[Hit]:
    hits = []
    for doc, distance in store.similarity_search_with_score(query, k=k):
        hits.append(
            Hit(
                chunk_id=doc.metadata.get("chunk_id", doc.page_content[:40]),
                source=doc.metadata.get("source", "?"),
                content=doc.page_content,
                distance=float(distance),
            )
        )
    return hits


def reciprocal_rank_fusion(ranked_lists: list[list[str]], c: int = 60) -> dict[str, float]:
    """Merge several rankings: each item scores sum(1 / (c + rank)) over the lists it appears in."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank)
    return scores


def retrieve(
    store: Chroma, queries: list[tuple[str, str]], k: int, per_query_k: int | None = None
) -> list[Hit]:
    """Search with every (label, text) query, fuse the rankings with RRF and return the top ``k``."""
    per_query_k = per_query_k or max(k, 5)
    by_key: dict[str, Hit] = {}
    ranked_lists: list[list[str]] = []
    for label, text in queries:
        hits = search(store, text, per_query_k)
        ranked_lists.append([hit.chunk_id for hit in hits])
        for hit in hits:
            seen = by_key.get(hit.chunk_id)
            if seen is None:
                hit.found_by = [label]
                by_key[hit.chunk_id] = hit
            else:
                seen.distance = min(seen.distance, hit.distance)
                seen.found_by.append(label)
    scores = reciprocal_rank_fusion(ranked_lists)
    ordered = sorted(by_key.values(), key=lambda hit: (-scores[hit.chunk_id], hit.distance))[:k]
    for rank, hit in enumerate(ordered, start=1):
        hit.fused_score = round(scores[hit.chunk_id], 5)
        hit.rank = rank
    return ordered


def format_context(hits: list[Hit]) -> str:
    """Number the chunks so the answer can cite them as [1], [2], ..."""
    return "\n\n".join(f"[{hit.rank}] (source: {hit.source})\n{hit.content}" for hit in hits)
