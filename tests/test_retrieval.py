from hybrid_rag.config import Settings
from hybrid_rag.fakes import HashEmbeddings
from hybrid_rag.indexing import build_index, load_documents, split_documents
from hybrid_rag.retrieval import format_context, reciprocal_rank_fusion, retrieve, search


def test_hash_embeddings_are_deterministic_and_normalised():
    emb = HashEmbeddings()
    a, b = emb.embed_query("refund within 14 days"), emb.embed_query("refund within 14 days")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6
    assert emb.embed_documents(["x", "y"]) == [emb.embed_query("x"), emb.embed_query("y")]


def test_rrf_prefers_items_ranked_well_in_several_lists():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"], ["b", "d"]])
    assert sorted(scores, key=scores.get, reverse=True)[:2] == ["b", "a"]


def test_documents_are_loaded_and_split_with_ids(settings):
    docs = load_documents(settings.docs_dir)
    assert len(docs) == 6
    chunks = split_documents(docs, 600, 100)
    assert len(chunks) > len(docs)
    assert all("#" in c.metadata["chunk_id"] and c.metadata["source"].endswith(".md") for c in chunks)
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


def test_index_is_reused_until_inputs_change(settings, tmp_path):
    own = Settings(fake=True, persist_dir=tmp_path / "chroma")  # separate dir: a rebuild drops the collection
    _, info = build_index(own)
    assert info.rebuilt is True and info.n_docs == 6 and info.n_chunks > 6
    _, again = build_index(own)
    assert again.rebuilt is False and again.sources == info.sources
    _, forced = build_index(own, force=True)
    assert forced.rebuilt is True and forced.n_chunks == info.n_chunks


def test_search_and_fusion_return_ranked_hits(index):
    store, _ = index
    hits = search(store, "refund reimbursement window 14 days", k=3)
    assert len(hits) == 3 and hits[0].source == "refund_policy.md"

    fused = retrieve(store, [("original", "refund window"), ("rewrite 1", "how many days to request a refund")], k=4)
    assert [h.rank for h in fused] == [1, 2, 3, 4]
    assert fused[0].fused_score >= fused[-1].fused_score
    assert any(len(h.found_by) > 1 for h in fused)  # some chunk was found by both queries

    context = format_context(fused)
    assert context.startswith("[1] (source: ")
    assert "[4] (source: " in context
