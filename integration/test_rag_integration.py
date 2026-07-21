"""Service-level integration tests against real Ollama + real (embedded) Qdrant.

These bypass the MCP stdio transport (covered separately in test_mcp_stdio.py)
so an MCP-layer regression is distinguishable from a RAG-stack regression.
"""

from __future__ import annotations

import pytest

from second_brain.service import RagService

pytestmark = pytest.mark.integration


def test_health_reports_real_backends_up(rag_service: RagService) -> None:
    health = rag_service.health()
    assert health == {"qdrant": True, "ollama": True, "fts": True}


def test_search_returns_semantically_ranked_hits(rag_service: RagService) -> None:
    result = rag_service.search("local-first retrieval augmented generation", top_k=5)
    assert result["hits"], "expected at least one real semantic/keyword hit"
    top_paths = [hit["metadata"].get("path") for hit in result["hits"]]
    assert "alpha.md" in top_paths


def test_search_filters_by_tag(rag_service: RagService) -> None:
    result = rag_service.search("project", filters={"tags": "gamma"}, top_k=10)
    paths = {hit["metadata"].get("path") for hit in result["hits"]}
    assert paths <= {"gamma.md"}
    assert paths, "tag filter should still return the matching note"


def test_query_builds_extractive_answer_with_citations(rag_service: RagService) -> None:
    result = rag_service.query("Gamma Project charter", top_k=5)
    assert result["answer_draft"] and result["answer_draft"] != "No relevant context found."
    assert result["citations"]
    assert all("path" in c for c in result["citations"])


def test_note_context_reports_links(rag_service: RagService) -> None:
    context = rag_service.note_context("alpha.md")
    assert context["chunk_count"] > 0
    assert "beta" in context["outlinks"]


def test_related_surfaces_graph_neighbors_with_evidence(rag_service: RagService) -> None:
    related = rag_service.related("alpha.md", top_k=5)
    assert related["found"] is True
    neighbor_paths = {n["path"]: n for n in related["neighbors"]}
    assert "beta.md" in neighbor_paths
    beta_edge = neighbor_paths["beta.md"]
    assert beta_edge["composite"] > 0
    assert "semantic" in beta_edge["signals"]


def test_connections_reports_direct_edge(rag_service: RagService) -> None:
    connections = rag_service.connections("alpha.md", "beta.md")
    assert connections["connected"] is True
    assert connections["direct_edge"] is not None
    assert connections["closeness"] > 0.0


def test_connections_unrelated_note_has_low_or_no_connection(rag_service: RagService) -> None:
    connections = rag_service.connections("alpha.md", "unrelated.md")
    # unrelated.md shares no tags/links/topic with alpha.md; it may still be
    # connected via a weak semantic edge, but must never be reported closer
    # than the true alpha<->beta relationship.
    direct_alpha_beta = rag_service.connections("alpha.md", "beta.md")
    assert connections["closeness"] <= direct_alpha_beta["closeness"]


def test_graph_map_summarizes_vault(rag_service: RagService) -> None:
    graph_map = rag_service.graph_map()
    assert graph_map["note_count"] >= 4
    assert isinstance(graph_map["clusters"], list)


def test_status_reflects_synced_index(rag_service: RagService) -> None:
    status = rag_service.status()
    assert status["model"] == rag_service.config.embedding_model
    assert status["model_available"] is True
    assert status["index_size"] > 0
    assert status["graph_nodes"] >= 4


def test_incremental_sync_is_idempotent_after_full_sync(rag_service: RagService) -> None:
    result = rag_service.sync(mode="incremental")
    assert result["errors"] == []
    assert result["processed"] == 0  # nothing changed since the session-scoped full sync
