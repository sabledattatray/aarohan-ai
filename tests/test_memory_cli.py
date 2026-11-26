import pytest

from scripts.memory_cli import run


def test_memory_cli_add_and_search_persists(tmp_path) -> None:
    database = tmp_path / "memory.sqlite"
    saved = run(["add", "--content", "AAROHAN stores memories locally.", "--source", "project docs"], database_path=database)
    found = run(["search", "memories locally"], database_path=database)
    assert saved["verification_status"] == "unverified"
    assert found["results"][0]["id"] == saved["id"]


def test_memory_cli_verified_add_requires_source(tmp_path) -> None:
    with pytest.raises(ValueError, match="Verified memories require a source"):
        run(["add", "--content", "A fact.", "--verified-by", "reviewer"], database_path=tmp_path / "memory.sqlite")


def test_memory_cli_can_filter_search_by_source(tmp_path) -> None:
    database = tmp_path / "memory.sqlite"
    first = run(["add", "--content", "The model uses causal attention.", "--source", "architecture.md"], database_path=database)
    run(["add", "--content", "The model uses rotary positions.", "--source", "notes.txt"], database_path=database)
    found = run(["search", "model", "--source-contains", "architecture.md"], database_path=database)
    assert [record["id"] for record in found["results"]] == [first["id"]]


def test_memory_cli_vector_search_and_graph_commands(tmp_path) -> None:
    database = tmp_path / "memory.sqlite"
    # 1. Add procedural memory
    run(["add", "--content", "Steps to compute gradient norm clipping.", "--type", "procedural", "--source", "manual"], database_path=database)

    # 2. Vector search
    v_res = run(["search", "gradient clipping", "--mode", "vector"], database_path=database)
    assert len(v_res["results"]) >= 1

    # 3. Entity add & list
    run(["entity-add", "RMSNorm", "--type", "normalization", "--description", "Layer normalization"], database_path=database)
    ent_list = run(["entity-list"], database_path=database)
    assert any(e["name"] == "RMSNorm" for e in ent_list["entities"])

    # 4. Relation add & query
    run(["entity-add", "Transformer", "--type", "architecture"], database_path=database)
    run(["relation-add", "RMSNorm", "Transformer", "--relation", "part_of"], database_path=database)
    graph_res = run(["graph-query", "Transformer"], database_path=database)
    assert len(graph_res["relations"]) >= 1

    # 5. Consolidation
    cons_res = run(["consolidate"], database_path=database)
    assert "episodic_records_inspected" in cons_res
