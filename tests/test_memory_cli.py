from scripts.memory_cli import run


def test_memory_cli_add_and_search_persists(tmp_path) -> None:
    database = tmp_path / "memory.sqlite"
    saved = run(["add", "--content", "AAROHAN stores memories locally.", "--source", "project docs"], database_path=database)
    found = run(["search", "memories locally"], database_path=database)
    assert saved["verification_status"] == "unverified"
    assert found["results"][0]["id"] == saved["id"]


def test_memory_cli_verified_add_requires_source(tmp_path) -> None:
    import pytest

    with pytest.raises(ValueError, match="Verified memories require a source"):
        run(["add", "--content", "A fact.", "--verified-by", "reviewer"], database_path=tmp_path / "memory.sqlite")


def test_memory_cli_can_filter_search_by_source(tmp_path) -> None:
    database = tmp_path / "memory.sqlite"
    first = run(["add", "--content", "The model uses causal attention.", "--source", "architecture.md"], database_path=database)
    run(["add", "--content", "The model uses rotary positions.", "--source", "notes.txt"], database_path=database)
    found = run(["search", "model", "--source-contains", "architecture.md"], database_path=database)
    assert [record["id"] for record in found["results"]] == [first["id"]]

