import pytest

from aarohan.memory import MemoryStore


def test_memory_persists_and_searches_with_source_metadata(tmp_path) -> None:
    database = tmp_path / "memory.sqlite"
    first_session = MemoryStore(database)
    saved = first_session.add("AAROHAN uses a local Transformer model.", source="project notes")
    assert saved["verification_status"] == "unverified"

    second_session = MemoryStore(database)
    matches = second_session.search("local Transformer")
    assert len(matches) == 1
    assert matches[0]["id"] == saved["id"]
    assert matches[0]["source"] == "project notes"


def test_verified_memory_requires_source_and_reviewer(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    with pytest.raises(ValueError, match="require a source"):
        store.add("A sourced fact.", verification_status="verified", verified_by="reviewer")
    verified = store.add("A sourced fact.", source="manual page", verification_status="verified", verified_by="reviewer")
    assert verified["verification_status"] == "verified"
    assert len(store.search("sourced fact", verified_only=True)) == 1


def test_sensitive_memory_is_blocked_at_storage_layer(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    with pytest.raises(ValueError, match="Sensitive memories are blocked"):
        store.add("private secret", privacy_classification="sensitive")


def test_duplicate_can_be_promoted_only_with_verification_evidence(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    unverified = store.add("The value is twelve.")
    duplicate = store.add(" The value is twelve. ", source="checked record", verification_status="verified", verified_by="adult reviewer")
    assert duplicate["id"] == unverified["id"]
    assert duplicate["verification_status"] == "verified"
    assert duplicate["source"] == "checked record"


def test_correction_marks_old_record_and_delete_removes_search_result(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    old = store.add("The count is eleven.", source="old note", verification_status="verified", verified_by="reviewer")
    corrected = store.correct(old["id"], "The count is twelve.", source="checked source", corrected_by="reviewer")
    assert store.get(old["id"])["verification_status"] == "superseded"
    assert store.get(old["id"])["superseded_by"] == corrected["id"]
    assert [row["id"] for row in store.search("count", verified_only=True)] == [corrected["id"]]
    assert store.delete(corrected["id"]) is True
    assert store.search("twelve") == []

