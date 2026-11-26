import json

import pytest

from scripts.validate_corpus import validate


def test_current_corpus_passes_validation() -> None:
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    result = validate(project_root / "data" / "corpora" / "original")
    assert result["ready"] is True
    assert result["train_paragraphs"] == 10
    assert result["validation_paragraphs"] == 5


def test_valid_original_corpus_passes_and_reports_hashes(tmp_path) -> None:
    (tmp_path / "corpus.txt").write_text("Original training paragraph.\nAnother training paragraph.\n", encoding="utf-8")
    (tmp_path / "validation.txt").write_text("Separate validation paragraph.\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "dataset_name": "test corpus",
        "author": "Project owner",
        "created_date": "2026-11-25",
        "license_or_permission": "Author grants AAROHAN local research and training use",
        "status": "approved_for_local_experiment",
        "privacy_reviewed": True,
        "third_party_text_included": False,
        "train_file": "corpus.txt",
        "validation_file": "validation.txt",
    }), encoding="utf-8")
    result = validate(tmp_path)
    assert result["ready"] is True
    assert result["train_sha256"]
    assert result["validation_sha256"]


def test_invalid_manifest_json_has_location_hint(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text('{"author": "missing comma"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON.*line 2, column 1"):
        validate(tmp_path)

