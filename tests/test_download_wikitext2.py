import json

import pytest

from scripts.download_wikitext2 import check_human_review


def test_wikitext_download_is_blocked_until_human_review(tmp_path) -> None:
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps({
        "training_allowed": False,
        "license_review": {"status": "needs_human_review"},
    }), encoding="utf-8")
    with pytest.raises(PermissionError, match="not approved"):
        check_human_review(review_path)


def test_wikitext_review_requires_adult_role_and_decision(tmp_path) -> None:
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps({
        "training_allowed": True,
        "license_review": {
            "status": "reviewed_approved",
            "reviewer": {"role": "project_owner", "name": "A", "review_date": "2026-11-25"},
            "decision_note": "reviewed",
        },
    }), encoding="utf-8")
    with pytest.raises(PermissionError, match="parent/guardian"):
        check_human_review(review_path)

