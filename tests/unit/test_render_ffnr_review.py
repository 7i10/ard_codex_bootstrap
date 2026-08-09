from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.render_ffnr_review import ReviewPanelError, load_panel, render_html


def _manifest(root: Path, *, extra: dict[str, object] | None = None) -> Path:
    image_dir = root / "images"
    image_dir.mkdir()
    (image_dir / "one.png").write_bytes(b"PNG")
    payload: dict[str, object] = {
        "contract": "ffnr_strong_blinded_candidates_v2",
        "contains_outcome_or_score_or_teacher_state": False,
        "rows": [{"class_id": 3, "image_path": "images/one.png", "panel_index": 1, "run_label": "L2", "sample_id": 7}],
    }
    payload.update(extra or {})
    path = root / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_render_html_embeds_only_public_rows_and_manifest_hash(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "review.html"
    digest = render_html(manifest, output)
    text = output.read_text(encoding="utf-8")
    assert digest
    assert "ffnr-human-review-v1" in text
    assert "images/one.png" in text
    assert "teacher" not in text.lower()
    assert "target/control" not in text.lower()
    assert "teacher_correct" not in text.lower()
    assert "possible_label_error" in text
    embedded = re.search(r'<script id="panel-data" type="application/json">(.*?)</script>', text, re.S)
    assert embedded is not None
    assert json.loads(embedded.group(1))["manifestSha256"] == digest
    with pytest.raises(ReviewPanelError, match="overwrite"):
        render_html(manifest, output)


def test_render_standalone_embeds_image_data(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "standalone.html"
    render_html(manifest, output, embed_images=True)
    text = output.read_text(encoding="utf-8")
    assert "data:image/png;base64,UE5H" in text
    assert 'imageSrc":"images/one.png' not in text


def test_render_rejects_diagnostic_row_fields(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        extra={
            "rows": [
                {
                    "class_id": 3,
                    "image_path": "images/one.png",
                    "panel_index": 1,
                    "run_label": "L2",
                    "sample_id": 7,
                    "teacher_correct": False,
                }
            ]
        },
    )
    with pytest.raises(ReviewPanelError, match="non-public"):
        load_panel(manifest)


def test_render_rejects_missing_or_escaping_image(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        extra={
            "rows": [
                {
                    "class_id": 3,
                    "image_path": "../outside.png",
                    "panel_index": 1,
                    "run_label": "L2",
                    "sample_id": 7,
                }
            ]
        },
    )
    with pytest.raises(ReviewPanelError, match="missing or escapes"):
        load_panel(manifest)
