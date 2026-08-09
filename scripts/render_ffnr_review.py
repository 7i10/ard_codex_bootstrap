#!/usr/bin/env python3
# ruff: noqa: E501
"""Render a local, role-blind HTML reviewer for the FFNR CIFAR image panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

CONTRACT = "ffnr_strong_blinded_candidates_v2"
PUBLIC_ROW_KEYS = {"class_id", "image_path", "panel_index", "run_label", "sample_id"}
FORBIDDEN_KEYS = {
    "outcome",
    "score",
    "teacher",
    "target",
    "control",
    "risk",
    "taxonomy",
    "persistent",
    "wrong",
    "correctness",
}


class ReviewPanelError(ValueError):
    """Raised when a panel is not safe to expose to a human reviewer."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_panel(manifest_path: Path) -> tuple[dict[str, Any], str]:
    """Validate the public manifest and return it with its content hash."""
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewPanelError("review manifest is unreadable") from exc
    if not isinstance(raw, dict) or raw.get("contract") != CONTRACT:
        raise ReviewPanelError("review manifest contract is not the role-blind FFNR contract")
    if raw.get("contains_outcome_or_score_or_teacher_state") is not False:
        raise ReviewPanelError("review manifest is not marked free of diagnostic state")
    rows = raw.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ReviewPanelError("review manifest has no rows")
    seen: set[int] = set()
    root = manifest_path.parent.resolve()
    for row in rows:
        if not isinstance(row, dict) or set(row) != PUBLIC_ROW_KEYS:
            raise ReviewPanelError("review row schema contains a non-public field")
        panel_index, sample_id, class_id, image_path, run_label = (
            row.get("panel_index"),
            row.get("sample_id"),
            row.get("class_id"),
            row.get("image_path"),
            row.get("run_label"),
        )
        if not isinstance(panel_index, int) or isinstance(panel_index, bool) or panel_index in seen:
            raise ReviewPanelError("panel indices must be unique integers")
        if not isinstance(sample_id, int) or isinstance(sample_id, bool) or sample_id < 0:
            raise ReviewPanelError("sample IDs must be non-negative integers")
        if not isinstance(class_id, int) or isinstance(class_id, bool) or not 0 <= class_id <= 9:
            raise ReviewPanelError("CIFAR-10 class IDs are invalid")
        if not isinstance(run_label, str) or not run_label:
            raise ReviewPanelError("run label is missing")
        if not isinstance(image_path, str) or not image_path or Path(image_path).is_absolute():
            raise ReviewPanelError("image paths must be relative")
        image = (root / image_path).resolve()
        if root not in image.parents or not image.is_file():
            raise ReviewPanelError("review image is missing or escapes the manifest directory")
        lowered = {str(key).lower() for key in row}
        if lowered & FORBIDDEN_KEYS:
            raise ReviewPanelError("review row contains a diagnostic key")
        seen.add(panel_index)
    return raw, _sha256(manifest_path)


def _html_data(manifest: dict[str, Any], manifest_sha256: str, output_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    root = manifest_path.parent.resolve()
    output_root = output_path.parent.resolve()
    data: list[dict[str, Any]] = []
    for row in manifest["rows"]:
        image = (root / row["image_path"]).resolve()
        data.append(
            {
                "panelIndex": row["panel_index"],
                "sampleId": row["sample_id"],
                "classId": row["class_id"],
                "runLabel": row["run_label"],
                "imageSrc": os.path.relpath(image, output_root),
            }
        )
    return data


def render_html(manifest_path: Path, output_path: Path, *, force: bool = False) -> str:
    """Create an atomic HTML reviewer and return the manifest SHA-256."""
    manifest_path = manifest_path.resolve()
    output_path = output_path.resolve()
    manifest, manifest_sha256 = load_panel(manifest_path)
    if output_path.exists() and not force:
        raise ReviewPanelError("refusing to overwrite an existing reviewer; use --force")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _html_data(manifest, manifest_sha256, output_path, manifest_path)
    embedded = json.dumps({"manifestSha256": manifest_sha256, "rows": rows}, ensure_ascii=False, separators=(",", ":"))
    # JSON is placed in a script data block; prevent a row string from ending it.
    embedded = embedded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    document = _DOCUMENT.replace("__PANEL_DATA__", embedded)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(document)
    os.replace(temporary, output_path)
    return manifest_sha256


_DOCUMENT = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FFNR blind image review</title>
<style>
:root { color-scheme: light; font-family: system-ui, sans-serif; }
body { margin: 0; background: #f5f6f8; color: #17202a; }
header { position: sticky; top: 0; z-index: 2; background: #17202a; color: white; padding: 12px 18px; }
header h1 { margin: 0 0 5px; font-size: 1.15rem; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
button, select, input, textarea { font: inherit; }
button { border: 1px solid #9aa4ad; border-radius: 5px; background: white; padding: 6px 10px; cursor: pointer; }
button:hover { background: #e8edf1; }
main { display: grid; grid-template-columns: minmax(280px, 520px) minmax(280px, 1fr); gap: 18px; max-width: 1120px; margin: 18px auto; padding: 0 16px 40px; }
.card { background: white; border: 1px solid #d7dce1; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px #0000000d; }
.image-card { text-align: center; }
#panel-image { width: min(100%, 480px); image-rendering: pixelated; border: 1px solid #d7dce1; background: #ddd; }
.meta { display: flex; justify-content: space-between; gap: 8px; margin: 10px 0; color: #4b5964; font-size: .95rem; }
.field { margin: 14px 0; }
.field > label, legend { font-weight: 650; }
fieldset { border: 0; padding: 0; margin: 0; }
.choices { display: grid; gap: 7px; margin-top: 8px; }
.choices label { display: flex; gap: 8px; align-items: flex-start; padding: 7px; border: 1px solid #d7dce1; border-radius: 5px; }
.choices label:has(input:checked) { border-color: #2563eb; background: #eef4ff; }
textarea { width: 100%; box-sizing: border-box; min-height: 90px; padding: 7px; border: 1px solid #aeb7bf; border-radius: 5px; }
.progress { height: 9px; background: #dce2e7; border-radius: 99px; overflow: hidden; margin: 7px 0; }
#progress-bar { height: 100%; width: 0; background: #2563eb; transition: width .15s; }
.hint { color: #4b5964; font-size: .9rem; line-height: 1.45; }
.warning { background: #fff7df; border-left: 4px solid #d79b00; padding: 10px; }
.nav { display: flex; justify-content: space-between; gap: 8px; margin-top: 14px; }
@media (max-width: 760px) { main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>FFNR CIFAR盲検画像レビュー</h1>
  <div class="toolbar">
    <button id="previous">← 前</button><button id="next">次 →</button>
    <button id="export">判定をJSON保存</button>
    <label>JSON読込 <input id="import" type="file" accept="application/json"></label>
    <button id="clear">保存済み判定を消去</button>
    <label>表示 <select id="filter"><option value="all">全て</option><option value="unreviewed">未確認のみ</option><option value="flagged">要確認のみ</option></select></label>
  </div>
  <div id="status" aria-live="polite"></div>
  <div class="progress"><div id="progress-bar"></div></div>
</header>
<main>
  <section class="card image-card">
    <img id="panel-image" alt="CIFAR-10 review image">
    <div class="meta"><span id="panel-counter"></span><span id="panel-run"></span></div>
    <div class="meta"><span id="panel-sample"></span><span id="panel-label"></span></div>
    <p class="hint">画像と提示ラベルだけを見て判定してください。学習時の診断状態や予測スコアは表示していません。</p>
  </section>
  <section class="card">
    <div class="warning"><strong>分類の基準</strong><br>
      <b>clear_match</b>: 提示ラベルの対象に明確に見える。<br>
      <b>ambiguous</b>: 小画像・遮蔽・構図などで判断不能だが、明確な誤りとも言えない。<br>
      <b>possible_label_error</b>: 提示ラベルと画像内容が明らかに食い違う疑いがある。<br>
      <b>ungradable</b>: 画像破損・極端な判別不能などで判定できない。
    </div>
    <div class="field"><fieldset><legend>画像と提示ラベルの一致</legend><div class="choices" id="classification"></div></fieldset></div>
    <div class="field"><fieldset><legend>判定への確信度</legend><div class="choices" id="confidence"></div></fieldset></div>
    <div class="field"><label for="proposed-class">誤りを疑う場合の候補クラス（任意）</label><select id="proposed-class"><option value="">記録しない</option></select></div>
    <div class="field"><label for="notes">メモ（任意）</label><textarea id="notes" placeholder="例: 車体の一部しか見えない / ラベル候補はbird"></textarea></div>
    <div class="nav"><button id="save-next">保存して次へ</button><button id="flag">要確認にする</button></div>
  </section>
</main>
<script id="panel-data" type="application/json">__PANEL_DATA__</script>
<script>
const data = JSON.parse(document.getElementById('panel-data').textContent);
const classNames = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'];
const classifications = [
  ['clear_match','clear_match — 明確に一致'],
  ['ambiguous','ambiguous — 曖昧'],
  ['possible_label_error','possible_label_error — ラベル誤り疑い'],
  ['ungradable','ungradable — 判定不能']
];
const confidences = [['high','high — 高'],['medium','medium — 中'],['low','low — 低']];
const stateKey = `ffnr-human-review-v1:${data.manifestSha256}`;
let state = JSON.parse(localStorage.getItem(stateKey) || '{}');
let position = 0;
const byId = id => document.getElementById(id);
const current = () => data.rows[position];
const record = () => state[String(current().panelIndex)] || {};
const setRecord = patch => { state[String(current().panelIndex)] = {...record(), ...patch, updated_at: new Date().toISOString()}; localStorage.setItem(stateKey, JSON.stringify(state)); renderStatus(); };
function options(container, values, name) {
  container.innerHTML = values.map(([value, text]) => `<label><input type="radio" name="${name}" value="${value}">${text}</label>`).join('');
  container.addEventListener('change', event => { if (event.target.checked) setRecord({[name]: event.target.value}); render(); });
}
options(byId('classification'), classifications, 'classification');
options(byId('confidence'), confidences, 'confidence');
for (let i = 0; i < 10; i++) { const option = document.createElement('option'); option.value = String(i); option.textContent = `${i}: ${classNames[i]}`; byId('proposed-class').appendChild(option); }
byId('proposed-class').addEventListener('change', event => setRecord({proposed_class: event.target.value}));
byId('notes').addEventListener('input', event => setRecord({notes: event.target.value}));
function visibleRows() {
  const filter = byId('filter').value;
  return data.rows.filter(row => { const item = state[String(row.panelIndex)] || {}; if (filter === 'unreviewed') return !item.classification; if (filter === 'flagged') return item.flagged; return true; });
}
function renderStatus() {
  const reviewed = data.rows.filter(row => state[String(row.panelIndex)]?.classification).length;
  const flagged = data.rows.filter(row => state[String(row.panelIndex)]?.flagged).length;
  byId('status').textContent = `${reviewed}/${data.rows.length} 件を分類済み / 要確認 ${flagged} 件`;
  byId('progress-bar').style.width = `${(100 * reviewed / data.rows.length).toFixed(1)}%`;
}
function render() {
  const row = current(), item = record();
  byId('panel-image').src = row.imageSrc;
  byId('panel-image').alt = `CIFAR-10 image, provided class ${row.classId}`;
  byId('panel-counter').textContent = `${position + 1} / ${data.rows.length}`;
  byId('panel-run').textContent = `panel ${row.runLabel}`;
  byId('panel-sample').textContent = `sample ID ${row.sampleId}`;
  byId('panel-label').textContent = `提示ラベル: ${row.classId} (${classNames[row.classId]})`;
  document.querySelectorAll('input[name="classification"], input[name="confidence"]').forEach(input => { input.checked = input.value === item[input.name]; });
  byId('proposed-class').value = item.proposed_class || '';
  byId('notes').value = item.notes || '';
  byId('flag').textContent = item.flagged ? '要確認を解除' : '要確認にする';
  renderStatus();
}
function move(delta) { const rows = visibleRows(); const index = rows.findIndex(row => row.panelIndex === current().panelIndex); if (!rows.length) return; const next = rows[Math.max(0, Math.min(rows.length - 1, index + delta))]; position = data.rows.findIndex(row => row.panelIndex === next.panelIndex); render(); }
byId('previous').onclick = () => move(-1); byId('next').onclick = () => move(1);
byId('save-next').onclick = () => { if (!record().classification) { byId('status').textContent = '分類を選択してから保存してください'; return; } move(1); };
byId('flag').onclick = () => setRecord({flagged: !record().flagged});
byId('filter').onchange = () => { const rows = visibleRows(); if (rows.length) position = data.rows.findIndex(row => row.panelIndex === rows[0].panelIndex); render(); };
byId('clear').onclick = () => { if (confirm('このmanifestに対するブラウザ保存を消去しますか？')) { state = {}; localStorage.removeItem(stateKey); render(); } };
byId('export').onclick = () => { const payload = {schema_version: 1, contract: 'ffnr_human_review_v1', manifest_sha256: data.manifestSha256, exported_at: new Date().toISOString(), rows: state}; const blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'}); const anchor = document.createElement('a'); anchor.href = URL.createObjectURL(blob); anchor.download = `ffnr-human-review-${data.manifestSha256.slice(0,12)}.json`; anchor.click(); URL.revokeObjectURL(anchor.href); };
byId('import').onchange = event => { const file = event.target.files[0]; if (!file) return; const reader = new FileReader(); reader.onload = () => { try { const payload = JSON.parse(reader.result); if (payload.manifest_sha256 !== data.manifestSha256 || payload.contract !== 'ffnr_human_review_v1' || typeof payload.rows !== 'object') throw new Error('manifest mismatch'); state = payload.rows; localStorage.setItem(stateKey, JSON.stringify(state)); render(); } catch (error) { alert(`JSONを読み込めません: ${error.message}`); } }; reader.readAsText(file); };
document.addEventListener('keydown', event => { if (event.target.matches('textarea, input, select')) return; if (event.key === 'ArrowRight') move(1); if (event.key === 'ArrowLeft') move(-1); });
render();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="overwrite an existing HTML reviewer")
    args = parser.parse_args(argv)
    manifest_sha256 = render_html(args.manifest, args.output, force=args.force)
    print(f"manifest_sha256={manifest_sha256}")
    print(f"review_html={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReviewPanelError as exc:
        raise SystemExit(str(exc)) from exc
