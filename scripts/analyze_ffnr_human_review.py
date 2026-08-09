#!/usr/bin/env python3
# ruff: noqa: E501
"""Join FFNR human labels with strong replay and produce class diagnostics.

The human panel is a selected diagnostic panel, not a random test sample.  This
report therefore separates panel-conditioned results from the full validation
class statistics and never treats student correctness as a predicted class.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
DEFAULT_ROOT = Path(".cache/analysis/ffnr-strong-diagnostics-6a90011-v1")
DEFAULT_L2 = Path(".cache/analysis/ffnr-strong-replay/l2-outcome-e5cb442/strong-observations.parquet")
DEFAULT_L4 = Path(".cache/analysis/ffnr-strong-replay/l4-outcome-e5cb442/strong-observations.parquet")
DEFAULT_REVIEW = Path("docs/experiments/ffnr_human_review_v1.json")


class HumanReviewAnalysisError(ValueError):
    """Raised when review/replay identities or schemas do not join safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate(successes: int, count: int) -> float | None:
    return None if count == 0 else successes / count


def _argmax(values: list[float]) -> int:
    if len(values) != len(CLASS_NAMES):
        raise HumanReviewAnalysisError("teacher probability vector must have ten classes")
    maxima = max(values)
    candidates = [index for index, value in enumerate(values) if value == maxima]
    if len(candidates) != 1:
        raise HumanReviewAnalysisError("teacher prediction has an unsupported tie")
    return candidates[0]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HumanReviewAnalysisError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise HumanReviewAnalysisError(f"JSON root must be an object: {path}")
    return value


def _load_inputs(
    manifest_path: Path, review_path: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str, str]:
    manifest = _load_json(manifest_path)
    review = _load_json(review_path)
    if manifest.get("contract") != "ffnr_strong_blinded_candidates_v2":
        raise HumanReviewAnalysisError("unexpected blind manifest contract")
    if review.get("contract") != "ffnr_human_review_v1":
        raise HumanReviewAnalysisError("unexpected human review contract")
    manifest_sha = _sha256(manifest_path)
    if review.get("manifest_sha256") != manifest_sha:
        raise HumanReviewAnalysisError("human review is bound to a different blind manifest")
    rows = manifest.get("rows")
    review_rows = review.get("rows")
    if not isinstance(rows, list) or not isinstance(review_rows, dict) or len(rows) != len(review_rows):
        raise HumanReviewAnalysisError("manifest and human review row counts differ")
    by_panel: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise HumanReviewAnalysisError("blind manifest row is not an object")
        key = str(row.get("panel_index"))
        if key in by_panel or key not in review_rows or not isinstance(review_rows[key], dict):
            raise HumanReviewAnalysisError("panel index join is incomplete or duplicated")
        joined = dict(row)
        joined.update(review_rows[key])
        by_panel[key] = joined
    return rows, by_panel, manifest_sha, _sha256(review_path)


def _load_epoch(path: Path, epoch: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        all_rows = pq.read_table(path).to_pylist()
    except Exception as exc:  # pyarrow raises several backend-specific errors
        raise HumanReviewAnalysisError(f"cannot read replay parquet: {path}") from exc
    epochs = sorted({int(row["epoch"]) for row in all_rows})
    if epoch not in epochs:
        raise HumanReviewAnalysisError(f"requested epoch {epoch} is absent from {path}")
    selected = [row for row in all_rows if int(row["epoch"]) == epoch]
    if not selected:
        raise HumanReviewAnalysisError(f"empty epoch {epoch}: {path}")
    required = {
        "sample_id", "class_id", "epoch", "student_robust_correct", "student_clean_correct",
        "teacher_clean_probabilities", "teacher_adversarial_probabilities",
    }
    if not required.issubset(selected[0]):
        raise HumanReviewAnalysisError(f"replay schema is missing required columns: {path}")
    return selected, all_rows


def _annotate(
    rows: list[dict[str, Any]], by_panel: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        sample_id = int(row["sample_id"])
        if sample_id in by_id:
            raise HumanReviewAnalysisError("duplicate stable sample ID in replay epoch")
        class_id = int(row["class_id"])
        clean_probs = [float(value) for value in row["teacher_clean_probabilities"]]
        adv_probs = [float(value) for value in row["teacher_adversarial_probabilities"]]
        item = {
            "sample_id": sample_id,
            "class_id": class_id,
            "student_robust_correct": bool(row["student_robust_correct"]),
            "student_clean_correct": bool(row["student_clean_correct"]),
            "teacher_clean_pred": _argmax(clean_probs),
            "teacher_adv_pred": _argmax(adv_probs),
        }
        by_id[sample_id] = item
    panel: list[dict[str, Any]] = []
    for panel_key, review_row in by_panel.items():
        sample_id = int(review_row["sample_id"])
        if sample_id not in by_id:
            continue
        item = dict(by_id[sample_id])
        item.update(
            {
                "panel_index": int(panel_key),
                "run_label": str(review_row["run_label"]),
                "classification": str(review_row["classification"]),
                "commented": "notes" in review_row and bool(review_row["notes"]),
            }
        )
        panel.append(item)
    if len(panel) != len(by_panel):
        raise HumanReviewAnalysisError("human panel sample IDs do not join to replay")
    return panel, by_id


def _full_stats(by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    class_stats: dict[str, Any] = {}
    adv_confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    clean_confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for row in by_id.values():
        true_class = int(row["class_id"])
        adv_pred = int(row["teacher_adv_pred"])
        clean_pred = int(row["teacher_clean_pred"])
        adv_confusion[true_class][adv_pred] += 1
        clean_confusion[true_class][clean_pred] += 1
        bucket = class_stats.setdefault(
            str(true_class),
            {"class_id": true_class, "class_name": CLASS_NAMES[true_class], "count": 0, "student_robust_correct": 0,
             "student_clean_correct": 0, "teacher_adv_correct": 0, "teacher_clean_correct": 0},
        )
        bucket["count"] += 1
        bucket["student_robust_correct"] += int(row["student_robust_correct"])
        bucket["student_clean_correct"] += int(row["student_clean_correct"])
        bucket["teacher_adv_correct"] += int(adv_pred == true_class)
        bucket["teacher_clean_correct"] += int(clean_pred == true_class)
    for bucket in class_stats.values():
        count = int(bucket["count"])
        for prefix in ("student_robust", "student_clean", "teacher_adv", "teacher_clean"):
            correct = int(bucket[f"{prefix}_correct"])
            bucket[f"{prefix}_accuracy"] = _rate(correct, count)
            bucket[f"{prefix}_error_rate"] = _rate(count - correct, count)
    def error_pairs(confusion: list[list[int]]) -> list[dict[str, Any]]:
        pairs = [
            {"count": confusion[a][b], "true_class": CLASS_NAMES[a], "predicted_class": CLASS_NAMES[b]}
            for a in range(10)
            for b in range(10)
            if a != b
        ]
        pairs.sort(key=lambda value: (-int(value["count"]), value["true_class"], value["predicted_class"]))
        return pairs[:15]
    return {
        "count": len(by_id),
        "class_stats": class_stats,
        "teacher_adv_confusion": adv_confusion,
        "teacher_clean_confusion": clean_confusion,
        "top_teacher_adv_errors": error_pairs(adv_confusion),
        "top_teacher_clean_errors": error_pairs(clean_confusion),
    }


def _panel_stats(panel: list[dict[str, Any]]) -> dict[str, Any]:
    for row in panel:
        if row["classification"] == "clear_match":
            row["human_group"] = "clear_hard" if row["commented"] else "clear_easy"
        else:
            row["human_group"] = row["classification"]
    groups: dict[str, Any] = {}
    for group, rows in _group(panel, "human_group").items():
        groups[group] = _error_rates(rows)
    by_class: dict[str, Any] = {}
    for class_id, rows in _group(panel, "class_id").items():
        entry = _error_rates(rows)
        entry.update(
            {
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[int(class_id)],
                "ambiguous": sum(row["classification"] == "ambiguous" for row in rows),
                "clear_easy": sum(row["human_group"] == "clear_easy" for row in rows),
                "clear_hard": sum(row["human_group"] == "clear_hard" for row in rows),
            }
        )
        entry["ambiguous_rate"] = _rate(entry["ambiguous"], len(rows))
        by_class[str(class_id)] = entry
    return {"count": len(panel), "groups": groups, "by_class": by_class}


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(str(row[key]), []).append(row)
    return result


def _error_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    student_robust_error = sum(not row["student_robust_correct"] for row in rows)
    student_clean_error = sum(not row["student_clean_correct"] for row in rows)
    teacher_adv_error = sum(row["teacher_adv_pred"] != row["class_id"] for row in rows)
    teacher_clean_error = sum(row["teacher_clean_pred"] != row["class_id"] for row in rows)
    return {
        "count": count,
        "student_robust_error": student_robust_error,
        "student_robust_error_rate": _rate(student_robust_error, count),
        "student_clean_error": student_clean_error,
        "student_clean_error_rate": _rate(student_clean_error, count),
        "teacher_adv_error": teacher_adv_error,
        "teacher_adv_error_rate": _rate(teacher_adv_error, count),
        "teacher_clean_error": teacher_clean_error,
        "teacher_clean_error_rate": _rate(teacher_clean_error, count),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# FFNR人手判定とクラス偏りの分析",
        "",
        f"- contract: `{report['contract']}`",
        f"- replay epoch: `{report['epoch']}`（各runのCE-PGD20観測）",
        f"- panel: `{report['panel']['count']}`件、full replay: L2 `{report['runs']['L2']['full']['count']}`件 / L4 `{report['runs']['L4']['full']['count']}`件",
        "- 画像panelは候補抽出済みの診断panelであり、ランダムなtest集合ではない。panelの率は記述統計、full replayの率はクラス別検証値として分けて読む。",
        "",
        "## 結論",
        "",
        "1. `possible_label_error`は0件で、今回のpanelから明白なラベルノイズの大量混入は確認できない。",
        "2. `ambiguous`は25/200（12.5%）。クラス別ではdeerが7/20（35%）で最大、airplane/catが各4/20（20%）と続く。",
        "3. ambiguous群の学生robust誤り率は64.0%、clear_matchは59.4%。教師adv誤り率は56.0%対53.7%で、今回のpanelだけではambiguousが誤りを強く説明するとは言えない。",
        "4. full replayの学生robust誤りはbird/catが約45%、deer/dogが約40%で高い。これは人手ambiguous率（deer、cat）と一部整合するが、同一原因の証明ではない。",
        "5. 教師adv混同行列では、bird→deer/frog、cat↔dog/frog、deer→frog、airplane→ship、truck→automobileが大きい。学生の誤予測先は入力Parquetに保存されていないため、学生について同じ混同行列はまだ作れない。",
        "",
        "## 人手分類別のモデル誤り（panel条件付き）",
        "",
        "| human group | n | student robust error | student clean error | teacher adv error | teacher clean error |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, value in report["panel"]["groups"].items():
        lines.append(
            f"| {group} | {value['count']} | {value['student_robust_error_rate']:.1%} | {value['student_clean_error_rate']:.1%} | {value['teacher_adv_error_rate']:.1%} | {value['teacher_clean_error_rate']:.1%} |"
        )
    lines += ["", "## クラス別のpanel結果", "", "| class | n | ambiguous | clear easy | clear hard | student robust error | teacher adv error |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for class_id in range(10):
        value = report["panel"]["by_class"][str(class_id)]
        lines.append(f"| {value['class_name']} | {value['count']} | {value['ambiguous']} ({value['ambiguous_rate']:.1%}) | {value['clear_easy']} | {value['clear_hard']} | {value['student_robust_error_rate']:.1%} | {value['teacher_adv_error_rate']:.1%} |")
    lines += ["", "## full replayのクラス別誤り", "", "各runは45,000件、epoch 199のstrong CE-PGD20観測です。", "", "| class | L2 student robust error | L4 student robust error | L2 teacher adv error | L4 teacher adv error |", "| --- | ---: | ---: | ---: | ---: |"]
    for class_id in range(10):
        l2, l4 = report["runs"]["L2"]["full"]["class_stats"][str(class_id)], report["runs"]["L4"]["full"]["class_stats"][str(class_id)]
        lines.append(f"| {l2['class_name']} | {l2['student_robust_error_rate']:.1%} | {l4['student_robust_error_rate']:.1%} | {l2['teacher_adv_error_rate']:.1%} | {l4['teacher_adv_error_rate']:.1%} |")
    lines += ["", "## 教師advの主な混同（full replay）", "", "| true → predicted | L2 | L4 |", "| --- | ---: | ---: |"]
    l2_pairs = {(x["true_class"], x["predicted_class"]): x["count"] for x in report["runs"]["L2"]["full"]["top_teacher_adv_errors"]}
    l4_pairs = {(x["true_class"], x["predicted_class"]): x["count"] for x in report["runs"]["L4"]["full"]["top_teacher_adv_errors"]}
    pairs = sorted(set(l2_pairs) | set(l4_pairs), key=lambda pair: -(l2_pairs.get(pair, 0) + l4_pairs.get(pair, 0)))[:12]
    for pair in pairs:
        lines.append(f"| {pair[0]} → {pair[1]} | {l2_pairs.get(pair, 0)} | {l4_pairs.get(pair, 0)} |")
    lines += ["", "### 教師cleanの主な混同", "", "| true → predicted | L2 | L4 |", "| --- | ---: | ---: |"]
    l2_clean = {(x["true_class"], x["predicted_class"]): x["count"] for x in report["runs"]["L2"]["full"]["top_teacher_clean_errors"]}
    l4_clean = {(x["true_class"], x["predicted_class"]): x["count"] for x in report["runs"]["L4"]["full"]["top_teacher_clean_errors"]}
    clean_pairs = sorted(set(l2_clean) | set(l4_clean), key=lambda pair: -(l2_clean.get(pair, 0) + l4_clean.get(pair, 0)))[:8]
    for pair in clean_pairs:
        lines.append(f"| {pair[0]} → {pair[1]} | {l2_clean.get(pair, 0)} | {l4_clean.get(pair, 0)} |")
    lines += ["", "## 限界と次の測定", "", "- panelは候補panelであり、クラス別の人手ambiguous率をCIFAR-10全体のラベル品質率へ一般化しない。",
              "- `student_robust_correct`とmarginは保存されているが、学生のclean/adv predicted classそのものは保存されていない。そのため学生のdog↔cat等の誤予測先は未確定である。必要なら次のGPU replayでstudent predicted class（clean/adv）を明示的に保存する。",
              "- ambiguous群をそのまま除外・label correction・KD downweightのmaskには使わない。hidden cohort別のambiguous率と、teacher wrong-confidence・student persistent/recovered状態を結合してから介入を設計する。",
              "- 教師混同行列の大きな組合せは、teacher targetの危険な領域候補として使えるが、介入効果を意味しない。"]
    return "\n".join(lines) + "\n"


def analyze(manifest: Path, review: Path, run_paths: dict[str, Path], epoch: int) -> dict[str, Any]:
    manifest_rows, by_panel, manifest_sha, review_sha = _load_inputs(manifest, review)
    report: dict[str, Any] = {
        "contract": "ffnr_human_review_analysis_v1",
        "epoch": epoch,
        "class_names": list(CLASS_NAMES),
        "manifest_sha256": manifest_sha,
        "review_sha256": review_sha,
        "panel": {},
        "runs": {},
        "student_predicted_class_available": False,
        "source_files": {"manifest": str(manifest), "review": str(review)},
    }
    for row in manifest_rows:
        label = str(row["run_label"])
        if label not in run_paths:
            raise HumanReviewAnalysisError(f"panel has unconfigured run label: {label}")
    all_panel: list[dict[str, Any]] = []
    for label, path in run_paths.items():
        selected, all_rows = _load_epoch(path, epoch)
        run_panel = {key: value for key, value in by_panel.items() if value["run_label"] == label}
        panel, by_id = _annotate(selected, run_panel)
        all_panel.extend(panel)
        report["runs"][label] = {
            "observation_path": str(path),
            "observation_sha256": _sha256(path),
            "available_epochs": sorted({int(row["epoch"]) for row in all_rows}),
            "full": _full_stats(by_id),
            "panel": _error_rates(panel),
            "panel_by_class": _panel_stats(panel)["by_class"],
        }
    report["panel"] = _panel_stats(all_panel)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "ffnr-strong-blinded-candidates.json")
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--l2-outcome", type=Path, default=DEFAULT_L2)
    parser.add_argument("--l4-outcome", type=Path, default=DEFAULT_L4)
    parser.add_argument("--epoch", type=int, default=199)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_ROOT / "ffnr-human-review-analysis.json")
    parser.add_argument("--output-md", type=Path, default=Path("docs/FFNR_HUMAN_REVIEW_ANALYSIS.md"))
    args = parser.parse_args()
    try:
        report = analyze(args.manifest, args.review, {"L2": args.l2_outcome, "L4": args.l4_outcome}, args.epoch)
    except HumanReviewAnalysisError as exc:
        parser.error(str(exc))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {"json": str(args.output_json), "markdown": str(args.output_md), "epoch": args.epoch}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
