"""Re-judge the e114 rows of the evidence reclassification against the measured floor.

`docs/EVIDENCE_RECLASSIFICATION.md` judged these rows against a bracket of 0.25
to 0.50 pp with 0.40 pp as a working value, because the post-decay floor had
never been measured.  Plan 0092 measured it: 0.092 pp at epoch 114.

Only rows measured at epoch 114 on the held-out CE-PGD20 endpoint are eligible.
The measurement covers e104, e109 and e114 of the I100 design and nothing else,
so an e199 row, an AutoAttack row and a trajectory-AUC row are all out of scope
however tempting the extrapolation looks.

Emits one dict and renders the report from it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
FLOOR_RECORD = ROOT / "docs/experiments/ard_post_decay_floor_v1.json"
REPORT = ROOT / "docs/POST_DECAY_FLOOR_RECLASSIFICATION.md"
ALPHA = 0.05
POWER = 0.80

# (row, arm, per-seed held-out effects in pp) -- transcribed from the
# reclassification table's "observed effect" column.
ELIGIBLE = [
    ("E1", "DPM (campaign 1 of the same arm)", (0.08, 0.12)),
    ("E1", "OS-PMP (campaign 2 of the same arm)", (0.14, 0.20)),
    ("E2", "D-BDD (campaign 1)", (0.04, 0.20)),
    ("E2", "OS-DBDP (campaign 2)", (0.14, 0.06)),
    ("E4", "SBF", (0.160, 0.040)),
    ("E5", "TPFM @S2T1", (0.220, 0.040)),
    ("E6", "PILOT_S3_T1_WEAK_ADVCE", (0.22, 0.04)),
    ("E6", "CLEAN_WRONG_PLAIN_ADVCE", (0.44, 0.10)),
    ("E6", "CLEAN_WRONG_A7_MARGIN_ONLY", (0.44, 0.08)),
]

OUT_OF_SCOPE = [
    ("A1", "held-out CE-PGD20 at e199", "a different horizon, 85 epochs past the ones measured"),
    ("A2", "held-out CE-PGD20 at e199", "same"),
    ("A3", "held-out CE-PGD20 at e199", "same"),
    ("A4", "last robust at e199 and trajectory AUC", "different horizon; the AUC floor has never been measured at all"),
    (
        "B2",
        "best AutoAttack over a full 200-epoch run",
        "different endpoint and horizon; judged against a floor from its own campaign's random arms",
    ),
    (
        "B3",
        "best validation CE-PGD20, e40 to e199",
        "different horizon; judged against a floor from placebo forks in its own campaign",
    ),
    ("B7", "held-out CE-PGD20 at e199", "a different horizon"),
    ("B8", "descriptor-to-probe-AUC association", "not an accuracy claim"),
    ("E3", "finite versus non-finite training loss", "not an accuracy claim: the observable is divergence"),
    (
        "E7",
        "held-out CE-PGD20 at five long horizons",
        "horizons outside the measured set, not independent of each other",
    ),
    (
        "E8",
        "direct train effect versus held-out effect",
        "a within-run contrast, judged against a separate train two-run gap",
    ),
]


def main() -> int:
    floor = json.load(FLOOR_RECORD.open())
    e114 = floor["horizon_statistics"]["114"]["pooled"]
    sigma_d = e114["sigma_d_pp"]
    df = e114["degrees_of_freedom"]
    t_crit = float(stats.t.ppf(1.0 - ALPHA / 2.0, df))
    mde_k2 = floor["horizon_statistics"]["114"]["minimum_detectable_effect_pp"]["2"]
    significance_k2 = t_crit * sigma_d / math.sqrt(2.0)

    rows = []
    for row, arm, effects in ELIGIBLE:
        mean = sum(effects) / len(effects)
        rows.append(
            {
                "row": row,
                "arm": arm,
                "effects_pp": list(effects),
                "mean_pp": mean,
                "preregistered_pass": all(abs(e) > mde_k2 for e in effects),
                "paired_test_pass": abs(mean) > significance_k2,
                "t_ratio": mean / (sigma_d / math.sqrt(2.0)),
            }
        )

    result = {
        "contract": "ard_post_decay_floor_reclassification_v1",
        "floor_record_sha256": (FLOOR_RECORD.parent / (FLOOR_RECORD.name + ".sha256")).read_text().strip(),
        "sigma_d_pp": sigma_d,
        "degrees_of_freedom": df,
        "old_working_floor_pp": 0.40,
        "preregistered_threshold_pp": mde_k2,
        "paired_test_threshold_pp": significance_k2,
        "eligible": rows,
        "out_of_scope": [{"row": r, "measurement": m, "why": w} for r, m, w in OUT_OF_SCOPE],
    }

    lines: list[str] = []
    add = lines.append
    add("# Re-judging the e114 evidence against the measured floor")
    add("")
    add(
        f"`docs/EVIDENCE_RECLASSIFICATION.md` judged its post-decay rows against a bracket of 0.25 to 0.50 pp "
        f"with **0.40 pp** as the working value, because the floor had never been measured. Plan 0092 measured "
        f"it: **sigma_d = {sigma_d:.3f} pp** at epoch 114 "
        f"(`docs/experiments/ard_post_decay_floor_v1.json`). This re-judges everything that measurement "
        f"legitimately covers."
    )
    add("")
    add("## Scope: five rows, not forty")
    add("")
    add(
        "The table has 38 rows. 21 are pre-decay and were already judged against the separately measured "
        "pre-decay floor of 1.14 to 1.25 pp, which this does not touch. Of the 17 post-decay rows, only "
        f"**{len({r for r, _, _ in ELIGIBLE})} rows covering {len(ELIGIBLE)} arm contrasts** are measured at "
        "epoch 114 on the held-out CE-PGD20 endpoint of the I100 design -- the one thing plan 0092 measured."
    )
    add("")
    add("The other twelve are out of scope, and saying why matters more than the count:")
    add("")
    add("| row | what it measures | why the new floor does not apply |")
    add("| --- | --- | --- |")
    for r, m, w in OUT_OF_SCOPE:
        add(f"| {r} | {m} | {w} |")
    add("")
    add(
        "The temptation is to extrapolate to e199, especially since the floor shrinks with post-decay epochs "
        "(0.159, 0.124, 0.092 pp at e104, e109, e114). That trend makes a smaller e199 floor plausible. "
        "Plausible is not measured, and every one of those rows would move in the direction that favours a "
        "positive verdict, which is exactly when extrapolation should be refused."
    )
    add("")
    add("## The eligible contrasts")
    add("")
    add(
        f"Two thresholds are shown. The **preregistered rule** in `docs/decisions/0002-...` asks whether both "
        f"seeds exceed the minimum detectable effect, {mde_k2:.3f} pp. The **paired test** asks whether the "
        f"two-seed mean exceeds {significance_k2:.3f} pp, which is "
        f"t({df}) = {t_crit:.3f} times sigma_d / sqrt(2)."
    )
    add("")
    add("| row | arm | per seed (pp) | mean (pp) | preregistered rule | paired test |")
    add("| --- | --- | --- | ---: | --- | --- |")
    for item in rows:
        seeds = " / ".join(f"{e:+.2f}" for e in item["effects_pp"])
        pre = "**passes**" if item["preregistered_pass"] else "fails"
        pair = "**passes**" if item["paired_test_pass"] else "fails"
        add(f"| {item['row']} | {item['arm']} | {seeds} | {item['mean_pp']:+.3f} | {pre} | {pair} |")
    add("")
    n_pre = sum(1 for r in rows if r["preregistered_pass"])
    n_pair = sum(1 for r in rows if r["paired_test_pass"])
    add(
        f"Under the preregistered rule: **{n_pre} of {len(rows)}**. Under the paired test: **{n_pair} of {len(rows)}**."
    )
    add("")
    add("## The preregistered rule was mis-specified, and this says so rather than quietly switching")
    add("")
    add(
        "The rule written into the decision packet asks each seed to exceed the minimum detectable effect. "
        "That is not a test. The minimum detectable effect is a property of the *mean* of k blocks at 80% "
        "power; applying it to each block separately is both the wrong statistic and the wrong quantity, and "
        "it is strictly more conservative than the test it was meant to stand in for."
    )
    add("")
    add(
        "The packet also says the rule must not change after the count is known. Both are therefore reported. "
        "**The preregistered column is the answer of record. The paired-test column is exploratory**, because "
        "the rule behind it was fixed after the numbers were visible, and anything it turns up is a candidate "
        "for a future screen rather than a finding."
    )
    add("")
    add("## What actually changed")
    add("")
    if n_pre == 0:
        add(
            "**No verdict changes under the preregistered rule.** A floor four times smaller does not rescue "
            "these arms, because the effects themselves are small: the largest single-seed value among the "
            "eligible contrasts is +0.44 pp and its partner seed is +0.08 pp. The verdicts were UNDERPOWERED "
            "and they remain UNDERPOWERED."
        )
    add("")
    if n_pair:
        add("Exploratory, from the paired test only:")
        add("")
        for item in rows:
            if item["paired_test_pass"]:
                add(
                    f"- **{item['row']} {item['arm']}**: {item['mean_pp']:+.3f} pp, "
                    f"t = {item['t_ratio']:.2f} against a critical value of {t_crit:.2f}. "
                    "Both are transfer arms from the Stage A lineage."
                )
        add("")
        add(
            "These are not results. Two seeds and a threshold chosen after the fact is the exact pattern the "
            "measurement standard exists to stop. What they are is the best-supported candidates if a screen "
            "is ever run on this question with enough blocks."
        )
    add("")
    add("## The number that matters most here is not in the table")
    pm = next(r for r in rows if r["arm"].startswith("OS-PMP"))
    add("")
    add(
        f"`PM(online)` -- plan 0093's declared primary arm -- is row E1's second campaign at "
        f"{pm['mean_pp']:+.3f} pp, against a paired-test threshold of {significance_k2:.3f} pp. "
        f"It misses by {significance_k2 - pm['mean_pp']:.3f} pp."
    )
    add("")
    add(
        "That is the most useful thing this exercise produces. Plan 0093 as designed, at two blocks, sits just "
        "under the line even if its effect is exactly what has been observed twice. Its power should be "
        "re-derived from the measured floor before it starts, not after."
    )
    add("")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "eligible"}, indent=2))
    print(f"\nwrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
