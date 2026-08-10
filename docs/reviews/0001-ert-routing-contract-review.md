# 0001 — ERT routing scientific-contract review

Date: 2026-08-11  
Reviewed revision: `2f6224127cfb7ebc060cd32adc9dba5b7e176394`  
Scope: ERT state overlay and online-routing proxy; no training, attack replay,
official test, or AutoAttack.

## Findings before corrective pass

### Student state semantics

- Status: `INTENTIONAL`
- Severity: P1 candidate, not an unbounded bug
- Files: `ert_state_overlay.py`, `ert_online_routing_proxy.py`,
  `configs/analysis/ert_state_overlay_v1.yaml`, plan 0035/0036
- Evidence: overlay v1 explicitly defines the historical pilot partition
  (`S1=adv-correct`, `S2=adv-wrong and clean-wrong`,
  `S3=clean-correct and adv-wrong`), while the routing proxy defines the
  canonical diagnostic partition (`S1=safe correct`, `S2=fragile correct`,
  `S3=wrong`). The overlay report and plan label the former as the pilot state;
  no consumer silently joins its labels to the proxy labels.
- Scientific impact: old v1 masks are not interchangeable with canonical
  routing states. Silent migration would invalidate historical mask hashes.
- Action: preserve v1 as legacy evidence, document the distinction, and do not
  launch Stage A. A future canonical overlay must be a new version.
- Artifact impact: none; historical v1 artifacts remain immutable.

### Online attack identity in the proxy

- Status: `CONFIRMED`
- Severity: P1
- Files: `ert_online_routing_proxy.py`, `_online_panel` call chain
- Evidence: proxy calls `ffnr_forecasting._online_panel()` but does not call
  the canonical `_validate_online_attack()` used by state mechanism/overlay.
  A mutated KL-PGD10 lineage can therefore reach proxy diagnostics.
- Scientific impact: online Student history could be interpreted under the
  wrong objective, steps, epsilon, target, or temperature.
- Action: reuse the canonical validator and add mutation tests.
- Artifact impact: old final9 report is historical; corrected output receives
  a new output directory.

### Hybrid state naming

- Status: `CONFIRMED`
- Severity: P1
- Files: `ert_online_routing_proxy.py`, proxy report/docs
- Evidence: `state_cells["online"]` and `source="online"` combine online
  Student state with Teacher state derived from strong CE-PGD20 replay. The
  human report warns about this, but the machine schema does not.
- Scientific impact: downstream consumers can mistake a hybrid cell or
  transition for an online joint observation.
- Action: version the proxy schema and name fields explicitly as
  `online_student__ce20_teacher` and `online_student`; preserve final9.
- Artifact impact: new final10 report; no overwrite.

### Clean-harm denominator

- Status: `CONFIRMED`
- Severity: P2
- Files: `ert_state_overlay.effect()`
- Evidence: `clean_harm_rate` is clean-harm count divided by cohort size.
  The conditional denominator `control_clean_correct_count` is not reported.
- Scientific impact: readers may confuse `P(harm | cohort)` with
  `P(treatment clean-wrong | control clean-correct)`.
- Action: retain legacy field and add explicit cohort and conditional fields;
  document robust rescue/harm denominators similarly.
- Artifact impact: additive fields in a new overlay output only.

### Quantile-transfer wording

- Status: `CONFIRMED`
- Severity: P2
- Files: `ert_online_routing_proxy.py`
- Evidence: online `margin_risk=(1-margin)/2`; the implementation selects the
  highest-risk q10, while metadata says “lower-risk q10”.
- Scientific impact: implementation and scientific description disagree.
- Action: correct wording only; preserve values.

### State-level proxy metrics

- Status: `CONFIRMED`
- Severity: P2
- Files: proxy report
- Evidence: confusion counts exist, but state-wise support, precision, recall,
  and F1 are absent from the human-facing output.
- Scientific impact: overall agreement hides minority S2 behavior.
- Action: add per-state metrics derived from the same confusion counts; no
  threshold tuning.

### Equal-rank composite population

- Status: `NEEDS HUMAN DECISION`
- Severity: P2 candidate
- Evidence: `equal_rank_score()` ranks over the supplied full ID universe;
  proxy constructs the composite before restricting Top-K evaluation to the
  strong current-correct cohort. No preregistered statement establishes
  whether this is a global deployable score or a conditional cohort score.
- Scientific impact: changing the population changes rankings and results.
- Action: no change in this pass; report current behavior and require a new
  version/decision before changing it.

### Factorial CE-PGD20 availability

- Status: `CONFIRMED`
- Severity: P2 traceability
- Evidence: configured L2/L4 CE-PGD20 raw paths do not exist locally or in the
  checked Ferret results directory. Existing summary docs are historical and
  cannot substitute for hash-bound Parquet lineage.
- Scientific impact: complete CE/KL × PGD10/20 raw comparison is unavailable.
- Action: preserve `unavailable`; correct documentation and do not replay in
  this contract-review pass.

### Partial output directories

- Status: `CONFIRMED`
- Severity: P2
- Evidence: proxy and overlay create the final directory before all inputs and
  outputs validate; an exception can leave a directory that blocks retry.
- Scientific impact: partial output can be mistaken for a failed-but-owned
  artifact and can cause unsafe manual reuse.
- Action: write to a sibling staging directory and atomically rename only on
  successful completion; add cleanup/non-overwrite tests.
- Artifact impact: no existing output is modified.

## Explicitly not found

- No attack weakening, official-test leakage, or new training path was found.
- The canonical online attack validator already exists; it is reused rather
  than duplicated.
- Historical overlay v1 hashes are not overwritten.

## Stop rule

Stage A remains blocked. This pass does not choose KD temperature, clean-CE or
clean-KD coefficients, a threshold, a route, or a new training arm.

## Resolution

Corrective commit: `bd6255d` (`Correct ERT routing scientific contracts`).

- Online attack validation now reuses `_validate_online_attack()` and rejects
  CE-PGD10, KL-PGD20, epsilon, target, and temperature mutations. Focused tests
  cover all five mutations.
- The proxy is now contract v2. Hybrid cells are named
  `online_student__ce20_teacher` and transitions are named
  `online_student` or `ce20_oracle_student`. Per-state support, precision,
  recall, and F1 are additive diagnostics.
- Quantile metadata now says `highest-margin-risk q10`; numeric state values are
  unchanged.
- Overlay effects retain legacy cohort-level fields and add explicit
  `control_clean_correct_count`, conditional clean-harm rate, and analogous
  robust rescue/harm denominators.
- Proxy and overlay now write through sibling staging directories and rename
  atomically. Focused failure tests confirm partial directories are removed.
- The historical `ert_state_overlay_v1` partition and artifacts were not
  rewritten. Its legacy semantics are explicitly documented; canonical state
  semantics remain in the proxy v2 contract.
- The current checkout was checked for configured CE-PGD20 factorial raw
  Parquets on Hamster and the available Ferret result directory; neither L2 nor
  L4 CE-PGD20 raw path exists. The unavailable status is retained.
- Equal-rank composite population was not changed: its intended global versus
  conditional population is not preregistered and requires a human decision.

## Verification and artifact comparison

- `pytest -q tests/unit/test_ert_online_routing_proxy.py
  tests/unit/test_ert_state_overlay.py tests/unit/test_ffnr_state_mechanism.py`:
  `18 passed`.
- `scripts/verify.py --changed`: selected proxy/overlay/verify-gate tests;
  `9 + 5 + 33 = 47 passed`.
- Ruff passed for all changed Python files. Targeted mypy reported only the
  pre-existing errors in autoattack/signal_audit/teacher_risk_replay/causal
  CE20; no errors originated in the changed routing modules.
- v2 CPU proxy CLI completed from clean `bd6255d` in
  `.cache/analysis/ert-online-routing-proxy-v2-final10/`.
- Corrective overlay CLI completed from the same clean revision in
  `.cache/analysis/ert-state-overlay-v1-review/`.
- Proxy legacy numeric fields (Top-K, cohorts, correlations, confusion counts,
  transitions, and factorial summary) are byte/value-equivalent after mapping
  only the renamed fields. Overlay state-table Parquet bytes and selected mask
  IDs are unchanged; legacy effect fields are equal. New differences are
  additive denominator fields and provenance/output schema.

Stage A remains blocked: there is no unresolved confirmed P1 in this review,
but treatment formulas and the equal-rank population decision are still not
frozen.
