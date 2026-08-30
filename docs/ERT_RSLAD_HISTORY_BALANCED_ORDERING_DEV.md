# ERT / RSLAD History-Minimality and First History-Conditioned Ordering

## Decision at the production gate

The requested CONTROL/HISTORY_BALANCED suffix training was **not launched**.
The pre-launch RNG audit is a scientific fail-closed blocker: the current
training attack assigns its random-start tensor by batch position, so changing
the sample order changes the random-start perturbation received by fixed sample
IDs.  The proposed intervention would therefore change both ordering and
attack randomness.

This report records the completed read-only minimality analysis, exact parent
verification, and the blocker.  No training, endpoint evaluation, coefficient
change, or confirmation seed was run.

## Stage A: P3 minimality

The analysis used the existing format-v3 epoch-boundary `SampleStateStore`
records.  The four preregistered P3 quantities were audited from the source:

- H1: `robust_correct_count / seen` (correctness frequency)
- H2: `margin_ema` (EMA decay `0.9`)
- H3: `forgetting_count / seen` (forgetting rate)
- H4: `current_correct_streak / seen` (correct-streak rate)

The target was the preregistered late failure rate
`1 - (hits_199 - hits_149) / (seen_199 - seen_149)`, with cutoff 99.  A single
Ridge model (`alpha=1.0`) was fit on pooled dev-1/dev-2 samples after
development-only standardization.  Confirmation rows were evaluated only
after the subset decision.

| candidate | features | dev-1 Spearman | dev-2 Spearman | dev mean |
|---|---|---:|---:|---:|
| H1 | correctness frequency | 0.835310 | 0.834249 | 0.834780 |
| H2 | margin EMA | 0.836990 | 0.836149 | 0.836570 |
| H3 | forgetting rate | -0.075181 | -0.075572 | -0.075376 |
| H4 | correct streak rate | 0.712954 | 0.710486 | 0.711720 |
| H5 | correctness frequency + margin EMA | 0.844755 | 0.843759 | 0.844257 |
| H6 | correctness frequency + forgetting rate | 0.828572 | 0.827178 | 0.827875 |
| H7 | correctness frequency + correct streak rate | 0.827780 | 0.826999 | 0.827389 |
| H8 | correctness frequency + margin EMA + forgetting rate | 0.845428 | 0.844458 | 0.844943 |
| H9 | full P3 | 0.845376 | 0.844411 | 0.844894 |

H9 is the reference.  H2 is within the preregistered mean tolerance
(`0.836570 >= 0.844894 - 0.01`) and both per-seed tolerances, and is the
smallest eligible candidate.  Therefore the frozen minimal predictor is:

```text
selected subset: H2 = margin_ema
Ridge alpha: 1.0
pooled fit: dev-1 + dev-2
predictor artifact: docs/experiments/ert_rslad_history_minimality_v1.json
predictor SHA-256: b6be23e7eaa31dd300ed21efb84746bba2270a4a044096c86d3be035c8abebbd
```

The post-hoc confirmation Spearman for H2 was `0.836420`, `0.837836`, and
`0.836549` for confirm-a/b/c (mean `0.836935`).  H9 confirmation mean was
`0.844550`; these values did not alter selection.

## Exact I100 parents

Both requested epoch-99 I100 parents are present as the existing stagewise
materializations.  The payload is at epoch 99 (end), with global step 35,200,
and the scheduler has already advanced to `last_epoch=100` with next learning
rate `0.01`.

| seed | path | SHA-256 | config hash |
|---|---|---|---|
| dev-1 | `.../ert-rslad-stagewise-v1/seed1/s100/epoch-100.pt` | `360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835` | `43b58e2fe01ae99bd9a5afb9970243ba2720a1605aac3225535511684212c3ff` |
| dev-2 | `.../ert-rslad-stagewise-v1/seed2/s100/epoch-100.pt` | `bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7` | `617e0ae4bce233dde1a822948e044179ff319554a04dd719827616e83e2ce1c0` |

These byte hashes match the parent hashes recorded in the accepted I100
fork lineage.  No parent was substituted.

## RNG-order coupling audit

The audit is recorded in
`docs/experiments/ert_rslad_order_rng_audit_v1.json`.

The current implementation has the following contract:

```text
Trainer._attack_generator(): one newly seeded generator per training batch
seed = train_seed + 1000003*global_step + 10007*rank
LinfPGD.generate(): batched delta.uniform_(-1, 1, generator=request.generator)
```

The generator is therefore not keyed by source ID.  A bounded deterministic
reproduction with two four-sample batches changed the order of sample IDs 0
and 1 and changed both fixed samples' initial perturbation tensors.  The audit
decision is `BLOCK_PRODUCTION_ORDERING_ONLY`.

Augmentation is independently source/epoch keyed, but that does not remove the
attack coupling.  A valid ordering-only study requires a separately reviewed
RNG-contract change, such as source/epoch-keyed per-sample random starts (or a
precomputed source-keyed random-start map), plus a new common control.  This
would alter the scientific stochasticity contract and must not be introduced
silently in this task.

## Stage B status

| item | status |
|---|---|
| CONTROL suffixes | not launched |
| HISTORY_BALANCED suffixes | not launched |
| CE-PGD20 endpoints | not launched |
| W&B production runs | none |
| telemetry / aggregation | not applicable |
| new seed / confirmation | not run |

The blocked manifest is
`docs/experiments/ert_rslad_history_ordering_manifest_v1.json`.  The next
action is a human decision on whether to approve a new source-keyed attack RNG
contract and redesign the paired control.  Until then, the requested
ordering-only causal intervention cannot be interpreted cleanly.
