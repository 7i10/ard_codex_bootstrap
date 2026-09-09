# ERT/RSLAD I100 Clean-Wrong Long-Horizon Confirmation

## 結論

epoch 114 から 199 までの continuation と、登録済み CE-PGD20 endpoint を確認した。Clean-Wrong **plain AdvCE** は direct train cohort の改善を示すが、held-out robust は dev-1 で -0.30 pp、dev-2 で 0.00 pp だった。**Teacher Positive-Floor Margin (TPFM)** は direct train と held-out robust が両seedで正方向（+0.04/+0.20 pp）だが、改善は小さく、2 seed のため confirmation とは呼ばない。

従って今回の結果は、短期 screen で見えた TPFM の方向と整合するものの、強い性能主張や自動的な次の intervention を正当化しない。CW1–CW6 の判定は **mixed / weak support** とし、e199 extension・新seed・dynamic routing は開始しない。

## Lineage / 実行状態

- scientific source: `c6032f9dc09f938fd0b9fb87379cf16c3f0f26bb`
- original manifest SHA-256: `2ebe1aa63a49b39c868d6f28d9dc19e52c44c5e92a9bb2ea7cdd3960cb040cb4`
- continuation parent: 各armの exact e114 checkpoint（別armのcheckpointや e99 parent は不使用）
- training: KL-PGD10, $\epsilon=8/255$, step $2/255$, 10 steps, random start, Teacher-clean target
- endpoint: CE-PGD20, $\epsilon=8/255$, step $2/255$, 20 steps, random start, eval mode
- endpoint attack identity: `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`
- Teacher: Chen2021LTD WRN34-10, SHA-256 `fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983`
- fixed Clean-Wrong mask（registered action-transfer artifact）: dev-1 9,263 (`6ec6a4aa…`), dev-2 8,709 (`31860b59…`)
- coefficients: weak AdvCE $\beta=0.11834514302628477$; TPFM margin coefficient `0.316427398202933`, floor `0.17963354289531708`, cap `0.5595575273036957`

5 training pathsはFerretで完了し、dev-2 TPFMのみFerretでの親SHA検証失敗を技術的に再試行した。最終的には同一checkpoint bytes、source/config/seed/parent/mask/calibrationを保持したまま Hamster GPU0 上で復旧し、epoch 199 と全endpointを完了した。これは科学的な再試行・条件変更ではなく technical recovery である。Hamster TPFM の平均 throughput は約491.7 images/s、Ferretで完了した他runは約371–378 images/sだった。

## Held-out CE-PGD20（validation 5,000）

値は clean / robust (%); 括弧内は同seedの I100_CONTROL との差（pp）である。

| seed | epoch | I100_CONTROL | plain AdvCE | TPFM |
| --- | ---: | ---: | ---: | ---: |
| dev-1 | 129 | 84.32 / 58.16 | 84.82 / 58.38 (+0.50/+0.22) | 84.82 / 58.48 (+0.50/+0.32) |
| dev-1 | 149 | 84.46 / 57.60 | 84.96 / 57.44 (+0.50/-0.16) | 84.80 / 57.98 (+0.34/+0.38) |
| dev-1 | 169 | 85.62 / 60.18 | 86.14 / 59.96 (+0.52/-0.22) | 85.98 / 60.44 (+0.36/+0.26) |
| dev-1 | 189 | 85.84 / 60.52 | 86.34 / 60.54 (+0.50/+0.02) | 86.00 / 61.10 (+0.16/+0.58) |
| dev-1 | 199 | 85.84 / 60.96 | 86.52 / 60.66 (+0.68/-0.30) | 86.28 / 61.00 (+0.44/+0.04) |
| dev-2 | 129 | 84.84 / 58.02 | 85.38 / 57.74 (+0.54/-0.28) | 85.40 / 58.22 (+0.56/+0.20) |
| dev-2 | 149 | 84.22 / 57.10 | 84.60 / 57.68 (+0.38/+0.58) | 84.24 / 57.98 (+0.02/+0.88) |
| dev-2 | 169 | 85.88 / 59.92 | 86.34 / 59.94 (+0.46/+0.02) | 86.04 / 59.88 (+0.16/-0.04) |
| dev-2 | 189 | 86.18 / 60.34 | 86.44 / 60.10 (+0.26/-0.24) | 86.36 / 60.60 (+0.18/+0.26) |
| dev-2 | 199 | 86.16 / 60.50 | 86.56 / 60.50 (+0.40/+0.00) | 86.48 / 60.70 (+0.32/+0.20) |

Primary e199 robust result is therefore plain AdvCE `-0.30/+0.00 pp` and TPFM `+0.04/+0.20 pp` for dev-1/dev-2. Clean accuracy increased for all treatment arms, so the robust effect is not a clean-collapse artifact.

## e199 train direct / spillover (paired stable IDs)

| seed | arm | scope | n | clean Δ | robust Δ | robust rescue / harm |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| dev-1 | plain AdvCE | direct CW | 9,263 | +9.44 pp | +4.12 pp | 389 / 7 |
| dev-1 | plain AdvCE | spillover | 35,737 | +0.06 pp | -0.22 pp | 327 / 406 |
| dev-1 | TPFM | direct CW | 9,263 | +6.98 pp | +3.69 pp | 352 / 10 |
| dev-1 | TPFM | spillover | 35,737 | +0.03 pp | +0.09 pp | 376 / 345 |
| dev-2 | plain AdvCE | direct CW | 8,709 | +8.44 pp | +4.09 pp | 368 / 12 |
| dev-2 | plain AdvCE | spillover | 36,291 | +0.06 pp | -0.22 pp | 333 / 413 |
| dev-2 | TPFM | direct CW | 8,709 | +5.79 pp | +3.87 pp | 347 / 10 |
| dev-2 | TPFM | spillover | 36,291 | +0.08 pp | +0.08 pp | 345 / 317 |

Binary accuracy identity was checked as $\Delta Acc=\text{rescue rate}-\text{harm rate}$. Direct CW gains are much larger than held-out gains, indicating a substantial generalization gap. The TPFM spillover is weakly positive in both seeds; this is descriptive only.

## Continuation trajectory / runtime

The continuation dense metrics cover epochs 115–199. The table is a **post-e114 normalized AUC**, not the full epoch-0–199 AUC; the historical prefix is not recomputed here.

| seed | arm | post-114 AUC | best val robust | last val robust | mean images/s | train time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dev-1 | control | 59.376% | 61.10% | 60.86% | 374.6 | 2.84 h |
| dev-1 | plain AdvCE | 59.243% | 60.94% | 60.52% | 375.1 | 2.83 h |
| dev-1 | TPFM | 59.603% | 61.34% | 61.08% | 371.7 | 2.86 h |
| dev-2 | control | 59.059% | 60.68% | 60.60% | 377.7 | 2.81 h |
| dev-2 | plain AdvCE | 58.985% | 60.56% | 60.50% | 376.9 | 2.82 h |
| dev-2 | TPFM | 59.202% | 60.84% | 60.60% | 491.7 | 2.16 h |

Relative to the same-seed control, TPFM post-e114 AUC is +0.227 pp (dev-1) and +0.143 pp (dev-2), while plain AdvCE is -0.133 pp and -0.074 pp. These are continuation-window diagnostics and should not be confused with a full-trajectory AUC claim.

## Decision and limitations

- **CW1 (robust rescue):** mixed. TPFM is positive in both held-out seeds at e199, but the effect is only +0.04/+0.20 pp; plain AdvCE is not positive.
- **CW2 (direct-to-held-out transfer):** not confirmed. Direct CW robust gains are ~+3.7–4.1 pp, but held-out gains are near zero.
- **CW3 (TPFM over plain AdvCE):** supported directionally at e199 (+0.34/+0.20 pp), with positive post-e114 AUC difference in both seeds; small-n descriptive.
- **CW4 (durability):** TPFM remains positive at e129/e149/e189/e199 in both seeds, but is slightly negative at dev-2/e169; no formal durability claim.
- **CW5 (runtime):** no scientific throughput penalty is established; Hamster recovery is faster than Ferret, but it is a host comparison, not an arm effect.
- **CW6 (promotion):** not automatic. Keep the preregistered TPFM result as evidence for human review only.

No new coefficient, threshold, seed, e199 extension, official test, AutoAttack, dynamic routing, or combined treatment was started. Full machine-readable results are in [ert_rslad_i100_cw_long_horizon_results_v1.json](experiments/ert_rslad_i100_cw_long_horizon_results_v1.json), with contract, direct/spillover, and runtime artifacts alongside it.
