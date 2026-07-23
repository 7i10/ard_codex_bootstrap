# Joint pilot acceptance semantics

## Failure

All three pilots at campaign SHA `2d54b8230b8d14d13c1ea7472ccba53491b4d38d` completed training and official
best/last PGD-20 evaluation. The production acceptor nevertheless rejected Joint because it required
`kd_weight < 1` after warmup.

That predicate belongs to legacy/downweight ablations, not the schema-v2 main `rslad_joint` method. Main Joint keeps
KD weight exactly `1.0` and passes risk into `teacher_target_uniform_mix@1`, where `rho = 0.5 * joint_risk`.

Observed epoch-2 sample statistics covered all 45,000 training samples:

- joint risk min/mean/max: `0.002404 / 0.134750 / 0.609255`
- KD weight: exactly `1.0`
- epoch-0 panel risk: zero; epoch-1 and epoch-2 panel risks: positive and nonconstant

## Fix and reuse decision

The acceptor now requires the exact canonical target policy, finite bounded positive/nonconstant risk after the
configured warmup, and uniform KD weight `1.0`. It rejects zero risk and unintended KD downweighting.

No trainer, config, policy, checkpoint, metric, attack, or W&B artifact changed. Therefore the completed `2d54b82`
pilots remain the production evidence; rerunning them would add no scientific information. The acceptor correction is
committed separately so future validation reproduces the gate decision.
