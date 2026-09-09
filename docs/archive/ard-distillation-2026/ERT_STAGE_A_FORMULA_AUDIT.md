# ERT Stage A — implemented RSLAD formula audit

This is a code audit, not a training result. The source of truth is
`src/ard/objectives/rslad.py`, `src/ard/objectives/kl.py`, and the resolved
Chen RSLAD configuration.

## Baseline found in code

For sample $i$, the current objective returns unreduced terms

$$
L_i^{base} = \frac{5}{6}L_{advKD,i} + \frac{1}{6}L_{cleanKD,i}.
$$

The adversarial term is

$$
L_{advKD,i}=KL(q_T(x_i)\parallel p_S(x_i^{adv})),
$$

and the clean term is

$$
L_{cleanKD,i}=KL(q_T(x_i)\parallel p_S(x_i)).
$$

The implementation uses `target_to_student_kl`, so the teacher target is
detached and the direction is teacher-to-student. The configured Chen
training attack is KL-PGD10 with teacher-clean target, random start,
$\epsilon=8/255$, step size $2/255$, pixel domain $[0,1]$, and
`temperature=1.0`, `temperature_squared=true`. Therefore the configured
$T^2$ factor is numerically one, but its presence remains part of the frozen
configuration contract.

The baseline policy is KD-only (`hard_weight=0`, `kd_weight=1`) and the final
batch reduction is the full valid-batch mean. The objective still exposes a
latent hard-CE vector for generic method composition, but it is not part of
the effective RSLAD baseline update.

## Stage A interpretation fixed before implementation

- `tau=2.0` is a separate Teacher-target temperature for selected samples
  only. It does not change the student objective temperature, attack target,
  attack steps, or the baseline $T^2$ setting.
- The S3-T3 `KD=1.0` arm means the baseline adversarial-KD coefficient remains
  $5/6$ (multiplier one), not an unweighted raw sum of KD branches.
- CleanCE calibration therefore uses the gradient of the effective baseline
  KD loss, not `ObjectiveTerms.total` including the policy-disabled hard CE.
- The old `ert_state_overlay_v1` labels are historical. Stage A uses the
  explicit predicates `student_clean_correct & student_adversarial_wrong` for
  S3 and the registered CE-PGD20 Teacher T1/T2/T3 masks.
- The equal-rank composite is not consumed by the Stage A selector. It remains
  a separate human-decision item and is not silently assigned a population.

## Calibration boundary

Calibration will measure component gradients from the exact epoch-79 parent
without optimizer, scheduler, sample-state, checkpoint, validation, or future
outcome mutation. Coefficients are frozen once in the machine-readable
calibration artifact and cannot be changed after endpoint results are seen.
