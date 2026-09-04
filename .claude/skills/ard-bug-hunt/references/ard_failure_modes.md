# ARD failure-mode checklist

## Threat model and attack

- Is the model input represented in `[0, 1]`, standardized space, or model-specific normalized space?
- Are `epsilon` and `alpha` converted exactly once?
- Is random start inside the allowed ball?
- Is projection performed in the intended space and norm?
- Is the final input clamped to the valid image domain after projection?
- Is the attack targeted or untargeted as intended?
- Is ascent/descent sign correct for the selected loss?
- Are gradients taken with respect to the intended student or teacher input?
- Is a stale tensor, detached tensor, or accumulated gradient reused across PGD steps?
- Are attack steps accidentally run under autocast when full precision is required?

## Model state

- Are student and teacher in the intended train/eval mode during attack and update?
- Are BatchNorm running statistics changing during attack generation?
- Is Dropout introducing unintended randomness?
- Are teacher parameters frozen while preserving input gradients where required?
- Are gradients zeroed at the correct boundary?

## Distillation objective

- Is KL direction consistent with the paper and upstream implementation?
- Are logits converted with the correct `softmax`/`log_softmax` dimensions?
- Is temperature applied to both distributions?
- Is `T^2` scaling present or intentionally absent?
- Are clean and adversarial inputs paired with the intended labels/logits?
- Are per-sample weights applied before or after reduction as intended?
- Are weights bounded, finite, and normalized consistently?

## Student-aware state

- Does each batch carry an immutable original sample index?
- Do augmentations preserve the mapping to the original index?
- Does DDP update each sample state exactly as designed?
- Are EMA coefficients and update timing correct?
- Is state updated from pre-update or post-update predictions as specified?
- Is sample state serialized and restored with checkpoints?

## Numerical behavior

- Are logits, losses, gradients, and weights finite?
- Does AMP change a reference computation beyond tolerance?
- Is GradScaler restored on resume?
- Are reductions identical across batch size and DDP world size?
- Are tolerance choices justified by dtype and hardware?

## Checkpoint and resume

- Are model, optimizer, scheduler, scaler, epoch, global step, best metric, RNG states, sampler state, sample state, and W&B ID saved?
- Does resume repeat or skip a data batch?
- Does scheduler advance twice?
- Is best checkpoint selection preserved?
- Are best and last artifacts distinct?

## W&B and distributed execution

- Does only rank 0 initialize/log?
- Are metric steps monotonic after resume?
- Is the same run ID reused rather than creating duplicates?
- Are offline runs retained until sync succeeds?
- Are config, Git SHA, external SHA, teacher hash, and seed present?
- Are media/table logs sparse enough to avoid training stalls?

## Differential diagnosis

For parity mismatches, compare in this order:

1. input batch and normalization
2. clean logits
3. random-start tensor
4. each PGD step tensor and loss
5. final adversarial logits
6. each unreduced loss component
7. per-sample weight
8. reduced loss
9. gradients on a selected parameter tensor
10. one optimizer step and resulting parameter delta

Stop at the first divergence; do not compare only final accuracy.
