# 0030: ConvStem variants collapse to a constant output under PGD-AT from random init

## What happened

`plan0103-phase2-convnext-atto-convstem-s0-v1` (source 5dd329e, Hamster 4090, AdamW lr 1e-3, wd 0.05,
10-epoch warmup, batch 128, PGD-3 at 4/255) ran 50/50 epochs. It used about 29 GPU-hours and ended
"completed", but every accuracy stayed at 0.10% (chance) and the training loss stayed at ln(1000) from
epoch 0. The deep-narrow control, with the same source, recipe and host, reached 57.97 / 35.53. The
collapse guard ("below half of the best value so far") never fired, because the best value was 0.1%
from the start. Decision packet: 0057.

## Root cause (CPU investigation, 2026-10-15)

This is not a wiring bug. The stem is in the optimizer and gets gradients. Our `ConvNeXtConvStem`
matches upstream `ConvBlock1` (Singh et al.). The cause is an interaction between the recipe and the
architecture:

- In a random-init network, the GELU / LayerNorm conv stem puts about 40-50% of the stage-0 stream
  energy into an input-independent per-channel offset ("DC"). This was measured as the share of
  activation energy that is constant across images and pixels:

  | Architecture | DC share at stem output | DC share at stage downsamples |
  |---|---|---|
  | ConvStem | 0.42 | 0.29-0.34 |
  | Standard ConvNeXt-Atto | 0.003 | 0.003 |
  | deep-narrow | 0.004 | 0.004 |

- Under PGD-AT from step 0, gradient descent uses that offset to make the network input-independent.
  This is the trivial robust minimum at loss ln(1000). One sign step on a downsample conv enlarges the
  constant part, and the next LayerNorm divides the input-dependent signal down. At the epoch-0
  checkpoint the relative input-dependent signal falls to 1e-3, 5e-5 and 2e-6 after stages 1-3. The
  logit standard deviation across images is 3e-7, and every image gets the same class.
- Reproduction (256 real val images, batch 16, lr 1e-4, the epoch-0 learning rate):

  | Model and condition | Logit std across images | Outcome |
  |---|---|---|
  | ConvStem, PGD-AT | 0.163 -> 0.025 in 25 steps | collapses |
  | ConvStem, clean CE | rises | learns |
  | Atto, PGD-AT | 0.33 -> 0.73 | learns |
  | deep-narrow, PGD-AT | 0.33 -> 0.73 | learns |
  | Atto with the same DC offset injected | collapses | DC is sufficient |
  | ConvStem with its init DC subtracted | recovers | DC is necessary |

- Weight decay is a consequence, not the trigger. It later erased the unused stem: the stem weight
  norm went from 2.4 to 2e-4.
- Ruled out: GPU numerics (the collapse reproduces on CPU in fp32). Default-init vs timm-style stem
  init (both collapse). Removing the final GELU (still collapses).
- Upstream differs in regime. Singh et al. start from a clean-pretrained body. This project trains from
  random init under PGD from step 0.

Scripts and logs: `/tmp/claude-1001/.../scratchpad/convstem/` (`train_probe.py`, `dc.py`,
`ck_forward.py`, `attrib2.py`). Session scratch; the numbers above are the record.

## Scope

- `deit_tiny_convstem_imagenet` shows the same signature: DC share 0.50, and it collapses in 25 probe
  steps. Its Ferret queue item was removed on 2026-10-15 before it started, pending the decision.
- Not affected:
  - `convnext_atto_ols`: DC 0.08, like the standard Atto.
  - The MobileNetV4 variants: BatchNorm removes per-channel DC in train mode.
  - The Singh teacher ConvStems: a separate code path, with trained, frozen weights.

## Follow-ups

1. **Never-learned guard (code).** Add `val_subset_distinct_predictions` and
   `val_subset_logit_std_over_images` to the selection-subset forward; they draw no RNG and do not
   change numerics. Abort and write the error marker when distinct predictions are <= 1, or the logit
   std is below about 1e-4, at both epoch 0 and epoch 1. Loss or accuracy cannot serve as the signal:
   healthy runs start at a loss of up to 7.15 and an accuracy as low as 0.21%.
2. **Structural test.** The stem-output DC share at seed 0 must be below 0.25 for every random-init
   ImageNet student. Today only the two ConvStem ids fail (0.41 / 0.53).
3. **Science (decision packet 0057, human).** Options:
   - Close the random-init ConvStem rows, recording this mechanism.
   - Clean-pretrained body. This conflicts with the 2026-10-01 no-pretraining decision.
   - Epsilon warmup, with a matched control.
   This connects to plan 0107's optimization-not-capacity hypothesis: a trivial constant-output
   attractor of PGD-AT from scratch.
