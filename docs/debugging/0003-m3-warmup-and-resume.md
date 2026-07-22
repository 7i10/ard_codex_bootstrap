# M3 student-aware warmup and resume correction

## Failure signature

The first implementation collected the specified detached pre-update robust
margin during epoch zero, but applied `hard_weight=0.5` and `kd_weight=0.5`.
Canonical `rslad_student` and `rslad_joint` require epoch zero to be exactly
baseline RSLAD: `hard_weight=0`, `kd_weight=valid_mask`, and `joint_risk=0`.
The same review found that canonical method IDs accepted alternate EMA decays
and warmup lengths without a distinct method identity, and their exact
epoch-boundary resume path lacked a two-epoch objective-level regression.

## Root cause and correction

Warmup had been interpreted as an equal hard/KD blend rather than uniform
baseline RSLAD. The Trainer warmup branch now returns the baseline weights while
still queuing FP32 detached margin and correctness observations before the
optimizer update. Canonical student/joint configs reject decay values other
than `0.9` and warmup values other than `1`; generalized experiments require a
separate method ID.

A deterministic CPU regression covers both student and joint policies for two
epochs with genuine RSLAD terms and sample state. It compares uninterrupted
training with an epoch-zero checkpoint/resume using one constant config hash,
checks epoch-zero baseline weights, verifies epoch-one risk from the restored
prior EMA (and current teacher risk for joint), and requires exact model, store,
global-step, and scheduler equality. The impact map selects both M3 regression
files for config, objective, engine, and training-CLI changes.

## Scientific boundaries checked

Attack configuration, pixel-space normalization, projection, attack loss,
teacher freezing, KL direction, temperature scaling, validation, and tracking
paths were not changed. Real two-rank Gloo verification remains an outside-
sandbox test because the restricted sandbox denies TCPStore sockets.
