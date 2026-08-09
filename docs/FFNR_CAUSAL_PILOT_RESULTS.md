# FF/NR causal pilot results

更新日: 2026-08-09

## Scope and immutable inputs

The preregistered five-arm screen was run from the same Chen epoch-79 parent
for each seed.  The child horizon is epoch 79 -> 84 (the terminal checkpoint
is epoch 83), before either LR milestone.  No CIFAR-10 official test or
AutoAttack result is included here.

| item | value |
|---|---|
| code SHA | `2084d45d6976eab8c679c84b87155f5d98a0819b` |
| teacher | Chen2021LTD_WRN34_10 (Chen ERT) |
| parent | epoch-79 `last.pt`, optimizer/scheduler/RNG/sample state restored |
| training attack | KL-PGD10, `epsilon=8/255`, step `2/255`, pixel `[0,1]`, random start |
| endpoint | common training sample statistics and validation CE-PGD20 at epoch 83 |
| Route A | selected-only KD multiplier 0.5 + adversarial CE coefficient 0.25 |
| Route B | selected-only KD multiplier 1.0 + adversarial CE coefficient 0.25; q=5% |
| random controls | class/state/margin matched, fixed random seed `20260809` |

Hamster ran L2 (seed 1) and Ferret ran L4 (seed 2).  All ten current
`*-r2` W&B runs completed and were synchronized online.  Ferret checkpoint
SHA-256 was verified after rsync; the local and remote hashes match for all
five `last.pt` files.

## Endpoint validation metrics

Values are validation clean/PGD-20 accuracy at epoch 83.  These are short-
horizon secondary outcomes, not a best-checkpoint claim.

| seed/run | C79 | RA selected | RAR matched random | RB selected | RBR matched random |
|---|---:|---:|---:|---:|---:|
| L2 / seed 1 | 78.44 / 46.20 | 77.90 / 46.20 | 78.24 / 44.64 | 77.56 / 45.84 | 78.06 / 46.46 |
| L4 / seed 2 | 77.14 / 45.90 | 78.06 / 47.32 | 78.72 / 46.76 | 79.14 / 46.16 | 78.66 / 46.94 |

The first number is clean accuracy (%) and the second is validation PGD-20
accuracy (%).  The L2 Route-A selected arm is neutral at this endpoint while
L4 Route-A is higher than its control; the matched-random outcomes move in
both directions.  Route-B is similarly mixed.  Therefore this pilot does
not establish a global validation improvement.

## Selected-sample rescue and harm

For each treatment, `rescue = control wrong -> treatment correct` and
`harm = control correct -> treatment wrong`, using the same epoch-79 mask and
the final epoch-83 train sample statistics.  This is a paired mechanism
diagnostic, not an individual causal estimate of a long run.

| seed/arm | selected n | rescue | harm | net rescue | control robust acc. | treatment robust acc. |
|---|---:|---:|---:|---:|---:|---:|
| L2 RA | 5,667 | 235 | 150 | +85 | 6.58% | 8.08% |
| L2 RAR | 5,667 | 258 | 158 | +100 | 7.71% | 9.48% |
| L2 RB | 1,124 | 111 | 83 | +28 | 59.52% | 62.01% |
| L2 RBR | 1,124 | 98 | 83 | +15 | 64.77% | 66.10% |
| L4 RA | 5,755 | 252 | 187 | +65 | 7.40% | 8.53% |
| L4 RAR | 5,755 | 263 | 211 | +52 | 8.93% | 9.83% |
| L4 RB | 1,138 | 120 | 81 | +39 | 59.67% | 63.09% |
| L4 RBR | 1,138 | 104 | 80 | +24 | 64.59% | 66.70% |

Selected Route-A samples improve in both seeds, but the matched-random arm
also improves and is at least as large in L2.  Selected Route-B has a larger
net-rescue rate than its matched random in both seeds (L2: 2.49% vs 1.33%;
L4: 3.43% vs 2.11%).  This is evidence for a Route-B mechanism signal, not
yet evidence that it improves global validation accuracy.

## Interpretation and next gate

1. Keep the five-arm artifacts and the paired rescue/harm table as the
   preregistered mechanism result.
2. Do not launch a 200-epoch intervention or claim a Best improvement from
   this short horizon.  The global validation result is mixed and the Route-A
   selected-vs-random contrast is not consistently positive.
3. Route B is the only candidate with a consistent selected-over-random
   selected-sample net-rescue contrast.  Before any long run, evaluate its
   non-selected spillover and clean/robust trade-off from the saved tables,
   then freeze one confirmatory seed/teacher decision without changing q or
   coefficients.
4. If the next gate is passed, use a fresh immutable parent and a separately
   named confirmation run.  Official test and AutoAttack remain separate
   saved-checkpoint evaluations.

Machine-readable aggregation is stored locally at
`.cache/analysis/ffnr-causal-pilot-result-v1.json` and is intentionally not
committed with checkpoints or W&B data.
