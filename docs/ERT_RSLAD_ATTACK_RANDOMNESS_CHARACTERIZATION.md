# ERT / RSLAD Attack Random-Start Randomness Characterization

This is a descriptive fixed-model plus 15-epoch attack-seed characterization. No attack intervention or seed promotion was performed.

## Fixed-model direct replay

| dev seed | n | risk→margin-SD Spearman | risk→attack-loss-SD Spearman |
| ---: | ---: | ---: | ---: |
| 1 | 8192 | 0.17833687249537553 | 0.1701108649604943 |
| 2 | 8192 | 0.16887184832630406 | 0.18289745540513824 |

## 15-epoch trajectory probe

| dev seed | attack AUC mean | SD | range | e114 robust mean |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.509753 | 0.000117 | 0.509581–0.509895 | 0.525231 |
| 2 | 0.509356 | 0.000134 | 0.509125–0.509517 | 0.521869 |

## Sample-level e114 sensitivity

| dev seed | validation rows | non-unanimous fraction | margin-SD mean |
| ---: | ---: | ---: | ---: |
| 1 | 5000 | 0.031000 | 0.006727 |
| 2 | 5000 | 0.030400 | 0.006372 |

## Interpretation

Attack-seed dispersion is reported against the pre-existing pure-order reference. The classification is characterization only; it does not authorize a training intervention, seed selection, or extension.

## Lineage

Registry SHA-256: `6b114d3100c7e4949e64877f9360cbe847c108f6b75203111228eb032be3e26b`
Source SHA: recorded in the frozen registry.
