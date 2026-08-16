# ERT Clean-Wrong Rescue Subtype Analysis

Read-only C0/C10/C12/C13 epoch-84 endpoint transition analysis. No new training or route selection.

## L2

Fixed Clean-Wrong cohort: 8623 samples.

| arm | group | n | teacher clean correct | teacher adv correct | student clean p mean | teacher adv p mean | ΔT mean |
|---|---|---:|---:|---:|---:|---:|---:|
| C10 | clean_and_robust_rescue | 11 | 0.909 | 0.818 | 0.2043 | 0.3142 | 0.0644 |
| C10 | clean_only_rescue | 738 | 0.837 | 0.428 | 0.1882 | 0.2263 | 0.0976 |
| C10 | robust_only_rescue | 182 | 0.967 | 0.791 | 0.2324 | 0.3122 | 0.0916 |
| C10 | neither_or_harm | 7692 | 0.602 | 0.317 | 0.1706 | 0.2059 | 0.0902 |
| C12 | clean_and_robust_rescue | 8 | 0.750 | 0.625 | 0.1960 | 0.2706 | 0.0610 |
| C12 | clean_only_rescue | 641 | 0.835 | 0.378 | 0.1894 | 0.2188 | 0.0992 |
| C12 | robust_only_rescue | 167 | 0.946 | 0.713 | 0.2264 | 0.3087 | 0.0933 |
| C12 | neither_or_harm | 7807 | 0.606 | 0.326 | 0.1710 | 0.2072 | 0.0902 |
| C13 | clean_and_robust_rescue | 5 | 1.000 | 1.000 | 0.2276 | 0.3032 | 0.0445 |
| C13 | clean_only_rescue | 412 | 0.905 | 0.522 | 0.2045 | 0.2507 | 0.1024 |
| C13 | robust_only_rescue | 132 | 0.992 | 0.894 | 0.2470 | 0.3393 | 0.0974 |
| C13 | neither_or_harm | 8074 | 0.610 | 0.318 | 0.1706 | 0.2058 | 0.0902 |

### C10/C13 rescue overlap

- clean: C10=749, C13=417, intersection=265, union=901, Jaccard=0.29411764705882354
- robust: C10=193, C13=137, intersection=66, union=264, Jaccard=0.25

### Pre-treatment Teacher reliability strata

| arm | stratum | n | Teacher adv correct | mean mT_adv | clean Δ | robust Δ | robust net rescue |
|---|---|---:|---:|---:|---:|---:|---:|
| C10 | CW-R | 2908 | 1.000 | 0.1023 | +0.0148 | +0.0048 | +0.0299 |
| C10 | CW-U | 5715 | 0.000 | -0.1193 | +0.0181 | +0.0094 | +0.0045 |
| C12 | CW-R | 2908 | 1.000 | 0.1023 | +0.0020 | +0.0000 | +0.0175 |
| C12 | CW-U | 5715 | 0.000 | -0.1193 | +0.0130 | +0.0090 | +0.0066 |
| C13 | CW-R | 2908 | 1.000 | 0.1023 | +0.0019 | +0.0085 | +0.0172 |
| C13 | CW-U | 5715 | 0.000 | -0.1193 | +0.0043 | +0.0084 | +0.0000 |

## L4

Fixed Clean-Wrong cohort: 8925 samples.

| arm | group | n | teacher clean correct | teacher adv correct | student clean p mean | teacher adv p mean | ΔT mean |
|---|---|---:|---:|---:|---:|---:|---:|
| C10 | clean_and_robust_rescue | 18 | 0.944 | 0.833 | 0.2476 | 0.3299 | 0.0556 |
| C10 | clean_only_rescue | 947 | 0.889 | 0.502 | 0.1938 | 0.2444 | 0.1002 |
| C10 | robust_only_rescue | 146 | 0.986 | 0.877 | 0.2435 | 0.3366 | 0.0862 |
| C10 | neither_or_harm | 7814 | 0.604 | 0.320 | 0.1692 | 0.2055 | 0.0899 |
| C12 | clean_and_robust_rescue | 14 | 0.929 | 0.786 | 0.2663 | 0.3268 | 0.0613 |
| C12 | clean_only_rescue | 665 | 0.878 | 0.529 | 0.2021 | 0.2531 | 0.0986 |
| C12 | robust_only_rescue | 180 | 0.978 | 0.811 | 0.2394 | 0.3162 | 0.0990 |
| C12 | neither_or_harm | 8066 | 0.613 | 0.324 | 0.1692 | 0.2061 | 0.0901 |
| C13 | clean_and_robust_rescue | 5 | 1.000 | 1.000 | 0.2515 | 0.3358 | 0.0668 |
| C13 | clean_only_rescue | 521 | 0.869 | 0.528 | 0.2022 | 0.2483 | 0.1010 |
| C13 | robust_only_rescue | 111 | 0.991 | 0.901 | 0.2331 | 0.3287 | 0.0858 |
| C13 | neither_or_harm | 8288 | 0.622 | 0.330 | 0.1706 | 0.2081 | 0.0903 |

### C10/C13 rescue overlap

- clean: C10=965, C13=526, intersection=376, union=1115, Jaccard=0.33721973094170404
- robust: C10=164, C13=116, intersection=43, union=237, Jaccard=0.18143459915611815

### Pre-treatment Teacher reliability strata

| arm | stratum | n | Teacher adv correct | mean mT_adv | clean Δ | robust Δ | robust net rescue |
|---|---|---:|---:|---:|---:|---:|---:|
| C10 | CW-R | 3119 | 1.000 | 0.1052 | +0.0250 | +0.0107 | +0.0151 |
| C10 | CW-U | 5806 | 0.000 | -0.1191 | +0.0203 | +0.0064 | +0.0010 |
| C12 | CW-R | 3119 | 1.000 | 0.1052 | +0.0078 | +0.0127 | +0.0212 |
| C12 | CW-U | 5806 | 0.000 | -0.1191 | +0.0120 | +0.0130 | +0.0036 |
| C13 | CW-R | 3119 | 1.000 | 0.1052 | +0.0005 | -0.0000 | -0.0022 |
| C13 | CW-U | 5806 | 0.000 | -0.1191 | +0.0037 | -0.0001 | -0.0012 |

## Interpretation

The strata were fixed before reading treatment outcomes:

$$
CW\text{-}R = \{i:m_{T,i}^{adv}(epoch79)>0\},\qquad
CW\text{-}U = \{i:m_{T,i}^{adv}(epoch79)\le 0\}.
$$

Because probability-margin positivity is equivalent to Teacher adversarial
correctness, this is a preregistered semantic split rather than a tuned
threshold. The L2 counts are 2,908/5,715 and L4 counts are 3,119/5,806.

### Does Teacher reliability select a better C13 action?

No clear two-seed confirmation was found. C13 robust effects were:

| arm / stratum | L2 robust Δ | L4 robust Δ | L2 robust net rescue | L4 robust net rescue |
|---|---:|---:|---:|---:|
| C13 / CW-R | +0.853 pp | −0.005 pp | +1.72 pp | −0.22 pp |
| C13 / CW-U | +0.843 pp | −0.006 pp | 0.00 pp | −0.12 pp |

The L2 gain is almost identical in CW-R and CW-U, while L4 is neutral in
both. Therefore the hypothesis “C13 is harmful mainly because it is applied
to Teacher-unreliable samples” is not supported by this endpoint.

### Does Teacher reliability select a better C10 action?

C10 has a more informative safety pattern than C13, but not a stronger robust
accuracy effect in CW-R across both seeds:

| arm / stratum | L2 clean Δ | L4 clean Δ | L2 robust Δ | L4 robust Δ |
|---|---:|---:|---:|---:|
| C10 / CW-R | +1.477 pp | +2.504 pp | +0.481 pp | +1.072 pp |
| C10 / CW-U | +1.813 pp | +2.027 pp | +0.943 pp | +0.640 pp |

CW-R has a higher robust net-rescue rate (`+2.99/+1.51 pp`) than CW-U
(`+0.45/+0.10 pp`), because harm is much lower. However, the paired robust
accuracy delta is not consistently larger in CW-R. Teacher reliability may
therefore be useful as a safety gate for CleanCE, but it is not yet validated
as a robust-benefit selector.

C12 is similarly inconclusive: robust Δ is `+0.005/+1.266 pp` in CW-R and
`+0.905/+1.299 pp` in CW-U. This does not show a reliable Teacher-gated
advantage.

### Decision

The correct conclusion is narrower than a routing claim:

1. The earlier outcome-conditioned result was not sufficient to establish a
   Teacher-reliability selector.
2. With treatment-pre Teacher reliability fixed first, C13 does not show a
   reproducible CW-R advantage.
3. C10 shows a possible harm-reduction/safety effect in CW-R, but not a
   reproducible larger robust-accuracy gain.
4. The next method, if pursued, should separate robust rescue from clean
   rescue and preregister safety (harm/net-rescue) as a distinct endpoint.

No threshold tuning, new treatment, official test, AutoAttack, or new seed was
started. Full feature fields and paired IDs are in the JSON artifact.
