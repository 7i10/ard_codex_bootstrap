# ERT Clean-Wrong Rescue Subtype Analysis

Read-only C0/C10/C12/C13 epoch-84 endpoint transition analysis. No new training or route selection.

## L2

Fixed Clean-Wrong cohort: 8623 samples.

| arm | group | n | teacher clean correct | teacher adv correct | student clean p mean | teacher adv p mean | ΔT mean |
|---|---|---:|---:|---:|---:|---:|---:|
| C10 | clean_and_robust_rescue | 11 | 0.909 | 0.818 | 0.2298 | 0.3119 | 0.0603 |
| C10 | clean_only_rescue | 738 | 0.837 | 0.434 | 0.1975 | 0.2251 | 0.0974 |
| C10 | robust_only_rescue | 182 | 0.967 | 0.813 | 0.3152 | 0.3091 | 0.0904 |
| C10 | neither_or_harm | 7692 | 0.602 | 0.321 | 0.1816 | 0.2055 | 0.0892 |
| C12 | clean_and_robust_rescue | 8 | 0.750 | 0.625 | 0.2266 | 0.2691 | 0.0529 |
| C12 | clean_only_rescue | 641 | 0.835 | 0.393 | 0.1958 | 0.2175 | 0.0988 |
| C12 | robust_only_rescue | 167 | 0.946 | 0.749 | 0.3158 | 0.3052 | 0.0932 |
| C12 | neither_or_harm | 7807 | 0.606 | 0.329 | 0.1822 | 0.2068 | 0.0892 |
| C13 | clean_and_robust_rescue | 5 | 1.000 | 1.000 | 0.2383 | 0.2956 | 0.0595 |
| C13 | clean_only_rescue | 412 | 0.905 | 0.515 | 0.2095 | 0.2491 | 0.1028 |
| C13 | robust_only_rescue | 132 | 0.992 | 0.886 | 0.3271 | 0.3370 | 0.0938 |
| C13 | neither_or_harm | 8074 | 0.610 | 0.324 | 0.1823 | 0.2054 | 0.0892 |

### C10/C13 rescue overlap

- clean: C10=749, C13=417, intersection=265, union=901, Jaccard=0.29411764705882354
- robust: C10=193, C13=137, intersection=66, union=264, Jaccard=0.25

## L4

Fixed Clean-Wrong cohort: 8925 samples.

| arm | group | n | teacher clean correct | teacher adv correct | student clean p mean | teacher adv p mean | ΔT mean |
|---|---|---:|---:|---:|---:|---:|---:|
| C10 | clean_and_robust_rescue | 18 | 0.944 | 0.833 | 0.2472 | 0.3279 | 0.0598 |
| C10 | clean_only_rescue | 947 | 0.889 | 0.483 | 0.2017 | 0.2424 | 0.1033 |
| C10 | robust_only_rescue | 146 | 0.986 | 0.863 | 0.3187 | 0.3340 | 0.0884 |
| C10 | neither_or_harm | 7814 | 0.604 | 0.313 | 0.1854 | 0.2041 | 0.0915 |
| C12 | clean_and_robust_rescue | 14 | 0.929 | 0.714 | 0.2640 | 0.3221 | 0.0758 |
| C12 | clean_only_rescue | 665 | 0.878 | 0.507 | 0.2125 | 0.2515 | 0.1022 |
| C12 | robust_only_rescue | 180 | 0.978 | 0.828 | 0.3171 | 0.3145 | 0.0996 |
| C12 | neither_or_harm | 8066 | 0.613 | 0.316 | 0.1846 | 0.2046 | 0.0917 |
| C13 | clean_and_robust_rescue | 5 | 1.000 | 1.000 | 0.2470 | 0.3418 | 0.0550 |
| C13 | clean_only_rescue | 521 | 0.869 | 0.514 | 0.2099 | 0.2470 | 0.1029 |
| C13 | robust_only_rescue | 111 | 0.991 | 0.883 | 0.3144 | 0.3261 | 0.0915 |
| C13 | neither_or_harm | 8288 | 0.622 | 0.323 | 0.1865 | 0.2066 | 0.0920 |

### C10/C13 rescue overlap

- clean: C10=965, C13=526, intersection=376, union=1115, Jaccard=0.33721973094170404
- robust: C10=164, C13=116, intersection=43, union=237, Jaccard=0.18143459915611815

## Interpretation

The feature replay is the C0 epoch-84 checkpoint under the same full-train
CE-PGD20 ordering as the saved endpoint. The transition labels are paired
against C0 at epoch 84. Thus this is a same-endpoint association analysis,
not a prospective predictor or a causal effect estimate.

### Why C10 clean recovery often remains non-robust

The decisive contrast is between `clean_only_rescue` and
`robust_only_rescue`:

| run / C10 group | n (%) | Student $m_S^{adv}$ | Teacher $m_T^{adv}$ | Teacher adv correct | Teacher $p_T(y\mid x^{adv})$ | $\Delta_T$ |
|---|---:|---:|---:|---:|---:|---:|
| L2 clean-only | 738 (8.56%) | −0.1662 | −0.0096 | 43.4% | 0.2251 | 0.0974 |
| L2 robust-only | 182 (2.11%) | −0.0477 | 0.0988 | 81.3% | 0.3091 | 0.0904 |
| L4 clean-only | 947 (10.61%) | −0.1816 | 0.0055 | 48.3% | 0.2424 | 0.1033 |
| L4 robust-only | 146 (1.64%) | −0.0613 | 0.1277 | 86.3% | 0.3340 | 0.0884 |

The clean-only group has a substantially more negative Student adversarial
margin and a Teacher whose adversarial prediction is correct only about
43–48% of the time. In contrast, robust-only rescue occurs mostly where the
Teacher is already adversarially correct with positive margin. The difference
is large for $m_T^{adv}$ and Teacher correctness, but small for $\Delta_T$;
therefore the useful modifier is Teacher residual adversarial reliability,
not merely the clean-to-adv margin drop.

For completeness, the C10 clean-only means for Student clean/adv true
probability are `0.1975/0.1194` (L2) and `0.2017/0.1212` (L4), while the
robust-only means are `0.3152/0.2019` and `0.3187/0.2021`. Teacher clean
correctness is also lower in clean-only (`83.7%/88.9%`) than robust-only
(`96.7%/98.6%`). This explains why extra CleanCE can repair the clean label
without making the adversarial decision robust: it acts on a sample where the
student and the Teacher adversarial signal are both weak.

### C13 and C12 comparison

C13 shows the same ordering: clean-only Teacher adversarial correctness is
`51.5%/51.4%`, versus `88.6%/88.3%` for robust-only. C12 is qualitatively
similar. The pattern is therefore not unique to CleanCE; it is a subtype
property of the samples that become clean-correct without crossing the robust
margin.

### C10/C13 overlap

| run | clean-rescue overlap | robust-rescue overlap |
|---|---:|---:|
| L2 | 265 / 901, Jaccard 0.294 | 66 / 264, Jaccard 0.250 |
| L4 | 376 / 1,115, Jaccard 0.337 | 43 / 237, Jaccard 0.181 |

The overlap is only moderate, especially for robust rescue. C10 and C13 are
therefore not simply selecting the same samples; a future treatment should
not blindly add the two losses. A preregistered subtype design could instead
retain CleanCE for clean-only candidates while using a Teacher-reliability or
adversarial-margin condition for candidates eligible for robust recovery.
That is a hypothesis for a new experiment, not an automatic route decision.

All feature means, medians, quartiles, class counts, sample-ID hashes, endpoint
lineage, and full C0/C10/C12/C13 group IDs are in the machine-readable JSON.
No new training, coefficient tuning, official test, or AutoAttack was run.
