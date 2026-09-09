# Clean-Wrong Teacher Reliability — Online Proxy & Safety Analysis

Pre-treatment epoch-79 reliability replay versus C0/C10/C12/C13 epoch-84 endpoints. No training, tuning, or route selection.

## Metric semantics audit

`accuracy_delta = rescue_rate - harm_rate`; `margin_delta` is the mean paired probability-margin change. They are stored as separate fields in the JSON and tested independently.

## KL10 versus CE20 agreement

| run | Pearson | Spearman | sign agreement | correctness agreement | KL-R→CE-R | KL-R→CE-U | KL-U→CE-R | KL-U→CE-U |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L2 | 0.9374 | 0.9203 | 0.911 | 0.911 | 2798 | 654 | 110 | 5061 |
| L4 | 0.9355 | 0.9149 | 0.909 | 0.909 | 3020 | 709 | 99 | 5097 |

## C10 CE20 quintile safety effects

| run | bin | n | mT range | robust accuracy Δ | robust rescue | robust harm | robust net rescue | clean accuracy Δ |
|---|---|---:|---|---:|---:|---:|---:|---:|
| L2 | Q1 | 1724 | [-0.6131, -0.1463] | +0.0012 | 0.0012 | 0.0000 | +0.0012 | +0.0238 |
| L2 | Q2 | 1725 | [-0.1463, -0.0798] | +0.0012 | 0.0029 | 0.0017 | +0.0012 | +0.0441 |
| L2 | Q3 | 1724 | [-0.0798, -0.0210] | +0.0093 | 0.0133 | 0.0041 | +0.0093 | +0.0818 |
| L2 | Q4 | 1725 | [-0.0210, 0.0559] | +0.0151 | 0.0290 | 0.0139 | +0.0151 | +0.0806 |
| L2 | Q5 | 1725 | [0.0560, 0.6312] | +0.0388 | 0.0655 | 0.0267 | +0.0388 | +0.0423 |
| L4 | Q1 | 1785 | [-0.5922, -0.1462] | -0.0017 | 0.0006 | 0.0022 | -0.0017 | +0.0162 |
| L4 | Q2 | 1785 | [-0.1462, -0.0776] | -0.0011 | 0.0000 | 0.0011 | -0.0011 | +0.0543 |
| L4 | Q3 | 1785 | [-0.0775, -0.0171] | +0.0039 | 0.0078 | 0.0039 | +0.0039 | +0.0902 |
| L4 | Q4 | 1785 | [-0.0171, 0.0633] | +0.0078 | 0.0190 | 0.0112 | +0.0078 | +0.1238 |
| L4 | Q5 | 1785 | [0.0634, 0.7393] | +0.0207 | 0.0644 | 0.0437 | +0.0207 | +0.1008 |

## Secondary C12/C13 quintile effects

The JSON contains clean/robust accuracy, margin, rescue, harm, and net-rescue fields for C10, C12, and C13 under both CE20 and KL10 quantile bins. Bins are determined from pre-treatment feature distributions only.

Machine report content hash: `05a288ad38b0d71181b166360aab9e2a0f76fb99d596525c638e60284077e82d`.
