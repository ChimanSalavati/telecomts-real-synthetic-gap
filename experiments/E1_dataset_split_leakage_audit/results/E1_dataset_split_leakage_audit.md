# E1 — Dataset, split, and leakage audit

* Total windows: **32,000**
* Normal: **30,765**, Anomaly: **1,235**
* Real Jamming: **279**, Synthetic anomalies: **956**

## Split sizes

| split              | partition   |     n |   n_normal |   n_anomaly |   n_real |   n_synthetic |
|:-------------------|:------------|------:|-----------:|------------:|---------:|--------------:|
| small_natural      | train       |   640 |        607 |          33 |        6 |            27 |
| small_natural      | val         |   160 |        153 |           7 |        0 |             7 |
| small_natural      | test        |   200 |        190 |          10 |        2 |             8 |
| balanced_detection | train       |  1580 |        790 |         790 |      163 |           627 |
| balanced_detection | val         |   396 |        198 |         198 |       50 |           148 |
| balanced_detection | test        |   494 |        247 |         247 |       66 |           181 |
| controlled_500     | train       |   300 |        150 |         150 |       75 |            75 |
| controlled_500     | val         |   100 |         50 |          50 |       25 |            25 |
| controlled_500     | test        |   100 |         50 |          50 |       25 |            25 |
| fullscale          | train       | 25600 |      24612 |         988 |      210 |           778 |
| fullscale          | test        |  6400 |       6153 |         247 |       69 |           178 |
| split_rca_balanced | train       |   988 |          0 |         988 |      223 |           765 |
| split_rca_balanced | test        |   247 |          0 |         247 |       56 |           191 |

## Leakage checks (sample-id overlap between partitions per split)

| split              | pair       |   overlap |
|:-------------------|:-----------|----------:|
| small_natural      | train-val  |         0 |
| small_natural      | train-test |         0 |
| small_natural      | val-test   |         0 |
| balanced_detection | train-val  |         0 |
| balanced_detection | train-test |         0 |
| balanced_detection | val-test   |         0 |
| controlled_500     | train-val  |         0 |
| controlled_500     | train-test |         0 |
| controlled_500     | val-test   |         0 |
| fullscale          | train-test |         0 |
| split_rca_balanced | train-test |         0 |

## Collection-protocol findings (must be surfaced for any reviewer)

* `zone == "In motion"` ↔ `mobility == "Yes"`: **True**
* Real Jamming only in Zone A: **True**
* Mobile sessions carry zero anomalies: **True**

### Zone × anomaly_origin (full corpus)

|           |   normal |   real |   synthetic |   ALL |
|:----------|---------:|-------:|------------:|------:|
| A         |     9948 |    279 |         311 | 10538 |
| B         |     9945 |      0 |         305 | 10250 |
| C         |     9945 |      0 |         340 | 10285 |
| In motion |      927 |      0 |           0 |   927 |
| ALL       |    30765 |    279 |         956 | 32000 |