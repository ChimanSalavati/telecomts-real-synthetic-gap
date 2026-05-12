"""Builds the E1 dataset/split/leakage audit notebook.

Run::

    python experiments/E1_dataset_split_leakage_audit/build_notebook.py

This regenerates ``E1_dataset_split_leakage_audit.ipynb`` next to this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
sys.path.insert(0, str(EXP_ROOT))
from _shared.notebook_builder import NotebookBuilder  # noqa: E402


def build() -> Path:
    nb = NotebookBuilder("E1: Dataset, split, and leakage audit")

    nb.md("""# E1 — Dataset, split, and leakage audit

Goal: verify all dataset counts, anomaly-origin labels, context metadata, and split manifests
that the CIKM2026 paper relies on. Outputs a CSV/Markdown summary plus per-split CSV
manifests under ``manifests/`` and ``results/``.

This notebook does **not** train any model; it only audits the corpus and the five splits
defined in ``_shared/data_utils.py``. Re-run after any change to a split definition.
""")

    nb.code("""# Bootstrap: make _shared importable, set plot defaults, suppress noisy warnings.
from pathlib import Path
import sys
sys.path.insert(0, str(Path('..').resolve()))
from _shared.notebook_helpers import setup_paths, configure_matplotlib
EXPERIMENTS_ROOT = setup_paths()
configure_matplotlib()

import json
import numpy as np
import pandas as pd
from collections import Counter

from _shared.data_utils import (
    load_corpus,
    KPI_NAMES,
    ANOMALY_TYPES,
    REAL_ANOMALY_TYPES,
    SYNTHETIC_ANOMALY_TYPES,
    CONTEXT_FIELDS,
    all_splits,
)

HERE = Path('.').resolve()
RESULTS = HERE / 'results'
MANIFESTS = HERE / 'manifests'
RESULTS.mkdir(exist_ok=True)
MANIFESTS.mkdir(exist_ok=True)
print('Working dir:', HERE)
""")

    nb.md("## 1. Load the full corpus and check basic counts")

    nb.code("""corpus = load_corpus(verbose=True)
n_total = corpus.n
n_norm = int((corpus.y == 0).sum())
n_anom = int((corpus.y == 1).sum())
print(f'Total windows         : {n_total:,}')
print(f'Normal windows        : {n_norm:,}')
print(f'Anomalous windows     : {n_anom:,}')
assert n_total == n_norm + n_anom, 'Counts do not add up'

type_counts = Counter(corpus.anomaly_type[corpus.y == 1])
type_table = pd.DataFrame(
    [(t, type_counts.get(t, 0)) for t in ANOMALY_TYPES],
    columns=['anomaly_type', 'count'],
)
print('\\nAnomaly types and counts (in canonical order):')
print(type_table.to_string(index=False))
print(f'\\nTotal across types    : {int(type_table["count"].sum()):,}')
""")

    nb.code("""real_mask = corpus.anomaly_origin == 'real'
syn_mask = corpus.anomaly_origin == 'synthetic'
n_real = int(real_mask.sum())
n_syn = int(syn_mask.sum())
print(f'Real (Jamming)        : {n_real}')
print(f'Synthetic (other 10)  : {n_syn}')
print(f'Real + synthetic      : {n_real + n_syn}')
assert n_real + n_syn == n_anom, 'real+synthetic must equal anomaly count'
assert set(REAL_ANOMALY_TYPES) == {'Jamming'}, 'real definition must remain Jamming-only'
""")

    nb.code("""# Verify shape and existence of context fields.
assert corpus.X.shape[1:] == (128, 16), corpus.X.shape
print('Window shape          :', corpus.X.shape[1:])
for field in CONTEXT_FIELDS:
    arr = getattr(corpus, field)
    assert arr is not None and arr.shape[0] == n_total
    print(f'context field {field:>11s} ok  | sample values: {sorted(set(arr.tolist()))[:6]}')
""")

    nb.md("""## 2. Build all five splits and write manifests

Splits are constructed deterministically by ``_shared/data_utils.py``. The audit below verifies:
* train/val/test sizes match the paper.
* No sample id appears in more than one partition.
* Test windows never appear in train or validation.
* Per-partition normal/anomaly, real/synthetic, anomaly-type, and context counts.
""")

    nb.code("""splits = all_splits(corpus, seed=42)
for name, parts in splits.items():
    sizes = {k: int(v.size) for k, v in parts.items()}
    print(f'{name:>20s}  sizes={sizes}')
""")

    nb.code("""def _summarise_partition(idx, partition_name, split_name):
    sel = corpus
    return {
        'split': split_name,
        'partition': partition_name,
        'n': int(idx.size),
        'n_normal': int((sel.y[idx] == 0).sum()),
        'n_anomaly': int((sel.y[idx] == 1).sum()),
        'n_real': int((sel.anomaly_origin[idx] == 'real').sum()),
        'n_synthetic': int((sel.anomaly_origin[idx] == 'synthetic').sum()),
    }


def _manifest(idx, split_name, partition_name):
    return pd.DataFrame({
        'sample_id': corpus.sample_id[idx],
        'split': split_name,
        'partition': partition_name,
        'y': corpus.y[idx],
        'anomaly_type': corpus.anomaly_type[idx],
        'anomaly_origin': corpus.anomaly_origin[idx],
        'zone': corpus.zone[idx],
        'application': corpus.application[idx],
        'mobility': corpus.mobility[idx],
        'congestion': corpus.congestion[idx],
    })


# Map paper labels -> file names.
manifest_filenames = {
    'small_natural':       {'train': 'small_natural_train.csv',       'val': 'small_natural_val.csv',       'test': 'small_natural_test.csv'},
    'balanced_detection':  {'train': 'balanced_detection_train.csv',  'val': 'balanced_detection_val.csv',  'test': 'balanced_detection_test.csv'},
    'controlled_500':      {'train': 'controlled_500_train.csv',      'val': 'controlled_500_val.csv',      'test': 'controlled_500_test.csv'},
    'fullscale':           {'train': 'fullscale_train.csv',                                                  'test': 'fullscale_test.csv'},
    'split_rca_balanced':  {'train': 'split_rca_balanced_train.csv',                                          'test': 'split_rca_balanced_test.csv'},
}

summary_rows = []
for split_name, parts in splits.items():
    for partition_name, idx in parts.items():
        summary_rows.append(_summarise_partition(idx, partition_name, split_name))
        man = _manifest(idx, split_name, partition_name)
        fname = manifest_filenames[split_name].get(partition_name)
        if fname is not None:
            man.to_csv(MANIFESTS / fname, index=False)

summary_df = pd.DataFrame(summary_rows)
summary_df
""")

    nb.code("""# Anomaly-type counts per partition (long format), then context counts.
type_rows = []
context_rows = []
for split_name, parts in splits.items():
    for partition_name, idx in parts.items():
        sub_types = pd.Series(corpus.anomaly_type[idx])
        for t in ANOMALY_TYPES:
            type_rows.append({
                'split': split_name,
                'partition': partition_name,
                'anomaly_type': t,
                'count': int((sub_types == t).sum()),
            })
        for field in CONTEXT_FIELDS:
            sub_field = pd.Series(getattr(corpus, field)[idx])
            for value, count in sub_field.value_counts().items():
                context_rows.append({
                    'split': split_name,
                    'partition': partition_name,
                    'context_field': field,
                    'value': value,
                    'count': int(count),
                })
type_counts_df = pd.DataFrame(type_rows)
context_counts_df = pd.DataFrame(context_rows)
print('anomaly-type rows:', len(type_counts_df))
print('context rows     :', len(context_counts_df))
type_counts_df.head(20)
""")

    nb.md("## 3. Leakage checks — no overlap across train/val/test, no test in train")

    nb.code("""leakage_rows = []
for split_name, parts in splits.items():
    ids = {p: set(corpus.sample_id[idx].tolist()) for p, idx in parts.items()}
    pairs = []
    keys = list(ids)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            inter = ids[a] & ids[b]
            pairs.append({
                'split': split_name,
                'pair': f'{a}-{b}',
                'overlap': len(inter),
            })
    leakage_rows.extend(pairs)

leakage_df = pd.DataFrame(leakage_rows)
print(leakage_df.to_string(index=False))
assert (leakage_df['overlap'] == 0).all(), 'Leakage detected: overlapping sample IDs!'
print('\\nNo leakage detected for any split.')
""")

    nb.code("""# Specific paper assertions: confirm headline test counts.
assertions = {
    'small_natural test == 200': int(splits['small_natural']['test'].size) == 200,
    'small_natural train == 640': int(splits['small_natural']['train'].size) == 640,
    'small_natural val == 160': int(splits['small_natural']['val'].size) == 160,
    'balanced_detection train == 1580': int(splits['balanced_detection']['train'].size) == 1580,
    'balanced_detection val == 396': int(splits['balanced_detection']['val'].size) == 396,
    'balanced_detection test == 494': int(splits['balanced_detection']['test'].size) == 494,
    'controlled_500 train == 300': int(splits['controlled_500']['train'].size) == 300,
    'controlled_500 val == 100': int(splits['controlled_500']['val'].size) == 100,
    'controlled_500 test == 100': int(splits['controlled_500']['test'].size) == 100,
    'fullscale train == 25600': int(splits['fullscale']['train'].size) == 25600,
    'fullscale test == 6400': int(splits['fullscale']['test'].size) == 6400,
    'split_rca_balanced train == 988': int(splits['split_rca_balanced']['train'].size) == 988,
    'split_rca_balanced test == 247': int(splits['split_rca_balanced']['test'].size) == 247,
}
for k, v in assertions.items():
    print(('OK   ' if v else 'FAIL ') + k)
assert all(assertions.values()), 'Some headline split sizes do not match.'
""")

    nb.md("""## 4. Collection-protocol findings (must be surfaced for any reviewer)

Two characteristics of the TelecomTS *collection protocol* affect every downstream claim
about real-vs-synthetic gap and context generalization. They are not bugs — they are
documented in TelecomTS Appendix B and Table 7. We surface them here so any reviewer can
verify them and so later experiments (E2, E6) can run zone-restricted variants.

1. ``zone == "In motion"`` is *equivalent to* ``mobility == "Yes"``. Mobile sessions
   cannot be assigned to a fixed zone, so the released labels use the string
   ``"In motion"`` as the zone. There are 3 fixed zones (A / B / C) plus this
   no-fixed-zone bucket.
2. **Real Jamming was collected only in Zone A** (the jammer was placed near the RU,
   per Table 7). Synthetic anomalies span all three fixed zones. Mobile sessions
   contain no anomalies of any kind.

Implication for the paper: any "real RSRP ≈ -76 dB vs synthetic RSRP ≈ -109 dB"
gap mixes (a) a real-vs-synthetic mechanism difference and (b) the expected
Zone-A-vs-averaged-A/B/C path-loss difference. E2 should report both the full
context-matched gap **and** a Zone-A-restricted variant.
""")

    nb.code("""from _shared.data_utils import context_collection_findings, ZONE_IN_MOTION

findings = context_collection_findings(corpus)
print('Zone counts                         :', findings['zone_counts'])
print('zone == "In motion" iff mobility=Yes:', findings['zone_in_motion_iff_mobility_yes'])
print('real Jamming only in Zone A         :', findings['real_jamming_only_in_zone_A'])
print('zero anomalies during mobile        :', findings['no_anomalies_during_mobility'])

print('\\nZone × anomaly_origin (full corpus):')
zo_df = pd.DataFrame(findings['zone_by_anomaly_origin']).T.fillna(0).astype(int)
zo_df = zo_df.reindex(['A', 'B', 'C', ZONE_IN_MOTION])
zo_df['ALL'] = zo_df.sum(axis=1)
zo_df.loc['ALL'] = zo_df.sum(axis=0)
print(zo_df.to_string())

# Persist for later experiments to reference.
import json
(RESULTS / 'E1_collection_findings.json').write_text(json.dumps(findings, indent=2))
zo_df.to_csv(RESULTS / 'E1_zone_by_anomaly_origin.csv')

assert findings['zone_in_motion_iff_mobility_yes'], 'zone="In motion" should match mobility=Yes'
assert findings['real_jamming_only_in_zone_A'], 'real Jamming should be Zone A only'
assert findings['no_anomalies_during_mobility'], 'mobile sessions should have no anomalies'
""")

    nb.md("""## 5. Write summary outputs

* ``results/E1_dataset_split_leakage_audit.csv`` — partition sizes + counts (one row per partition).
* ``results/E1_split_anomaly_type_counts.csv`` — long-format anomaly-type counts.
* ``results/E1_split_context_counts.csv`` — long-format context counts.
* ``results/E1_leakage_checks.csv`` — pairwise overlap audit.
* ``results/E1_collection_findings.json`` — collection-protocol findings (Zone A confound, In-motion encoding).
* ``results/E1_zone_by_anomaly_origin.csv`` — zone × origin contingency table.
* ``results/E1_dataset_split_leakage_audit.md`` — Markdown summary.
""")

    nb.code("""summary_df.to_csv(RESULTS / 'E1_dataset_split_leakage_audit.csv', index=False)
type_counts_df.to_csv(RESULTS / 'E1_split_anomaly_type_counts.csv', index=False)
context_counts_df.to_csv(RESULTS / 'E1_split_context_counts.csv', index=False)
leakage_df.to_csv(RESULTS / 'E1_leakage_checks.csv', index=False)

md_lines = ['# E1 — Dataset, split, and leakage audit', '']
md_lines.append(f'* Total windows: **{n_total:,}**')
md_lines.append(f'* Normal: **{n_norm:,}**, Anomaly: **{n_anom:,}**')
md_lines.append(f'* Real Jamming: **{n_real}**, Synthetic anomalies: **{n_syn}**')
md_lines.append('')
md_lines.append('## Split sizes')
md_lines.append('')
md_lines.append(summary_df.to_markdown(index=False))
md_lines.append('')
md_lines.append('## Leakage checks (sample-id overlap between partitions per split)')
md_lines.append('')
md_lines.append(leakage_df.to_markdown(index=False))
md_lines.append('')
md_lines.append('## Collection-protocol findings (must be surfaced for any reviewer)')
md_lines.append('')
md_lines.append(f'* `zone == "In motion"` ↔ `mobility == "Yes"`: **{findings["zone_in_motion_iff_mobility_yes"]}**')
md_lines.append(f'* Real Jamming only in Zone A: **{findings["real_jamming_only_in_zone_A"]}**')
md_lines.append(f'* Mobile sessions carry zero anomalies: **{findings["no_anomalies_during_mobility"]}**')
md_lines.append('')
md_lines.append('### Zone × anomaly_origin (full corpus)')
md_lines.append('')
md_lines.append(zo_df.to_markdown())
(RESULTS / 'E1_dataset_split_leakage_audit.md').write_text('\\n'.join(md_lines))
print('Saved summary files in', RESULTS)
""")

    out = HERE / "E1_dataset_split_leakage_audit.ipynb"
    return nb.write(out)


if __name__ == "__main__":
    p = build()
    print(f"Wrote notebook to {p}")
