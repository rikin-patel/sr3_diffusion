#ADDED: new file (FMGAI assignment) - three-run comparison table, paired differences and a visual grid.
"""Compare Run 0, Run A and Run B on the shared test set.

Writes assignment_results/comparison/ with a summary table (CSV and JSON),
paired per-image differences between runs, and a grid of example outputs.
"""
import csv
import json
import os

import cv2
import numpy as np

ROOT = 'assignment_results'
RUNS = [('Run 0', 'run_0_baseline'), ('Run A', 'run_a_original_schedule'), ('Run B', 'run_b_power_schedule')]
METRICS = [('psnr', 'higher'), ('ssim', 'higher'), ('lpips', 'lower')]


def load(run_dir):
    with open(os.path.join(ROOT, run_dir, 'metrics_per_image.csv')) as f:
        return {r['image']: r for r in csv.DictReader(f)}


def main():
    out = os.path.join(ROOT, 'comparison')
    os.makedirs(out, exist_ok=True)
    data = {label: load(d) for label, d in RUNS}
    images = list(data['Run 0'])
    assert all(list(v) == images for v in data.values()), 'runs were not evaluated on the same images'

    def col(label, key):
        return np.array([float(data[label][i][key]) for i in images])

    rows = [['method'] + ['{}_{}'.format(m, s) for m, _ in METRICS for s in ('mean', 'std')]]
    rows.append(['Bicubic'] + [f(col('Run 0', 'bicubic_' + m)) for m, _ in METRICS
                               for f in (lambda v: v.mean(), lambda v: v.std(ddof=1))])
    for label, _ in RUNS:
        rows.append([label] + [f(col(label, m)) for m, _ in METRICS
                               for f in (lambda v: v.mean(), lambda v: v.std(ddof=1))])
    with open(os.path.join(out, 'summary_table.csv'), 'w', newline='') as f:
        csv.writer(f).writerows(rows)

    paired = {}
    for a, b in (('Run 0', 'Run A'), ('Run 0', 'Run B'), ('Run A', 'Run B')):
        entry = {}
        for m, better in METRICS:
            d = col(b, m) - col(a, m)
            se = d.std(ddof=1) / np.sqrt(len(d))
            entry[m] = {
                'mean_difference': float(d.mean()), 'standard_error': float(se),
                't_statistic': float(d.mean() / se),
                'images_where_second_run_is_better': int((d > 0).sum() if better == 'higher' else (d < 0).sum()),
                'n_images': len(d), 'better_when': better,
            }
        paired['{} -> {}'.format(a, b)] = entry
    with open(os.path.join(out, 'paired_differences.json'), 'w') as f:
        json.dump(paired, f, indent=2)

    # Example grid: columns = bicubic | Run 0 | Run A | Run B | ground truth
    tiles = []
    for name in images[:8]:
        row = [cv2.imread(os.path.join(ROOT, 'test_inputs', 'bicubic_16_128', name + '.png'))]
        row += [cv2.imread(os.path.join(ROOT, d, 'test_outputs', 'sr', name + '.png')) for _, d in RUNS]
        row.append(cv2.imread(os.path.join(ROOT, 'test_inputs', 'hr_128', name + '.png')))
        tiles.append(np.concatenate(row, axis=1))
    cv2.imwrite(os.path.join(out, 'example_grid.png'), np.concatenate(tiles, axis=0))
    with open(os.path.join(out, 'example_grid_columns.json'), 'w') as f:
        json.dump({'columns': ['bicubic', 'Run 0', 'Run A', 'Run B', 'ground truth'], 'rows': images[:8]}, f, indent=2)

    for r in rows:
        print(', '.join(x if isinstance(x, str) else '{:.4f}'.format(x) for x in r))
    print(json.dumps(paired['Run A -> Run B'], indent=2))


if __name__ == '__main__':
    main()
