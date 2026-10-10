# ADDED: new file (FMGAI assignment) - copies a fine-tuning run's logs and curves into assignment_results/.
"""Collect what Phase 2 needs from an experiments/<name>_<timestamp>/ folder.

Copies train.log / val.log and the TensorBoard events, turns the logs into CSV
files and curves, and builds a montage of validation samples over training.
Checkpoints stay in experiments/ (too large for git); their path is recorded.
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TRAIN_RE = re.compile(r'<epoch:\s*(\d+), iter:\s*([\d,]+)> l_pix: ([\d.e+-]+)')
VAL_RE = re.compile(r'<epoch:\s*(\d+), iter:\s*([\d,]+)> psnr: ([\d.e+-]+), ssim: ([\d.e+-]+)')


def latest_experiment(name):
    runs = sorted(d for d in glob.glob(os.path.join('experiments', name + '_*'))
                  if os.path.isfile(os.path.join(d, 'logs', 'train.log')))
    if not runs:
        raise SystemExit('No experiments/{}_* folder with a train.log found'.format(name))
    return runs[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='config "name", e.g. kvasir_runA_original')
    parser.add_argument('--run_dir', required=True, help='e.g. assignment_results/run_a_original_schedule')
    parser.add_argument('--exp', default=None, help='experiment folder; default: latest for --name')
    parser.add_argument('--n_montage', type=int, default=4)
    args = parser.parse_args()

    exp = args.exp or latest_experiment(args.name)
    logs_out = os.path.join(args.run_dir, 'logs')
    os.makedirs(logs_out, exist_ok=True)
    for f in glob.glob(os.path.join(exp, 'logs', '*.log')):
        shutil.copy2(f, logs_out)
    tb_out = os.path.join(args.run_dir, 'tb_logger')
    os.makedirs(tb_out, exist_ok=True)
    for f in glob.glob(os.path.join(exp, 'tb_logger', 'events.*')):
        shutil.copy2(f, tb_out)

    train_rows, val_rows = [], []
    with open(os.path.join(exp, 'logs', 'train.log'), encoding='utf-8', errors='replace') as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                train_rows.append((int(m.group(1)), int(m.group(2).replace(',', '')), float(m.group(3))))
    with open(os.path.join(exp, 'logs', 'val.log'), encoding='utf-8', errors='replace') as f:
        for line in f:
            m = VAL_RE.search(line)
            if m:
                val_rows.append((int(m.group(1)), int(m.group(2).replace(',', '')),
                                 float(m.group(3)), float(m.group(4))))

    with open(os.path.join(args.run_dir, 'train_loss.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([('epoch', 'iteration', 'l_pix')] + train_rows)
    with open(os.path.join(args.run_dir, 'validation_metrics.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([('epoch', 'iteration', 'psnr', 'ssim')] + val_rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    it = np.array([r[1] for r in train_rows])
    loss = np.array([r[2] for r in train_rows])
    axes[0].plot(it, loss, alpha=0.35, color='#1f4e79', label='logged every print_freq')
    if len(loss) >= 10:
        w = 10
        axes[0].plot(it[w - 1:], np.convolve(loss, np.ones(w) / w, mode='valid'),
                     color='#1f4e79', label='moving average ({})'.format(w))
    axes[0].set_title('Training loss (L1 on predicted noise)')
    axes[0].legend(frameon=False)
    axes[1].plot([r[1] for r in val_rows], [r[2] for r in val_rows], marker='o', color='#c0392b')
    axes[1].set_title('Validation PSNR (dB)')
    axes[2].plot([r[1] for r in val_rows], [r[3] for r in val_rows], marker='o', color='#27ae60')
    axes[2].set_title('Validation SSIM')
    for ax in axes:
        ax.set_xlabel('iteration')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.run_dir, 'training_curves.png'), dpi=180)

    # Montage: one row per validation pass, the same few validation images in every row.
    montage_rows, steps = [], []
    for step in sorted({r[1] for r in val_rows}):
        tiles = []
        for i in range(1, args.n_montage + 1):
            hits = glob.glob(os.path.join(exp, 'results', '*', '{}_{}_sr.png'.format(step, i)))
            if hits:
                tiles.append(cv2.imread(hits[0]))
        if len(tiles) == args.n_montage:
            montage_rows.append(np.concatenate(tiles, axis=1))
            steps.append(step)
    if montage_rows:
        gt = [cv2.imread(glob.glob(os.path.join(exp, 'results', '*', '{}_{}_hr.png'.format(steps[0], i)))[0])
              for i in range(1, args.n_montage + 1)]
        cv2.imwrite(os.path.join(args.run_dir, 'validation_progress.png'),
                    np.concatenate(montage_rows + [np.concatenate(gt, axis=1)], axis=0))

    checkpoints = sorted(os.path.basename(p) for p in glob.glob(os.path.join(exp, 'checkpoint', '*_gen.pth')))
    with open(os.path.join(args.run_dir, 'training_run_info.json'), 'w') as f:
        json.dump({'experiment_dir': exp.replace('\\', '/'), 'checkpoints': checkpoints,
                   'logged_train_points': len(train_rows), 'validation_passes': len(val_rows),
                   'validation_progress_rows': ['iteration {}'.format(s) for s in steps] + ['ground truth'],
                   'final_train_loss': train_rows[-1][2] if train_rows else None,
                   'final_val_psnr': val_rows[-1][2] if val_rows else None,
                   'final_val_ssim': val_rows[-1][3] if val_rows else None}, f, indent=2)
    print('collected', exp, '->', args.run_dir)


if __name__ == '__main__':
    main()
