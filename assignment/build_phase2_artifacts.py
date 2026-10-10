# ADDED: new file (FMGAI assignment) - builds the Phase 2 deliverables from the recorded runs.
"""Phase 2 submission artifacts, all derived from the three recorded runs.

Writes to assignment_results/phase2/:

    finetuning_logs/        runA_train_loss.csv, runB_train_loss.csv (epoch, train_loss)
    checkpoint_samples/     generated samples at epochs 20, 40, 60, 80, 100 (labelled)
    inference_trajectories/ x_t at t = 1800, 1400, 1000, 600, 200 from the final models (labelled)
    figures/                figures used in the report
    tables/                 numbers used in the report

Nothing is re-sampled here. Checkpoint samples are the validation samples drawn
during fine-tuning; trajectories are the snapshots saved by assignment/evaluate.py.
"""
import csv
import json
import os
import subprocess
import sys

import cv2
import lpips
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.logger as Logger  # noqa: E402
import core.metrics as Metrics  # noqa: E402
import model as Model  # noqa: E402
from assignment.evaluate import load_config  # noqa: E402
from sr import SCHEDULE_EXPONENT_X, apply_schedule_exponent  # noqa: E402

ROOT = 'assignment_results'
OUT = os.path.join(ROOT, 'phase2')
RUNS = {
    'Run A': {'dir': 'run_a_original_schedule', 'tag': 'runA', 'config': 'config/kvasir_runA_original.json',
              'color': '#1f4e79', 'schedule': r'$\bar\alpha_t$'},
    'Run B': {'dir': 'run_b_power_schedule', 'tag': 'runB', 'config': 'config/kvasir_runB_power.json',
              'color': '#c0392b', 'schedule': r'$\bar\alpha_t^{\,%.2f}$' % SCHEDULE_EXPONENT_X},
}
SAMPLE_EPOCHS = [20, 40, 60, 80, 100]          # 5 equally spaced fine-tuning epochs
SAMPLE_IDS = [1, 2, 3]                         # first three validation images
TRAJ_T = [1800, 1400, 1000, 600, 200]          # 5 equally spaced reverse-process timesteps
ITERS_PER_EPOCH = 200                          # 800 training images / batch size 4
TILE = 128
BASE_COMMIT = '01d27a7'                        # upstream commit the fork started from


def rgb(path):
    img = cv2.imread(path)
    assert img is not None, path
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def labelled(img, text, scale=2):
    """Upscale with nearest neighbour and add a caption bar so a single file is self-describing."""
    big = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    bar = np.full((22, big.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([bar, big], axis=0)


def save_rgb(path, img):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def image_grid(cells, col_titles, row_titles, path, suptitle=None, cell=1.45):
    rows, cols = len(cells), len(cells[0])
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell + 0.5, rows * cell + (0.75 if suptitle else 0.4)),
                             squeeze=False)
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            ax.imshow(cells[r][c], interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9.5)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=9.5)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=0.995)
    fig.tight_layout(pad=0.4, w_pad=0.15, h_pad=0.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def experiment_dir(run):
    with open(os.path.join(ROOT, RUNS[run]['dir'], 'training_run_info.json')) as f:
        return json.load(f)['experiment_dir']


# ---------------------------------------------------------------- fine-tuning logs
def finetuning_logs():
    out = os.path.join(OUT, 'finetuning_logs')
    os.makedirs(out, exist_ok=True)
    curves = {}
    for run, info in RUNS.items():
        per_epoch = {}
        with open(os.path.join(ROOT, info['dir'], 'train_loss.csv')) as f:
            for r in csv.DictReader(f):
                per_epoch.setdefault(int(r['epoch']), []).append(float(r['l_pix']))
        epochs = sorted(per_epoch)
        loss = [float(np.mean(per_epoch[e])) for e in epochs]
        with open(os.path.join(out, '{}_train_loss.csv'.format(info['tag'])), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'train_loss'])
            w.writerows(zip(epochs, ['{:.6f}'.format(v) for v in loss]))
        curves[run] = (np.array(epochs), np.array(loss), [len(per_epoch[e]) for e in epochs])

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    for run, (ep, loss, _) in curves.items():
        c = RUNS[run]['color']
        ax.plot(ep, loss, color=c, alpha=0.25, lw=1)
        w = 10
        ax.plot(ep[w - 1:], np.convolve(loss, np.ones(w) / w, mode='valid'), color=c, lw=2,
                label='{} ({}), 10-epoch moving average'.format(run, RUNS[run]['schedule']))
    ax.set_yscale('log')
    ax.set_xlabel('fine-tuning epoch')
    ax.set_ylabel('train_loss (L1 on predicted noise)')
    ax.set_title('Fine-tuning loss, Run A and Run B (thin lines: per-epoch values)', fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figures', 'loss_curves.png'), dpi=200)
    plt.close(fig)

    stats = {}
    for run, (ep, loss, n) in curves.items():
        stats[run] = {'epochs': int(len(ep)), 'logged_batches_per_epoch': sorted(set(n)),
                      'mean_all_epochs': float(loss.mean()), 'median_all_epochs': float(np.median(loss)),
                      'mean_first_10_epochs': float(loss[:10].mean()), 'mean_last_10_epochs': float(loss[-10:].mean()),
                      'mean_epochs_1_50': float(loss[:50].mean()), 'mean_epochs_51_100': float(loss[50:].mean())}
    return stats


# ---------------------------------------------------------------- checkpoint samples
def checkpoint_samples():
    out = os.path.join(OUT, 'checkpoint_samples')
    os.makedirs(out, exist_ok=True)
    lp = lpips.LPIPS(net='alex', verbose=False).cuda().eval()

    def lpips_of(a, b):
        prep = lambda x: (torch.from_numpy(x).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0).cuda()
        with torch.no_grad():
            return float(lp(prep(a), prep(b)))

    exp = {run: experiment_dir(run) for run in RUNS}
    for run, info in RUNS.items():
        cells = []
        for i in SAMPLE_IDS:
            first = os.path.join(exp[run], 'results', str(SAMPLE_EPOCHS[0]),
                                 '{}_{}'.format(SAMPLE_EPOCHS[0] * ITERS_PER_EPOCH, i))
            row = [rgb(first + '_inf.png')]
            for ep in SAMPLE_EPOCHS:
                img = rgb(os.path.join(exp[run], 'results', str(ep), '{}_{}_sr.png'.format(ep * ITERS_PER_EPOCH, i)))
                row.append(img)
                save_rgb(os.path.join(out, '{}_epoch{:03d}_sample{}.png'.format(info['tag'], ep, i)),
                         labelled(img, '{} | epoch {} | sample {}'.format(run, ep, i)))
            row.append(rgb(first + '_hr.png'))
            cells.append(row)
        image_grid(cells,
                   ['bicubic input'] + ['{}, epoch {}'.format(run, ep) for ep in SAMPLE_EPOCHS] + ['ground truth'],
                   ['val. sample {}'.format(i) for i in SAMPLE_IDS],
                   os.path.join(OUT, 'figures', 'checkpoint_samples_{}.png'.format(info['tag'])),
                   suptitle='{}: generated samples during fine-tuning (schedule {})'.format(run, info['schedule']))

    # paired comparison of the two runs on the 20 validation images, every validation epoch
    rows = []
    for ep in range(10, 101, 10):
        it = ep * ITERS_PER_EPOCH
        m = {run: {'psnr': [], 'ssim': [], 'lpips': []} for run in RUNS}
        for i in range(1, 21):
            hr = rgb(os.path.join(exp['Run A'], 'results', str(ep), '{}_{}_hr.png'.format(it, i)))
            for run in RUNS:
                sr = rgb(os.path.join(exp[run], 'results', str(ep), '{}_{}_sr.png'.format(it, i)))
                m[run]['psnr'].append(Metrics.calculate_psnr(sr, hr))
                m[run]['ssim'].append(Metrics.calculate_ssim(sr, hr))
                m[run]['lpips'].append(lpips_of(sr, hr))
        row = {'epoch': ep, 'iteration': it, 'n_images': 20}
        for key in ('psnr', 'ssim', 'lpips'):
            a, b = np.array(m['Run A'][key]), np.array(m['Run B'][key])
            d = b - a
            row.update({'runA_' + key: a.mean(), 'runB_' + key: b.mean(), 'diff_' + key: d.mean(),
                        'se_diff_' + key: d.std(ddof=1) / np.sqrt(len(d))})
        row['runB_better_psnr_count'] = int((np.array(m['Run B']['psnr']) > np.array(m['Run A']['psnr'])).sum())
        rows.append(row)
    with open(os.path.join(OUT, 'tables', 'validation_by_epoch.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ('{:.5f}'.format(v) if isinstance(v, float) else v) for k, v in r.items()})
    return rows


# ---------------------------------------------------------------- inference trajectories
def inference_trajectories():
    out = os.path.join(OUT, 'inference_trajectories')
    os.makedirs(out, exist_ok=True)
    for run, info in RUNS.items():
        tdir = os.path.join(ROOT, info['dir'], 'trajectories')
        with open(os.path.join(tdir, 'columns.json')) as f:
            meta = json.load(f)
        cols = meta['columns']                       # bicubic_input, 'x_t at t=2000', ..., ground_truth
        cells = []
        for n, name in enumerate(meta['images'][:3], 1):
            strip = rgb(os.path.join(tdir, name + '.png'))
            tile = lambda label: strip[:, cols.index(label) * TILE:(cols.index(label) + 1) * TILE]
            row = [tile('bicubic_input')]
            for t in TRAJ_T:
                img = tile('x_t at t={}'.format(t))
                row.append(img)
                save_rgb(os.path.join(out, '{}_t{:04d}_sample{}.png'.format(info['tag'], t, n)),
                         labelled(img, '{} | t = {} | sample {}'.format(run, t, n)))
            row += [tile('x_t at t=0'), tile('ground_truth')]
            cells.append(row)
        image_grid(cells,
                   ['bicubic input'] + ['{}, t = {}'.format(run, t) for t in TRAJ_T] + ['final output (t = 0)', 'ground truth'],
                   ['test sample {}'.format(n) for n in (1, 2, 3)],
                   os.path.join(OUT, 'figures', 'trajectory_{}.png'.format(info['tag'])),
                   suptitle='{}: noisy sample $x_t$ during the reverse process, final model (schedule {})'.format(
                       run, info['schedule']))
        with open(os.path.join(out, '{}_samples.json'.format(info['tag'])), 'w') as f:
            json.dump({'run': run, 'timesteps': TRAJ_T, 'test_images': meta['images'][:3],
                       'note': 'x_t is the state after the reverse step that produces timestep t; '
                               't = 2000 is the initial Gaussian noise and t = 0 the final output.'}, f, indent=2)


# ---------------------------------------------------------------- result figures
def result_figures():
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, 'test_inputs', 'hr_128')))[:8]
    inp = lambda sub, n: rgb(os.path.join(ROOT, 'test_inputs', sub, n + '.png'))
    out = lambda run_dir, n: rgb(os.path.join(ROOT, run_dir, 'test_outputs', 'sr', n + '.png'))
    up = lambda img: cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
    titles = ['test {}'.format(i) for i in range(1, 9)]
    image_grid([[up(inp('lr_16', n)) for n in names], [inp('bicubic_16_128', n) for n in names],
                [out('run_0_baseline', n) for n in names], [inp('hr_128', n) for n in names]],
               titles, ['input 16×16', 'bicubic', 'Run 0 output', 'ground truth'],
               os.path.join(OUT, 'figures', 'run0_samples.png'),
               suptitle='Run 0: pretrained SR3 applied to Kvasir-SEG test images without fine-tuning', cell=1.25)
    image_grid([[inp('bicubic_16_128', n) for n in names], [out('run_0_baseline', n) for n in names],
                [out('run_a_original_schedule', n) for n in names], [out('run_b_power_schedule', n) for n in names],
                [inp('hr_128', n) for n in names]],
               titles, ['bicubic', 'Run 0', 'Run A', 'Run B', 'ground truth'],
               os.path.join(OUT, 'figures', 'three_runs.png'),
               suptitle='Same test images and same starting noise in all three runs', cell=1.25)
    return names


# ---------------------------------------------------------------- schedule
def schedule_figure():
    x = SCHEDULE_EXPONENT_X
    T = 2000
    betas = np.linspace(1e-6, 1e-2, T, dtype=np.float64)
    theta = np.cumprod(1. - betas)
    theta_x = theta ** x
    t = np.arange(1, T + 1)
    snr, snr_x = theta / (1 - theta), theta_x / (1 - theta_x)
    first = lambda cond: int(t[np.argmax(cond)])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    axes[0].plot(t, theta, color='#1f4e79', lw=2, label=r'$\theta_t=\bar\alpha_t$ (Run 0, Run A)')
    axes[0].plot(t, theta_x, color='#c0392b', lw=2, label=r'$\theta_t^{\,x}=\bar\alpha_t^{\,%.2f}$ (Run B)' % x)
    axes[0].set_ylabel(r'noise parameter')
    axes[0].set_title(r'$\{\theta_t\}$ and $\{\theta_t^{\,x}\}$ against $t$')
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].semilogy(t, snr, color='#1f4e79', lw=2, label='native')
    axes[1].semilogy(t, snr_x, color='#c0392b', lw=2, label='x = %.2f' % x)
    axes[1].axhline(1.0, color='grey', lw=0.8, ls='--')
    axes[1].axhline(snr[-1], color='grey', lw=0.8, ls=':')
    axes[1].text(40, snr[-1] * 2.2, 'lowest SNR seen in pretraining', fontsize=7, color='dimgrey')
    axes[1].text(1420, 1.6, 'SNR = 1', fontsize=7, color='dimgrey')
    axes[1].set_ylabel(r'SNR$(t)=\theta_t/(1-\theta_t)$')
    axes[1].set_title('Signal-to-noise ratio')
    axes[1].legend(frameon=False, fontsize=8.5)
    for ax in axes:
        ax.set_xlabel('timestep t')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figures', 'schedule.png'), dpi=200)
    plt.close(fig)

    t_cross = first(theta_x <= theta[-1])
    info = {
        'x': x, 'T': T,
        'theta_range_native': [float(theta.max()), float(theta.min())],
        'theta_range_power': [float(theta_x.max()), float(theta_x.min())],
        'all_in_open_unit_interval': bool(((theta > 0) & (theta < 1) & (theta_x > 0) & (theta_x < 1)).all()),
        'beta_max_native': float(betas.max()),
        'beta_max_power': float((1 - (1 - betas) ** x).max()),
        't_where_snr_drops_below_1': {'native': first(snr < 1), 'power': first(snr_x < 1)},
        't_where_theta_drops_below_0.5': {'native': first(theta < 0.5), 'power': first(theta_x < 0.5)},
        'first_t_where_power_schedule_is_below_native_terminal_level': t_cross,
        'timesteps_of_power_schedule_below_native_terminal_level': int((theta_x < theta[-1]).sum()),
        'fraction_below_native_terminal_level': float((theta_x < theta[-1]).mean()),
        'timesteps_with_snr_above_1': {'native': int((snr > 1).sum()), 'power': int((snr_x > 1).sum())},
        'timesteps_with_theta_above_0.9': {'native': int((theta > 0.9).sum()), 'power': int((theta_x > 0.9).sum())},
        'snr_ratio_power_over_native_at': {str(k): float(snr_x[k - 1] / snr[k - 1]) for k in (1, 250, 500, 1000, 1500, 2000)},
        'snr_at': {str(k): {'native': float(snr[k - 1]), 'power': float(snr_x[k - 1])} for k in (1, 250, 500, 1000, 1500, 2000)},
        'theta_at': {str(k): {'native': float(theta[k - 1]), 'power': float(theta_x[k - 1])} for k in (1, 250, 500, 1000, 1500, 2000)},
        'equivalent_native_timestep_tau': {str(k): first(theta <= theta_x[k - 1]) for k in (200, 400, 600, 800, 1000, 1200)},
        'sqrt_theta_min_native': float(np.sqrt(theta.min())), 'sqrt_theta_min_power': float(np.sqrt(theta_x.min())),
        'theta_at_trajectory_timesteps': {str(k): {'native': float(theta[k - 1]), 'power': float(theta_x[k - 1])}
                                          for k in TRAJ_T},
    }
    return info


# ---------------------------------------------------------------- verification
def schedule_verification():
    betas = np.linspace(1e-6, 1e-2, 2000, dtype=np.float64)
    native = np.cumprod(1 - betas)
    out = {'SCHEDULE_EXPONENT_X': SCHEDULE_EXPONENT_X}
    for run, info in RUNS.items():
        ck = os.path.join(experiment_dir(run), 'checkpoint', 'I20000_E100_gen.pth')
        opt = apply_schedule_exponent(load_config(info['config']))
        opt['phase'], opt['distributed'] = 'val', False
        opt['path']['resume_state'] = None
        net = Model.create_model(Logger.dict_to_nonedict(opt)).netG
        stored = torch.load(ck, map_location='cpu', weights_only=True)['alphas_cumprod'].numpy()
        s64 = stored.astype(np.float64)
        with open(os.path.join(ROOT, info['dir'], 'eval_config.json')) as f:
            ev = json.load(f)
        out[run] = {
            'checkpoint': ck.replace('\\', '/'),
            'theta_power_resolved_from_constant': net.theta_power,
            'stored_alpha_bar_1': float(s64[0]), 'stored_alpha_bar_T': float(s64[-1]),
            'stored_buffer_bitwise_equal_to_schedule_built_from_constant':
                bool((stored == net.alphas_cumprod.cpu().numpy()).all()),
            'stored_buffer_matches_native': bool(np.allclose(s64, native, rtol=1e-5, atol=0)),
            'stored_buffer_matches_native_pow_x': bool(np.allclose(s64, native ** SCHEDULE_EXPONENT_X, rtol=1e-5, atol=0)),
            'test_eval_theta_power': ev['theta_power'], 'test_eval_alpha_bar_T': ev['alpha_bar_T'],
        }
    return out


# ---------------------------------------------------------------- environment, dataset, error analysis
def environment():
    import importlib.metadata as md
    import platform
    out = {'python': platform.python_version(), 'os': platform.platform()}
    for pkg in ('torch', 'torchvision', 'numpy', 'opencv-python', 'Pillow', 'tqdm', 'tensorboardX', 'lmdb', 'wandb',
                'lpips', 'matplotlib', 'pandas', 'reportlab', 'gdown', 'scipy'):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = 'not installed'
    out.update({'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                'gpu': torch.cuda.get_device_name(0),
                'gpu_memory_gb': round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)})
    with open(os.path.join(OUT, 'tables', 'environment.json'), 'w') as f:
        json.dump(out, f, indent=2)
    return out


def dataset_stats(raw='kvasir_data/raw/images', splits='kvasir_data/splits'):
    import collections
    import hashlib
    from PIL import Image
    files = sorted(os.listdir(raw))
    sizes, modes, hashes = collections.Counter(), collections.Counter(), {}
    for name in files:
        path = os.path.join(raw, name)
        with Image.open(path) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
        with open(path, 'rb') as fh:
            hashes.setdefault(hashlib.sha256(fh.read()).hexdigest(), []).append(name)
    split = {s: [l.strip() for l in open(os.path.join(splits, s + '.txt')) if l.strip()] for s in ('train', 'val', 'test')}
    names = sum(split.values(), [])
    where = {n: s for s, members in split.items() for n in members}
    dups = [v for v in hashes.values() if len(v) > 1]
    widths, heights = [s[0] for s in sizes.elements()], [s[1] for s in sizes.elements()]
    out = {'n_images': len(files), 'modes': dict(modes), 'distinct_resolutions': len(sizes),
           'width_min_max': [min(widths), max(widths)], 'height_min_max': [min(heights), max(heights)],
           'smallest_image_wxh': list(min(sizes, key=lambda s: s[0] * s[1])),
           'largest_image_wxh': list(max(sizes, key=lambda s: s[0] * s[1])),
           'images_smaller_than_128_on_short_side': sum(v for k, v in sizes.items() if min(k) < 128),
           'split_sizes': {k: len(v) for k, v in split.items()},
           'split_union_is_all_files': sorted(names) == files, 'split_overlap': len(names) - len(set(names)),
           'exact_duplicate_file_groups': len(dups),
           'duplicate_groups_across_splits': sum(1 for v in dups if len({where[n] for n in v}) > 1)}
    with open(os.path.join(OUT, 'tables', 'dataset_stats.json'), 'w') as f:
        json.dump(out, f, indent=2)
    return out


def error_decomposition(sigma=3.0):
    """Split each output's error into a smooth part (Gaussian low-pass) and the remaining detail part."""
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, 'test_inputs', 'hr_128')))
    load = lambda p: rgb(p).astype(np.float64)
    high = lambda img: img - cv2.GaussianBlur(img, (0, 0), sigma)
    out = {}
    for label, run_dir in (('Bicubic', None), ('Run 0', 'run_0_baseline'), ('Run A', 'run_a_original_schedule'),
                           ('Run B', 'run_b_power_schedule')):
        mse, bias, low, hi, energy = [], [], [], [], []
        for n in names:
            hr = load(os.path.join(ROOT, 'test_inputs', 'hr_128', n + '.png'))
            sr = load(os.path.join(ROOT, 'test_inputs', 'bicubic_16_128', n + '.png') if run_dir is None
                      else os.path.join(ROOT, run_dir, 'test_outputs', 'sr', n + '.png'))
            err = sr - hr
            smooth = cv2.GaussianBlur(err, (0, 0), sigma)
            mse.append((err ** 2).mean())
            bias.append((err.reshape(-1, 3).mean(0) ** 2).mean())
            low.append((smooth ** 2).mean())
            hi.append(((err - smooth) ** 2).mean())
            energy.append((high(sr) ** 2).mean() / (high(hr) ** 2).mean())
        out[label] = {'mean_mse': float(np.mean(mse)), 'low_freq_mse': float(np.mean(low)),
                      'high_freq_mse': float(np.mean(hi)),
                      'share_of_mse_from_global_colour_bias': float(np.mean(bias) / np.mean(mse)),
                      'high_freq_energy_ratio_vs_ground_truth': float(np.mean(energy))}
    with open(os.path.join(OUT, 'tables', 'error_decomposition.json'), 'w') as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------- code changes
def code_changes():
    # compared against the working tree, so the listing matches the submitted files
    files = subprocess.check_output(['git', 'diff', '--name-status', BASE_COMMIT, '--', '*.py'],
                                    text=True).split('\n')
    out = []
    for line in files:
        if not line.strip():
            continue
        status, path = line.split('\t')[0], line.split('\t')[-1]
        stat = subprocess.check_output(['git', 'diff', '--numstat', BASE_COMMIT, '--', path], text=True).split()
        out.append({'file': path, 'status': 'new file' if status == 'A' else 'modified',
                    'lines_added': int(stat[0]), 'lines_removed': int(stat[1])})
    return out


def main():
    for sub in ('figures', 'tables'):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    report = {
        'loss': finetuning_logs(),
        'validation_by_epoch': checkpoint_samples(),
        'schedule': schedule_figure(),
        'verification': schedule_verification(),
        'figure_test_images': result_figures(),
        'code_changes': code_changes(),
        'sample_epochs': SAMPLE_EPOCHS, 'trajectory_timesteps': TRAJ_T,
    }
    inference_trajectories()
    environment()
    dataset_stats()
    error_decomposition()
    with open(os.path.join(OUT, 'tables', 'phase2_numbers.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ('loss', 'schedule', 'verification', 'code_changes')}, indent=2))


if __name__ == '__main__':
    main()
