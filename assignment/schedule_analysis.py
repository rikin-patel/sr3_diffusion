#ADDED: new file (FMGAI assignment) - numerical check and plots of the native vs power-transformed schedule.
"""Schedule analysis for theta_t = alpha_bar_t (gamma_t in the SR3 paper).

Builds the schedule through the same code path the model uses, checks that
theta_t and theta_t ** x stay strictly inside (0, 1), compares the result with
the buffers stored in the pretrained checkpoint, and writes a CSV, a JSON
summary and a figure to assignment_results/schedule_analysis/.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.sr3_modules.diffusion import GaussianDiffusion  # noqa: E402


def build(theta_power, n_timestep, linear_start, linear_end):
    g = GaussianDiffusion(denoise_fn=None, image_size=128)
    g.set_new_noise_schedule({'schedule': 'linear', 'n_timestep': n_timestep,
                              'linear_start': linear_start, 'linear_end': linear_end,
                              'theta_power': theta_power}, device='cpu')
    return g


def describe(theta):
    return {
        'min': float(theta.min()), 'argmin_t': int(theta.argmin()) + 1,
        'max': float(theta.max()), 'argmax_t': int(theta.argmax()) + 1,
        'strictly_in_open_unit_interval': bool(((theta > 0) & (theta < 1)).all()),
        'strictly_decreasing': bool((np.diff(theta) < 0).all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, default=2.50)
    parser.add_argument('--n_timestep', type=int, default=2000)
    parser.add_argument('--linear_start', type=float, default=1e-6)
    parser.add_argument('--linear_end', type=float, default=1e-2)
    parser.add_argument('--checkpoint', default='pretrained/sr3_16_128/I640000_E37_gen.pth')
    parser.add_argument('--out', default='assignment_results/schedule_analysis')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # float64 reference, computed independently of the model code
    betas = np.linspace(args.linear_start, args.linear_end, args.n_timestep, dtype=np.float64)
    theta = np.cumprod(1. - betas)
    theta_x = theta ** args.x
    beta_x = 1. - theta_x / np.append(1., theta_x[:-1])
    t = np.arange(1, args.n_timestep + 1)

    native, power = build(1.0, args.n_timestep, args.linear_start, args.linear_end), \
        build(args.x, args.n_timestep, args.linear_start, args.linear_end)
    summary = {
        'theta_definition': 'theta_t = alpha_bar_t = prod_{s<=t} (1 - beta_s)',
        'x': args.x, 'T': args.n_timestep,
        'beta_schedule': {'type': 'linear', 'start': args.linear_start, 'end': args.linear_end},
        'theta_native_float64': describe(theta),
        'theta_power_float64': describe(theta_x),
        'theta_power_float32_min': float(theta_x.astype(np.float32).min()),
        'implied_beta_native': {'min': float(betas.min()), 'max': float(betas.max())},
        'implied_beta_power': {'min': float(beta_x.min()), 'max': float(beta_x.max())},
        'model_code_matches_reference': {
            'native': bool(np.allclose(native.alphas_cumprod.numpy(), theta, rtol=1e-5, atol=0)),
            'power': bool(np.allclose(power.alphas_cumprod.numpy(), theta_x, rtol=1e-5, atol=0)),
        },
    }
    if os.path.isfile(args.checkpoint):
        sd = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        summary['pretrained_checkpoint'] = {
            'path': args.checkpoint.replace('\\', '/'),
            'alphas_cumprod_equals_native_schedule': bool(
                np.allclose(sd['alphas_cumprod'].numpy(), theta, rtol=1e-5, atol=0)),
            'betas_equal_linear_schedule': bool(
                np.allclose(sd['betas'].numpy(), betas, rtol=1e-5, atol=0)),
        }

    rows = np.stack([t, betas, theta, beta_x, theta_x,
                     theta / (1 - theta), theta_x / (1 - theta_x)], axis=1)
    np.savetxt(os.path.join(args.out, 'schedule_values.csv'), rows, delimiter=',',
               header='t,beta_native,theta_native,beta_power,theta_power,snr_native,snr_power',
               comments='', fmt=['%d'] + ['%.12e'] * 6)
    with open(os.path.join(args.out, 'schedule_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    label_x = r'$\bar\alpha_t^{\,%.2f}$ (Run B)' % args.x
    axes[0].plot(t, theta, label=r'$\bar\alpha_t$ (Run 0 / Run A)', color='#1f4e79')
    axes[0].plot(t, theta_x, label=label_x, color='#c0392b')
    axes[0].set_title(r'Noise parameter $\theta_t=\bar\alpha_t$')
    axes[0].set_ylabel(r'$\theta_t$')
    axes[1].semilogy(t, theta, color='#1f4e79')
    axes[1].semilogy(t, theta_x, color='#c0392b')
    axes[1].set_title(r'$\theta_t$ on a log scale')
    axes[2].semilogy(t, theta / (1 - theta), color='#1f4e79')
    axes[2].semilogy(t, theta_x / (1 - theta_x), color='#c0392b')
    axes[2].set_title(r'Signal-to-noise ratio $\theta_t/(1-\theta_t)$')
    for ax in axes:
        ax.set_xlabel('timestep t')
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'schedule_comparison.png'), dpi=200)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
