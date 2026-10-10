#ADDED: new file (FMGAI assignment) - test-set evaluation shared by Run 0, Run A and Run B.
"""Sample the 100 held-out test images with a given checkpoint and schedule.

For every test image the full reverse process is run (all T steps) starting
from seeded Gaussian noise, so the three runs are compared on identical
starting noise. Saved under --run_dir:

    test_outputs/sr/            one super-resolved PNG per test image
    panels/                     LR | bicubic | SR | HR side by side
    trajectories/               reverse-process snapshots for the first few images
    metrics_per_image.csv       PSNR, SSIM, LPIPS for the model and for bicubic
    metrics_summary.json        mean and standard deviation over the test set
    eval_config.json            exact settings used
    logs/eval.log
"""
import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from collections import OrderedDict

import cv2
import lpips
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.logger as Logger  # noqa: E402
import core.metrics as Metrics  # noqa: E402
import model as Model  # noqa: E402
from data.LRHR_dataset import LRHRDataset  # noqa: E402


def load_config(path):
    json_str = ''
    with open(path, 'r') as f:
        for line in f:
            json_str += line.split('//')[0] + '\n'
    return json.loads(json_str, object_pairs_hook=OrderedDict)


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return None


def to_uint8(t):
    """CHW tensor in [-1, 1] -> HWC uint8 RGB."""
    return Metrics.tensor2img(t)


def upscale_nearest(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def lpips_distance(fn, a_uint8, b_uint8, device):
    def prep(x):
        return (torch.from_numpy(x).permute(2, 0, 1).float() / 127.5 - 1.0).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(fn(prep(a_uint8), prep(b_uint8)).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--run_dir', required=True, help='e.g. assignment_results/run_0_baseline')
    parser.add_argument('--checkpoint', default=None,
                        help='checkpoint prefix (without _gen.pth); defaults to path.resume_state')
    parser.add_argument('--split', default='test')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_traj', type=int, default=8, help='images whose trajectory is saved')
    parser.add_argument('--limit', type=int, default=-1, help='evaluate only the first N images')
    parser.add_argument('--inputs_dir', default='assignment_results/test_inputs')
    parser.add_argument('-enable_wandb', action='store_true')
    args = parser.parse_args()

    opt = load_config(args.config)
    opt['phase'] = 'val'
    opt['distributed'] = False
    opt['gpu_ids'] = [0]
    if args.checkpoint:
        opt['path']['resume_state'] = args.checkpoint
    data_opt = opt['datasets'][args.split]
    opt = Logger.dict_to_nonedict(opt)

    out = {k: os.path.join(args.run_dir, k) for k in
           ('logs', 'test_outputs/sr', 'panels', 'trajectories')}
    for d in out.values():
        os.makedirs(d, exist_ok=True)
    Logger.setup_logger(None, out['logs'], 'eval', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')

    dataset = LRHRDataset(dataroot=data_opt['dataroot'], datatype=data_opt['datatype'],
                          l_resolution=data_opt['l_resolution'], r_resolution=data_opt['r_resolution'],
                          split='val', data_len=args.limit, need_LR=True)
    names = [os.path.splitext(os.path.basename(p))[0] for p in dataset.hr_path[:len(dataset)]]
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info('Evaluating {} images from {}'.format(len(dataset), data_opt['dataroot']))

    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    net = diffusion.netG
    net.eval()
    T = net.num_timesteps
    snap_every = T // 10
    hr_size = data_opt['r_resolution']

    eval_cfg = {
        'config': args.config.replace('\\', '/'), 'checkpoint': opt['path']['resume_state'],
        'split': args.split, 'n_images': len(dataset), 'batch_size': args.batch_size,
        'seed': args.seed, 'sampling_steps': T, 'theta_power': net.theta_power,
        'alpha_bar_1': net.alphas_cumprod[0].item(), 'alpha_bar_T': net.alphas_cumprod[-1].item(),
        'git_commit': git_commit(), 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(0),
    }
    logger.info(json.dumps(eval_cfg, indent=2))

    wandb_run = None
    if args.enable_wandb:
        import wandb
        os.makedirs('experiments', exist_ok=True)
        wandb_run = wandb.init(project=opt['wandb']['project'], group=opt['wandb']['group'],
                               name='{}-test-eval'.format(opt['name']), job_type='test-eval',
                               config=dict(eval_cfg, model=opt['model']), dir='./experiments')

    lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    for sub in ('hr_{}'.format(hr_size), 'lr_{}'.format(data_opt['l_resolution']),
                'bicubic_{}_{}'.format(data_opt['l_resolution'], hr_size)):
        os.makedirs(os.path.join(args.inputs_dir, sub), exist_ok=True)

    rows, traj_done, t_start = [], False, time.time()
    for b_idx, batch in enumerate(loader):
        cond = batch['SR'].to(device)
        # Seeded per batch: Run 0, A and B start every image from the same noise.
        torch.manual_seed(args.seed + b_idx)
        torch.cuda.manual_seed_all(args.seed + b_idx)
        img = torch.randn_like(cond)
        k = min(args.n_traj, cond.shape[0]) if not traj_done else 0
        snaps, snap_t = ([img[:k].cpu()], [T]) if k else ([], [])
        tb = time.time()
        with torch.no_grad():
            for i in reversed(range(T)):
                img = net.p_sample(img, i, condition_x=cond)
                if k and i % snap_every == 0:
                    snaps.append(img[:k].cpu())
                    snap_t.append(i)
        logger.info('batch {} ({} images) sampled in {:.1f}s'.format(b_idx, cond.shape[0], time.time() - tb))
        sr = img.cpu()

        for j in range(cond.shape[0]):
            name = names[int(batch['Index'][j])]
            sr_img, hr_img = to_uint8(sr[j]), to_uint8(batch['HR'][j])
            bic_img, lr_img = to_uint8(batch['SR'][j]), to_uint8(batch['LR'][j])
            Metrics.save_img(sr_img, os.path.join(out['test_outputs/sr'], name + '.png'))
            Metrics.save_img(hr_img, os.path.join(args.inputs_dir, 'hr_{}'.format(hr_size), name + '.png'))
            Metrics.save_img(lr_img, os.path.join(
                args.inputs_dir, 'lr_{}'.format(data_opt['l_resolution']), name + '.png'))
            Metrics.save_img(bic_img, os.path.join(
                args.inputs_dir, 'bicubic_{}_{}'.format(data_opt['l_resolution'], hr_size), name + '.png'))
            Metrics.save_img(np.concatenate(
                [upscale_nearest(lr_img, hr_size), bic_img, sr_img, hr_img], axis=1),
                os.path.join(out['panels'], name + '.png'))
            rows.append(OrderedDict([
                ('image', name),
                ('psnr', Metrics.calculate_psnr(sr_img, hr_img)),
                ('ssim', Metrics.calculate_ssim(sr_img, hr_img)),
                ('lpips', lpips_distance(lpips_fn, sr_img, hr_img, device)),
                ('bicubic_psnr', Metrics.calculate_psnr(bic_img, hr_img)),
                ('bicubic_ssim', Metrics.calculate_ssim(bic_img, hr_img)),
                ('bicubic_lpips', lpips_distance(lpips_fn, bic_img, hr_img, device)),
            ]))

        if k:
            # one strip per image: bicubic input | x_T ... x_0 | ground truth
            strips = []
            for j in range(k):
                name = names[int(batch['Index'][j])]
                strip = np.concatenate(
                    [to_uint8(batch['SR'][j])] + [to_uint8(s[j]) for s in snaps] + [to_uint8(batch['HR'][j])],
                    axis=1)
                Metrics.save_img(strip, os.path.join(out['trajectories'], name + '.png'))
                strips.append(strip)
            Metrics.save_img(np.concatenate(strips, axis=0),
                             os.path.join(out['trajectories'], '_trajectory_grid.png'))
            with open(os.path.join(out['trajectories'], 'columns.json'), 'w') as f:
                json.dump({'columns': ['bicubic_input'] + ['x_t at t={}'.format(t) for t in snap_t]
                           + ['ground_truth'], 'images': [names[int(i)] for i in batch['Index'][:k]]},
                          f, indent=2)
            traj_done = True

    with open(os.path.join(args.run_dir, 'metrics_per_image.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    keys = [k for k in rows[0] if k != 'image']
    summary = OrderedDict(n_images=len(rows), sampling_seconds=round(time.time() - t_start, 1))
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        summary[key + '_mean'] = float(vals.mean())
        summary[key + '_std'] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    with open(os.path.join(args.run_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.run_dir, 'eval_config.json'), 'w') as f:
        json.dump(eval_cfg, f, indent=2)
    logger.info('# Test # ' + json.dumps(summary, indent=2))

    if wandb_run is not None:
        import wandb
        wandb_run.summary.update({'test/' + k: v for k, v in summary.items()})
        table = wandb.Table(columns=['image', 'panel (LR | bicubic | SR | HR)', 'psnr', 'ssim', 'lpips'])
        for r in rows[:16]:
            table.add_data(r['image'], wandb.Image(os.path.join(out['panels'], r['image'] + '.png')),
                           r['psnr'], r['ssim'], r['lpips'])
        wandb_run.log({'test/examples': table,
                       'test/trajectory_grid': wandb.Image(
                           os.path.join(out['trajectories'], '_trajectory_grid.png'))})
        wandb_run.finish()


if __name__ == '__main__':
    main()
