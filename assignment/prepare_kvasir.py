# ADDED: new file (FMGAI assignment) - builds the Kvasir-SEG train/val/test splits for SR3 16 -> 128.
"""Prepare Kvasir-SEG for 16x16 -> 128x128 super-resolution.

Reads the endoscopy images (the segmentation masks are not used), draws a fixed
800/100/100 split with seed 42, and writes each split in the layout the SR3 data
loader expects:

    kvasir_data/prepared/<split>_16_128/
        hr_128/      ground-truth 128x128
        lr_16/       16x16 low-resolution input
        sr_16_128/   lr_16 upsampled back to 128x128 with bicubic (the model's condition)

The resizing follows data/prepare_data.py: shorter side resized with bicubic
interpolation, then a centre crop to a square.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as trans_fn
from tqdm import tqdm

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def find_image_dir(raw_root):
    """Kvasir-SEG unpacks to <something>/images and <something>/masks."""
    candidates = [p for p in raw_root.rglob('images') if p.is_dir()]
    if not candidates:
        raise SystemExit(
            'No "images" folder found under {}. Unzip kvasir-seg.zip there first.'.format(raw_root))
    return max(candidates, key=lambda p: sum(1 for f in p.iterdir() if f.suffix.lower() in IMG_EXT))


def resize_and_crop(img, size):
    img = trans_fn.resize(img, size, InterpolationMode.BICUBIC)
    return trans_fn.center_crop(img, size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='kvasir_data/raw')
    parser.add_argument('--out', default='kvasir_data/prepared')
    parser.add_argument('--splits', default='kvasir_data/splits')
    parser.add_argument('--size', default='16,128')
    parser.add_argument('--counts', default='800,100,100', help='train,val,test')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    lr_size, hr_size = [int(s) for s in args.size.split(',')]
    n_train, n_val, n_test = [int(s) for s in args.counts.split(',')]

    image_dir = find_image_dir(Path(args.raw))
    files = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in IMG_EXT)
    if len(files) != n_train + n_val + n_test:
        raise SystemExit('Expected {} images, found {} in {}'.format(
            n_train + n_val + n_test, len(files), image_dir))

    # Sorted file names + a seeded shuffle make the split independent of the OS listing order.
    order = list(files)
    random.Random(args.seed).shuffle(order)
    splits = {
        'train': sorted(order[:n_train]),
        'val': sorted(order[n_train:n_train + n_val]),
        'test': sorted(order[n_train + n_val:]),
    }

    splits_dir = Path(args.splits)
    splits_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'source_dir': str(image_dir).replace('\\', '/'), 'seed': args.seed,
                'lr_size': lr_size, 'hr_size': hr_size, 'splits': {}}

    for name, members in splits.items():
        root = Path(args.out) / '{}_{}_{}'.format(name, lr_size, hr_size)
        dirs = {k: root / k for k in ('lr_{}'.format(lr_size), 'hr_{}'.format(hr_size),
                                      'sr_{}_{}'.format(lr_size, hr_size))}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        for f in tqdm(members, desc=name):
            img = Image.open(f).convert('RGB')
            lr = resize_and_crop(img, lr_size)
            hr = resize_and_crop(img, hr_size)
            sr = resize_and_crop(lr, hr_size)
            lr.save(dirs['lr_{}'.format(lr_size)] / (f.stem + '.png'))
            hr.save(dirs['hr_{}'.format(hr_size)] / (f.stem + '.png'))
            sr.save(dirs['sr_{}_{}'.format(lr_size, hr_size)] / (f.stem + '.png'))
        names = [f.name for f in members]
        (splits_dir / '{}.txt'.format(name)).write_text('\n'.join(names) + '\n')
        manifest['splits'][name] = {
            'count': len(names),
            'sha256_of_file_list': hashlib.sha256('\n'.join(names).encode()).hexdigest(),
        }

    (splits_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
