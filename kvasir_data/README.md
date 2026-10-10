# Kvasir-SEG data folder

Everything related to the dataset lives here.

| Folder | Contents | In git? |
| --- | --- | --- |
| `raw/` | The downloaded `kvasir-seg.zip` and its extracted `images/` and `masks/` folders | No |
| `splits/` | `train.txt`, `val.txt`, `test.txt` and `manifest.json` (the fixed 800 / 100 / 100 split, seed 42) | Yes |
| `prepared/` | The three splits resized for SR3 (`lr_16`, `hr_128`, `sr_16_128`) | No, rebuilt by script |

## Getting the data

1. Download `kvasir-seg.zip` (about 46 MB) from <https://datasets.simula.no/kvasir-seg/>.
2. Unzip it into `kvasir_data/raw/`, so that the images end up in `kvasir_data/raw/Kvasir-SEG/images/`.
   Any folder depth is fine; the script looks for the `images` folder.
3. From the repository root run:

```bash
python assignment/prepare_kvasir.py
```

Only the 1,000 endoscopy images are used. The segmentation masks are ignored.
