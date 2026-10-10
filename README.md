# Image Super-Resolution via Iterative Refinement

<!-- #ADDED: FMGAI assignment section (everything up to the "Original README" marker) -->

## FMGAI assignment: SR3 on Kvasir-SEG under a power-transformed noise schedule

This fork is used for the *Foundation Models and Generative AI* assignment. A pretrained SR3 model
(16×16 → 128×128, trained on faces) is applied to and fine-tuned on Kvasir-SEG endoscopy images under two
noise schedules.

| Run | Branch | Weights | Schedule θ<sub>t</sub> | Config |
| --- | --- | --- | --- | --- |
| Run 0 | `feature/run-0-baseline` | pretrained, no fine-tuning | ᾱ<sub>t</sub> (native) | `config/kvasir_run0_baseline.json` |
| Run A | `feature/run-a-original-schedule` | fine-tuned 20,000 iterations | ᾱ<sub>t</sub> (native) | `config/kvasir_runA_original.json` |
| Run B | `feature/run-b-power-schedule` | fine-tuned 20,000 iterations | ᾱ<sub>t</sub><sup>2.50</sup> | `config/kvasir_runB_power.json` |

The shared code lives on `feature/phase1-setup`. The run branches are stacked on top of it in the order
Run 0 → Run A → Run B, so `feature/run-b-power-schedule` contains everything.

The Phase 1 declaration is at `assignment_results/declaration/Phase1_Declaration.pdf`.

### Where things are

| Path | Contents |
| --- | --- |
| `kvasir_data/` | Dataset: raw download, split lists, prepared images (see `kvasir_data/README.md`) |
| `pretrained/sr3_16_128/` | Pretrained checkpoint (not in git) |
| `assignment/` | Scripts added for the assignment |
| `assignment_results/declaration/` | Phase 1 declaration PDF |
| `assignment_results/schedule_analysis/` | Schedule values, bound checks and figure |
| `assignment_results/test_inputs/` | The 100 test images: ground truth, 16×16 input, bicubic upsampling |
| `assignment_results/run_0_baseline/` | Run 0 outputs, metrics, trajectories, log |
| `assignment_results/run_a_original_schedule/` | Run A fine-tuning logs and curves, outputs, metrics, trajectories |
| `assignment_results/run_b_power_schedule/` | Run B fine-tuning logs and curves, outputs, metrics, trajectories |
| `assignment_results/comparison/` | Three-run summary table, paired differences, example grid |
| `experiments/` | Raw training folders with checkpoints (not in git) |

Inside each run folder: `test_outputs/sr/` (one PNG per test image), `panels/` (LR, bicubic, SR and ground
truth side by side), `trajectories/` (reverse-process snapshots), `metrics_per_image.csv`,
`metrics_summary.json`, `eval_config.json` and `logs/`. The fine-tuned runs also have `train_loss.csv`,
`validation_metrics.csv`, `training_curves.png`, `validation_progress.png` and `tb_logger/`.

### Code changes

Every changed line in the original files carries a `#MODIFIED` comment and every new line an `#ADDED`
comment. New files say `#ADDED: new file` on their first line.

| File | Change |
| --- | --- |
| `model/sr3_modules/diffusion.py` | `theta_power` option: raises ᾱ<sub>t</sub> to the power x and re-derives β<sub>t</sub> and the posterior terms; batched sampling returns the whole batch |
| `model/model.py` | Rebuilds the schedule from the config after loading a checkpoint (the checkpoint stores schedule buffers); `reset_optimizer` option to fine-tune from weights only; logs the active schedule |
| `sr.py` | Random seed; validation on batches with a fixed sampling seed; SSIM during validation; iteration logged to W&B |
| `core/logger.py` | Validation set size during training is configurable (`train.val_data_len`) |
| `core/wandb_logger.py` | Run name, group and job type passed to W&B |
| `data/__init__.py` | Validation batch size is configurable |
| `assignment/*.py` | New: data preparation, schedule analysis, test evaluation, log collection, declaration PDF |

### Step 0: environment

Tested on Windows 11, Python 3.12, PyTorch 2.8 (CUDA), one RTX 5070 Ti Laptop GPU (12 GB). Commands are run
from the repository root.

```bash
python -m venv .venv --system-site-packages
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirement.txt
```

```bash
wandb login
```

W&B logging is optional. Leave out `-enable_wandb` in the commands below to run without it.

### Step 1: pretrained checkpoint

Downloads `I640000_E37_gen.pth` and `I640000_E37_opt.pth` (the "16×16 -> 128×128 on FFHQ-CelebaHQ" row of the
pretrained-model table further down).

```bash
python -m gdown --folder https://drive.google.com/drive/folders/12jh0K8XoM1FqpeByXvugHHAF3oAZ8KRu -O pretrained/sr3_16_128
```

### Step 2: dataset

Download `kvasir-seg.zip` from <https://datasets.simula.no/kvasir-seg/> and unzip it into `kvasir_data/raw/`.
Then build the fixed 800 / 100 / 100 split (seed 42) at 16 → 128:

```bash
python assignment/prepare_kvasir.py
```

### Step 3: schedule analysis and declaration

```bash
python assignment/schedule_analysis.py --x 2.50
```

```bash
python assignment/build_declaration.py
```

### Step 4: Run 0, pretrained model without fine-tuning

```bash
git checkout feature/run-0-baseline
```

```bash
python assignment/evaluate.py -c config/kvasir_run0_baseline.json --run_dir assignment_results/run_0_baseline -enable_wandb
```

### Step 5: Run A, fine-tuning with the native schedule

```bash
git checkout feature/run-a-original-schedule
```

```bash
python sr.py -p train -c config/kvasir_runA_original.json -enable_wandb
```

```bash
python assignment/collect_training_artifacts.py --name kvasir_runA_original --run_dir assignment_results/run_a_original_schedule
```

Evaluate the final checkpoint. Replace `<timestamp>` with the folder name printed by the previous command:

```bash
python assignment/evaluate.py -c config/kvasir_runA_original.json --checkpoint experiments/kvasir_runA_original_<timestamp>/checkpoint/I20000_E100 --run_dir assignment_results/run_a_original_schedule -enable_wandb
```

### Step 6: Run B, fine-tuning with the power-transformed schedule

```bash
git checkout feature/run-b-power-schedule
```

```bash
python sr.py -p train -c config/kvasir_runB_power.json -enable_wandb
```

```bash
python assignment/collect_training_artifacts.py --name kvasir_runB_power --run_dir assignment_results/run_b_power_schedule
```

```bash
python assignment/evaluate.py -c config/kvasir_runB_power.json --checkpoint experiments/kvasir_runB_power_<timestamp>/checkpoint/I20000_E100 --run_dir assignment_results/run_b_power_schedule -enable_wandb
```

The two fine-tuning configs differ only in the experiment name and in `theta_power`. To check:

```bash
git diff --no-index config/kvasir_runA_original.json config/kvasir_runB_power.json
```

### Step 7: compare the three runs

Writes the summary table, paired per-image differences and an example grid to `assignment_results/comparison/`.

```bash
python assignment/compare_runs.py
```

### Results

100 Kvasir-SEG test images, 2,000 sampling steps, same starting noise in every run. Mean ± standard
deviation over the test set. Higher is better for PSNR and SSIM, lower is better for LPIPS.

| Method | PSNR (dB) | SSIM | LPIPS | W&B |
| --- | --- | --- | --- | --- |
| Bicubic upsampling (reference) | 23.21 ± 1.50 | 0.636 ± 0.086 | 0.560 ± 0.065 | |
| Run 0: pretrained, no fine-tuning | 22.49 ± 1.89 | 0.572 ± 0.092 | 0.273 ± 0.061 | [run](https://wandb.ai/rikin-patel-indian-institute-of-technology-jodhpur/FMGAI-Assn/runs/zuwie16r) |
| Run A: fine-tuned, native schedule ᾱ<sub>t</sub> | 24.15 ± 2.33 | 0.614 ± 0.098 | 0.170 ± 0.053 | [train](https://wandb.ai/rikin-patel-indian-institute-of-technology-jodhpur/FMGAI-Assn/runs/pa0ph10p), [test](https://wandb.ai/rikin-patel-indian-institute-of-technology-jodhpur/FMGAI-Assn/runs/p30ues96) |

Run A was fine-tuned in `experiments/kvasir_runA_original_261010_160318` and evaluated from its
`I20000_E100` checkpoint. Validation PSNR on 20 validation images moved between 22.8 and 24.1 dB over the ten
validation passes and ended at 23.82 dB (see `assignment_results/run_a_original_schedule/validation_metrics.csv`).

| Run B: fine-tuned, power schedule ᾱ<sub>t</sub><sup>2.50</sup> | 23.43 ± 2.17 | 0.598 ± 0.096 | 0.171 ± 0.046 | [train](https://wandb.ai/rikin-patel-indian-institute-of-technology-jodhpur/FMGAI-Assn/runs/s9wk031l), [test](https://wandb.ai/rikin-patel-indian-institute-of-technology-jodhpur/FMGAI-Assn/runs/hb8yzh6v) |

Run B was fine-tuned in `experiments/kvasir_runB_power_261010_194732` and evaluated from its `I20000_E100`
checkpoint, sampling with the same ᾱ<sub>t</sub><sup>2.50</sup> schedule it was trained on. Validation PSNR
moved between 22.7 and 24.2 dB and ended at 22.80 dB.

Paired per-image differences, Run B minus Run A, on the 100 test images
(`assignment_results/comparison/paired_differences.json`):

| Metric | Mean difference | Standard error | Images where Run B is better |
| --- | --- | --- | --- |
| PSNR (dB) | −0.72 | 0.15 | 15 / 100 |
| SSIM | −0.016 | 0.004 | 25 / 100 |
| LPIPS | +0.001 | 0.003 | 50 / 100 |

Both fine-tuned runs improve clearly on Run 0. Between them, the native schedule gives higher PSNR and SSIM,
while LPIPS is the same within noise. Each run is a single training seed and a single sample per image.

---

<!-- Original README of the SR3 implementation follows. -->

[Paper](https://arxiv.org/pdf/2104.07636.pdf ) |  [Project](https://iterative-refinement.github.io/ )

## Brief

This is an unofficial implementation of **Image Super-Resolution via Iterative Refinement(SR3)** by **PyTorch**.

There are some implementation details that may vary from the paper's description, which may be different from the actual `SR3` structure due to details missing. Specifically, we:

- Used the ResNet block and channel concatenation style like vanilla `DDPM`.
- Used the attention mechanism in low-resolution features ( $16 \times 16$ ) like vanilla `DDPM`.
- Encode the $\gamma$ as `FilM` structure did in `WaveGrad`, and embed it without affine transformation.
- Define the posterior variance as $\dfrac{1-\gamma_{t-1}}{1-\gamma_{t}} \beta_t$  rather than $\beta_t$,  which gives similar results to the vanilla paper.

**If you just want to upscale $(64 \times 64)\text{px} \rightarrow (512 \times 512)\text{px}$ images using the pre-trained model, check out [this google colab script](https://colab.research.google.com/drive/1G1txPI1GKueKH0cSi_DgQFKwfyJOXlhY?usp=sharing).**

## Status

**★★★ NEW: The follow-up [Palette-Image-to-Image-Diffusion-Models](https://arxiv.org/abs/2111.05826) is now available; See the details [here](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models) ★★★**

### Conditional Generation (with Super Resolution)

- [x] 16×16 -> 128×128 on FFHQ-CelebaHQ
- [x] 64×64 -> 512×512 on FFHQ-CelebaHQ

### Unconditional Generation

- [x] 128×128 face generation on FFHQ
- [ ] ~~1024×1024 face generation by a cascade of 3 models~~

### Training Step

- [x] log / logger
- [x] metrics evaluation
- [x] multi-gpu support
- [x] resume training / pretrained model
- [x] validate alone script
- [x] [Weights and Biases Logging](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/pull/44) 🌟 NEW



## Results

*Note:*  We set the maximum reverse steps budget to $2000$. We limited the model parameters in `Nvidia 1080Ti`, **image noise** and **hue deviation** occasionally appear in high-resolution images, resulting in low scores.  There is a lot of room for optimization.  **We are welcome to any contributions for more extensive experiments and code enhancements.**

| Tasks/Metrics        | SSIM(+) | PSNR(+) | FID(-)  | IS(+)   |
| -------------------- | ----------- | -------- | ---- | ---- |
| 16×16 -> 128×128 | 0.675       | 23.26    | - | - |
| 64×64 -> 512×512     | 0.445 | 19.87 | - | - |
| 128×128 | - | - | | |
| 1024×1024 | - | - |      |      |

- #### 16×16 -> 128×128 on FFHQ-CelebaHQ [[More Results](https://drive.google.com/drive/folders/1Vk1lpHzbDf03nME5fV9a-lWzSh3kMK14?usp=sharing)]

| <img src="./misc/sr_process_16_128_0.png" alt="show" style="zoom:90%;" /> |  <img src="./misc/sr_process_16_128_1.png" alt="show" style="zoom:90%;" />    |   <img src="./misc/sr_process_16_128_2.png" alt="show" style="zoom:90%;" />   |
| ------------------------------------------------------------ | ---- | ---- |

- #### 64×64 -> 512×512 on FFHQ-CelebaHQ [[More Results](https://drive.google.com/drive/folders/1yp_4xChPSZUeVIgxbZM-e3ZSsSgnaR9Z?usp=sharing)]

| <img src="./misc/sr_64_512_0_inf.png" alt="show" style="zoom:90%;" /> | <img src="./misc/sr_64_512_0_sr.png" alt="show" style="zoom:90%;" /> | <img src="./misc/sr_64_512_0_hr.png" alt="show" style="zoom:90%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="./misc/sr_64_512_1_sr.png" alt="show" style="zoom:90%;" /> | <img src="./misc/sr_64_512_2_sr.png" alt="show" style="zoom:90%;" /> | <img src="./misc/sr_64_512_3_sr.png" alt="show" style="zoom:90%;" /> |

- #### 128×128 face generation on FFHQ [[More Results](https://drive.google.com/drive/folders/13AsjRwDw4wMmL0bK7wPd2rP7ds7eyAMh?usp=sharing)]

| <img src="./misc/sample_process_128_0.png" alt="show" style="zoom:90%;" /> |  <img src="./misc/sample_process_128_1.png" alt="show" style="zoom:90%;" />    |   <img src="./misc/sample_process_128_2.png" alt="show" style="zoom:90%;" />   |
| ------------------------------------------------------------ | ---- | ---- |



## Usage
### Environment
```python
pip install -r requirement.txt
```

### Pretrained Model

This paper is based on "Denoising Diffusion Probabilistic Models", and we build both DDPM/SR3 network structures, which use timesteps/gamma as model embedding inputs, respectively. In our experiments, the SR3 model can achieve better visual results with the same reverse steps and learning rate. You can select the JSON files with annotated suffix names to train the different models.

| Tasks                             | Platform（Code：qwer)                                        | 
| --------------------------------- | ------------------------------------------------------------ |
| 16×16 -> 128×128 on FFHQ-CelebaHQ | [Google Drive](https://drive.google.com/drive/folders/12jh0K8XoM1FqpeByXvugHHAF3oAZ8KRu?usp=sharing)\|[Baidu Yun](https://pan.baidu.com/s/1OzsGZA2Vmq1ZL_VydTbVTQ) |  
| 64×64 -> 512×512 on FFHQ-CelebaHQ | [Google Drive](https://drive.google.com/drive/folders/1mCiWhFqHyjt5zE4IdA41fjFwCYdqDzSF?usp=sharing)\|[Baidu Yun](https://pan.baidu.com/s/1orzFmVDxMmlXQa2Ty9zY3g) |   
| 128×128 face generation on FFHQ   | [Google Drive](https://drive.google.com/drive/folders/1ldukMgLKAxE7qiKdFJlu-qubGlnW-982?usp=sharing)\|[Baidu Yun](https://pan.baidu.com/s/1Vsd08P1A-48OGmnRV0E7Fg ) | 

```python
# Download the pretrained model and edit [sr|sample]_[ddpm|sr3]_[resolution option].json about "resume_state":
"resume_state": [your pretrained model's path]
```

### Data Prepare

#### New Start

If you didn't have the data, you can prepare it by following steps:

- [FFHQ 128×128](https://github.com/NVlabs/ffhq-dataset) | [FFHQ 512×512](https://www.kaggle.com/arnaud58/flickrfaceshq-dataset-ffhq)
- [CelebaHQ 256×256](https://www.kaggle.com/badasstechie/celebahq-resized-256x256) | [CelebaMask-HQ 1024×1024](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view)

Download the dataset and prepare it in **LMDB** or **PNG** format using script.

```python
# Resize to get 16×16 LR_IMGS and 128×128 HR_IMGS, then prepare 128×128 Fake SR_IMGS by bicubic interpolation
python data/prepare_data.py  --path [dataset root]  --out [output root] --size 16,128 -l
```

then you need to change the datasets config to your data path and image resolution: 

```json
"datasets": {
    "train": {
        "dataroot": "dataset/ffhq_16_128", // [output root] in prepare.py script
        "l_resolution": 16, // low resolution need to super_resolution
        "r_resolution": 128, // high resolution
        "datatype": "lmdb", //lmdb or img, path of img files
    },
    "val": {
        "dataroot": "dataset/celebahq_16_128", // [output root] in prepare.py script
    }
},
```

#### Own Data

You also can use your image data by following steps, and we have some examples in dataset folder.

At first, you should organize the images layout like this, this step can be finished by `data/prepare_data.py` automatically:

```shell
# set the high/low resolution images, bicubic interpolation images path 
dataset/celebahq_16_128/
├── hr_128 # it's same with sr_16_128 directory if you don't have ground-truth images.
├── lr_16 # vinilla low resolution images
└── sr_16_128 # images ready to super resolution
```

```python
# super resolution from 16 to 128
python data/prepare_data.py  --path [dataset root]  --out celebahq --size 16,128 -l
```

*Note: Above script can be used whether you have the vanilla high-resolution images or not.*

then you need to change the dataset config to your data path and image resolution: 

```json
"datasets": {
    "train|val": { // train and validation part
        "dataroot": "dataset/celebahq_16_128",
        "l_resolution": 16, // low resolution need to super_resolution
        "r_resolution": 128, // high resolution
        "datatype": "img", //lmdb or img, path of img files
    }
},
```

### Training/Resume Training

```python
# Use sr.py and sample.py to train the super resolution task and unconditional generation task, respectively.
# Edit json files to adjust network structure and hyperparameters
python sr.py -p train -c config/sr_sr3.json
```

### Test/Evaluation

```python
# Edit json to add pretrain model path and run the evaluation 
python sr.py -p val -c config/sr_sr3.json

# Quantitative evaluation alone using SSIM/PSNR metrics on given result root
python eval.py -p [result root]
```

### Inference Alone

Set the  image path like steps in `Own Data`, then run the script:

```python
# run the script
python infer.py -c [config file]
```

## Weights and Biases 🎉

The library now supports experiment tracking, model checkpointing and model prediction visualization with [Weights and Biases](https://wandb.ai/site). You will need to [install W&B](https://pypi.org/project/wandb/) and login by using your [access token](https://wandb.ai/authorize). 

```
pip install wandb

# get your access token from wandb.ai/authorize
wandb login
```

W&B logging functionality is added to the `sr.py`, `sample.py` and `infer.py` files. You can pass `-enable_wandb` to start logging.

- `-log_wandb_ckpt`: Pass this argument along with `-enable_wandb` to save model checkpoints as [W&B Artifacts](https://docs.wandb.ai/guides/artifacts). Both `sr.py` and `sample.py` is enabled with model checkpointing. 
- `-log_eval`: Pass this argument along with `-enable_wandb` to save the evaluation result as interactive [W&B Tables](https://docs.wandb.ai/guides/data-vis). Note that only `sr.py` is enabled with this feature. If you run `sample.py` in eval mode, the generated images will automatically be logged as image media panel. 
- `-log_infer`: While running `infer.py` pass this argument along with `-enable_wandb` to log the inference results as interactive W&B Tables. 

You can find more on using these features [here](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/pull/44). 🚀


## Acknowledgements

Our work is based on the following theoretical works:

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Image Super-Resolution via Iterative Refinement](https://arxiv.org/pdf/2104.07636.pdf)
- [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/abs/2009.00713)
- [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)

Furthermore, we are benefitting a lot from the following projects:

- https://github.com/bhushan23/BIG-GAN
- https://github.com/lmnt-com/wavegrad
- https://github.com/rosinality/denoising-diffusion-pytorch
- https://github.com/lucidrains/denoising-diffusion-pytorch
- https://github.com/hejingwenhejingwen/AdaFM
