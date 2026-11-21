# PredictiveCodingAffordance

## Environment

macOS (development):

```
conda create -n pcaff --override-channels -c conda-forge python=3.11
conda activate pcaff
pip install -r requirements.txt
pip install -e external/predify --no-deps
python -m pytest tests -v -s           # loader tests (cpu, mps / cuda if available)
python scripts/benchmark_pvgg16.py --devices cpu mps   # throughput + memory
```

Windows (RTX 3060 6 GB; all real experiments), PowerShell:

```
conda create -n pcaff --override-channels -c conda-forge python=3.11
conda activate pcaff
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install numpy==2.4.6 pillow==13.0.0 toml==0.10.2 psutil==7.2.2 pytest==9.2.0
pip install huggingface_hub==2.4.0 pyarrow==26.0.1
pip install -e external/predify --no-deps
python -m pytest tests -v -s
python scripts/benchmark_pvgg16.py --devices cuda --batch-sizes 1 8 16 24 32
```

- Do not run `pip install -r requirements.txt` on Windows: its `torch==2.15.0`
  pin would replace the CUDA build with the CPU-only wheel from PyPI.
- Driver 592.00 supports CUDA <= 13.1 (`nvidia-smi`). On 2026-11-20 torch 2.15.0
  CUDA wheels existed only for cu132, so Windows uses torch 2.14.1+cu130 /
  torchvision 0.29.1+cu130 (one release behind the Mac). t = 0 still equals
  plain VGG16 exactly (max|diff| = 0) on cpu and cuda.
- Use `curl.exe`, not `curl` (a PowerShell alias for Invoke-WebRequest), and
  `New-Item -ItemType Directory -Force <dir>` instead of `mkdir -p`.
- Benchmark (T = 15, PCoderN): batch 16 is fastest, 0.40 s/image; batch 32
  needs 6.1 GB and spills into shared memory
  (results/benchmark_Windows-AMD64_20261120-221214).

### PCoderN C_sqrt (configs/pcoder_c_sqrt.json)

PCoderN's sqrt(C) is calibrated once on CPU and loaded as fixed values on every
device (upstream's runtime calibration is inflated by cuDNN noise on CUDA).
The file in the repo holds the official values (2026-11-20), calibrated on the
16 ImageNet val images in `configs/imagenet_val_c_sqrt_subset.json`:

| PCoder | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| C_sqrt | 8.660254 (sqrt 75) | 66.949844 | 125.134354 | 168.783661 | 221.532562 |

Do not regenerate it. To reproduce it (the test `test_c_sqrt_file_reproducible`
does this automatically when the images are present):

```
python scripts/calibrate_c_sqrt.py --input imagenet --force --out <other path>
```

`--input synthetic` (seeded randn) only exists for machines without ImageNet;
its output is marked temporary and the loader warns about it.

### ImageNet validation (data/imagenet_val/, git-ignored)

Only the validation split of `ILSVRC/imagenet-1k` on Hugging Face (accept the
terms on the dataset page, then `hf auth login`). 14 parquet shards, 50,000
images, 6,693,093,726 bytes (~6.7 GB; check free disk space first):

```
hf download ILSVRC/imagenet-1k --repo-type dataset --revision 49e2ee26f3810fb5a7536bbf732a7b07389a47b5 --include "data/validation-*" --local-dir data/imagenet_val
```

Shards land in `data/imagenet_val/data/validation-000NN-of-00014.parquet`.
Columns: `image` {bytes, path}, `label` (int, torchvision class order). Rows
are shuffled; `image.path` keeps the original filename, e.g.
`ILSVRC2012_val_00027555_n01824575.JPEG`.

Fixed subsets (committed in `configs/`) and their JPEGs (extracted to
`data/imagenet_val/images/`, original bytes, 274 MB):

```
python scripts/prepare_imagenet_val.py           # writes both subset files + extracts JPEGs
python scripts/check_vgg16_labels.py --device cuda
```

- `imagenet_val_c_sqrt_subset.json`: 16 images, val_id 1..16 (first by filename).
- `imagenet_val_gate1_subset.json`: 2 per class (2,000), `numpy.random.default_rng(0)`,
  disjoint from the calibration subset.
- Label check: plain VGG16 + Predify preprocessing on every 10th Gate 1 image
  (200): top-1 69.5 %, top-5 93.0 % (Table S4 t = 0: 71.63 %), so HF labels use
  torchvision's class order (results/vgg16_label_check_20261120-233306).

On another machine, rerunning `prepare_imagenet_val.py --force` must produce the
same subset files; only the `created` field may differ.

### Sanity images (data/sanity_images/, git-ignored)

Five public JPEGs used only to check that t = 0 predictions match plain VGG16:

```
mkdir -p data/sanity_images && cd data/sanity_images
curl -L -o dog.jpg    https://github.com/pytorch/hub/raw/master/images/dog.jpg
curl -L -o cat.jpg    https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg
curl -L -o bridge.jpg https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg
curl -L -o banana.jpg https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg
curl -L -o pizza.jpg  https://upload.wikimedia.org/wikipedia/commons/a/a3/Eq_it-na_pizza-margherita_sep2005_sml.jpg
```

Wikimedia answers rapid requests with HTTP 429 and curl then saves the error
page as the .jpg: add `--fail` and space the downloads out. The tests use
whatever JPEGs are present (the Windows machine has 4; banana.jpg got 429).

## Gate 1 positive control (ImageNet val, Gaussian noise)

Passed on 2026-11-21. Frozen for the affordance experiments: fixed HP set + PCoderN.

```
python scripts/run_gate1_grid.py --device cuda      # 4 configs, ~62 min each on the RTX 3060
python scripts/analyze_pc_terms.py --device cuda    # alpha term vs other update terms, 16 images
python scripts/summarize_gate1.py --runs <fixed_PCoderN> <tuned_PCoderN> <fixed_PCoder> <tuned_PCoder> --pc-terms <pc_terms folder>
```

- One run: `python scripts/run_gate1.py --hp-set fixed --pcoder-type PCoderN --device cuda --drift`
  (add `--n-images 64` for a smoke run). The first `--runs` folder is the one the pass rule is applied to.
- 2,000 Gate 1 images, batch 16, t = 0..15, sigma in {0, 0.5, 0.75, 1.0}; noise in normalized-image
  units, no clamping, one generator per image seeded with `val_id` (same noise for every sigma and config).
- Per-image predictions (`predictions_sigma*.npz`) are git-ignored and stay on the Windows machine;
  `summarize_gate1.py` recomputes every metric from them and checks it against each run's `metrics.json`.

Top-1 (%), t = 0 -> t = 15 (gain, paired bootstrap 95 % CI), 2,000 images:

| sigma | Table S4 | fixed + PCoderN | tuned + PCoderN | fixed + PCoder | tuned + PCoder |
|---|---|---|---|---|---|
| 0 | 71.63 -> 71.47 | 70.90 -> 71.05 (+0.15 [-0.45, +0.75]) | 70.90 -> 70.55 (-0.35) | 70.90 -> 70.95 (+0.05) | 70.90 -> 70.55 (-0.35) |
| 0.5 | 35.61 -> 38.59 | 34.75 -> 37.15 (+2.40 [+1.40, +3.40]) | 34.75 -> 38.65 (+3.90) | 34.75 -> 37.10 (+2.35) | 34.75 -> 38.70 (+3.95) |
| 0.75 | 16.69 -> 18.46 | 15.10 -> 17.60 (+2.50 [+1.65, +3.40]) | 15.10 -> 18.70 (+3.60) | 15.10 -> 17.60 (+2.50) | 15.10 -> 18.70 (+3.60) |
| 1.0 | 5.59 -> 7.05 | 4.65 -> 6.05 (+1.40 [+0.75, +2.05]) | 4.65 -> 6.65 (+2.00) | 4.65 -> 6.05 (+1.40) | 4.65 -> 6.70 (+2.05) |

- The tuned set was tuned on sigma = 0.5, so its gains there are in-sample.
- PCoder vs PCoderN differ by at most 0.25 pp: the alpha (error) term is ~1e-4..3e-3 of the
  feedforward term with PCoderN and ~1e-9..1e-7 with PCoder (`results/gate1_pc_terms_20261121-075459`).
- t = 0 equals plain VGG16 exactly (max |logit diff| = 0) in every run. Speed 0.46 s/image,
  peak 4.2 GB allocated / 5.8 GB reserved.
- Full tables, pass rule per condition and feature drift: `results/gate1_summary_20261121-075640/summary.md`.

## Upstream versions

Upstream repos are cloned into `external/` (git-ignored, never edited).

| Repo | Commit | Notes |
|---|---|---|
| https://github.com/miladmozafari/predify | `7538a683a383c85dd5909e7aea2b868e931f39ff` | HEAD 2026-01-16; only changes the demo notebook, library code identical to `eca7a741` (2023-09) |
| https://github.com/bhavinc/predify2021 | `e9684141087cf4c5fb7ce7af437edfe4f6970bc5` | HEAD 2022-06-19; last code change `c3d6715a` (2022-01) |

Set up on a new machine (macOS or Windows PowerShell, from the repo root):

```
git clone https://github.com/miladmozafari/predify.git external/predify
git -C external/predify checkout 7538a683a383c85dd5909e7aea2b868e931f39ff
git clone https://github.com/bhavinc/predify2021.git external/predify2021
git -C external/predify2021 checkout e9684141087cf4c5fb7ce7af437edfe4f6970bc5
```

### PVGG16 feedback weights

Source: Dropbox link in the predify2021 README (still works as of 2026-11-19).
These are only the feedback (pmodule) weights; the VGG16 backbone comes from
torchvision (`VGG16_Weights.IMAGENET1K_V1`).

```
mkdir -p weights/predify
curl -L -o weights/predify/weights_pvgg16_imagenet.zip "https://www.dropbox.com/s/8lzp6wfo6n3bymk/weights_pvgg16_imagenet.zip?dl=1"
cd weights/predify && unzip weights_pvgg16_imagenet.zip   # Windows: tar -xf weights_pvgg16_imagenet.zip
```

Check with `shasum -a 256 <file>` (macOS) or `Get-FileHash <file>` (Windows):

| File | Bytes | SHA-256 |
|---|---|---|
| `weights_pvgg16_imagenet.zip` | 315165878 | `379154c7bc4ba089c331f5781b9b5c0231313169ccd970e5a9b365530012f0c6` |
| `weights_pvgg16_imagenet/pvgg16_imagenet_pretrained_pc1_pmodule.pth` | 20353 | `2c1df5f6c7c99d1c72a25aff8c71e5488554293ed8e1b3b11b614303109fd1a7` |
| `weights_pvgg16_imagenet/pvgg16_imagenet_pretrained_pc2_pmodule.pth` | 3278209 | `71bafee34c5d34f1930257cfe8fef3131fc05a5035f9081194d8736e64c23ea1` |
| `weights_pvgg16_imagenet/pvgg16_imagenet_pretrained_pc3_pmodule.pth` | 25691777 | `a5a27d8288893a76975e053d632f1f43fed6c12866a71a6f948a3700134f8fdf` |
| `weights_pvgg16_imagenet/pvgg16_imagenet_pretrained_pc4_pmodule.pth` | 102762625 | `b0c3ee8deb059def935292098eac7297349d4c8cc8f8f18b8bce2d69270e6ba1` |
| `weights_pvgg16_imagenet/pvgg16_imagenet_pretrained_pc5_pmodule.pth` | 205524161 | `0280fab754482ebbc16fbea6ab083921fb7697fc6904334a0776583bd173599f` |
