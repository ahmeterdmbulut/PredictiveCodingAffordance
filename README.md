# PredictiveCodingAffordance

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
cd weights/predify && unzip weights_pvgg16_imagenet.zip
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
