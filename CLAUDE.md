# CLAUDE.md
 
Course project for CMPE 58I (Human-Inspired Machine Intelligence, Boğaziçi, Fall 2026).
I want to understand every step, so explain what you do and keep changes small.
 
## Research question
Do Predify-style predictive-coding (PC) dynamics improve part-level affordance
predictions on degraded images, and how does this improvement differ from the
improvement in object-category recognition on the same images?
 
- Affordance = what an object part is for (e.g. knife handle -> grasp, blade -> cut).
- Possible outcomes: affordance-specific gain, generic robustness gain, no gain,
  or feedback suppressing fine part detail. All are valid results. I just want to see the actual result, so never tune toward a desired outcome.
- We do NOT claim that the model "selects between competing actions" or that
  "the brain works like this". Wording: "Predify-style dynamics".
## Design
- Backbone: pretrained PVGG16 from Predify (Choksi et al., 2021), ImageNet.
  Feedforward and feedback weights are FROZEN for the main experiments.
- t = 0 is the plain feedforward pass (must equal standard VGG16 output).
  Evaluate t = 0..10.
- Trained by us: affordance decoder (multi-scale PCoder features ->
  one sigmoid map per affordance) and a linear category read-out on the same
  features. Both trained on CLEAN images at t = 0, then evaluated unchanged at
  every timestep and degradation.
- Datasets: IIT-AFF (main; work on object crops from bounding boxes so that
  affordance and category use the same input) and UMD RGB-D Part Affordance
  (RGB only; category split and novel split). Code must be dataset-agnostic.
- Degradations (TEST images only): gaussian_noise, gaussian_blur, occlusion;
  4 severities + clean, calibrated on validation so category accuracy drops
  roughly 10/25/40/60 % of its clean value.
## PC hyperparameters (beta = ffm, lambda = fbm, alpha = erm)
- Candidate sets: (a) fixed set from Choksi 2021 Sec. 2.2: beta = 0.8,
  lambda = 0.1, alpha = 0.01; (b) PVGG16 tuned set from Appendix A.7 Table S2.
  (Upstream code disagrees with itself: PVGG16SeparateHP defaults, get_model()
  overrides and the mce script all use different values - do not trust them.)
- Both sets are tried in the Gate 1 positive control on ImageNet val. The set
  used for all affordance experiments is chosen there and frozen before looking
  at any IIT-AFF/UMD test data.
- Gradient scaling: the paper (App. A.4) describes scaled error gradients, but
  predify2021's pvgg16_shared.py uses plain PCoder while config_PVGG16.toml sets
  gradient_scaling = true (PCoderN). Test both in the positive control and keep
  the one that reproduces the paper.
- Hyperparameters are float32 (MPS has no float64; upstream set_hyperparams()
  uses float64).
- Feature drift: log per-PCoder ||rep(t) - rep(0)|| / ||rep(0)|| on clean val
  images at every timestep, to separate "feedback suppresses detail" from
  "decoder sees off-distribution features".
## Metrics
- Primary: mIoU excluding background and per-class F1. Accumulate TP/FP/FN over
  the whole test set per class, then compute (no per-image averaging).
- Secondary: weighted F1 (F-beta-w, beta = 1), for comparison with literature.
- Category: top-1 accuracy.
- Recovery ratio = [m(T) - m(0)] / [m(0, clean) - m(0, degraded)], reported
  only when the degradation loss exceeds a margin fixed on validation.
- Break results down by part size (small vs large parts).
- 3 decoder seeds; bootstrap confidence intervals over test images.
## Baselines and controls (all run with the same number of timesteps)
1. t = 0 (no PC) - main baseline.
2. Random and shuffled feedback weights - sanity check only.
3. Filter baseline: denoise/sharpen the degraded image, then t = 0.
4. Predify ablations: alpha = 0 (no error correction), lambda = 0 (no feedback).
5. Supervised feedback: same deconv pathways trained with task loss instead of
   reconstruction.
6. Feedback retrained on the target dataset vs ImageNet feedback (ablation).
7. Object-mask baseline: predict each affordance as the whole object extent.
8. Positive control: reproduce Predify's Gaussian-noise gain on ImageNet
   classification.
Optional if time: PEfficientNetB0; CORnet-S as an external recurrent reference.
## Gates
- Gate 1: Predify weights load and the Gaussian-noise gain reproduces.
  If not -> Plan B2: small 3-layer CNN with PC dynamics (Alamia et al., 2021).
  Target: Choksi 2021 Table S4, PVGG16 top-1, t = 0 -> t = 15:
  sigma 0: 71.63 -> 71.47; 0.5: 35.61 -> 38.59; 0.75: 16.69 -> 18.46;
  1.0: 5.59 -> 7.05. Pass if t = 0 accuracies are close and the gain has the
  same sign and rough size. The positive control runs t = 0..15 (affordance
  experiments stay t = 0..10).
  Table S4 / Fig. 2a were produced with the "fixed" set (App. A.7), so the
  primary Gate 1 comparison uses the fixed set. The tuned set was tuned on
  sigma = 0.5 Gaussian noise, which is in-sample for it.
  PCoderN implements App. A.4 (gradient x K/sqrt(C), batch-size independent);
  plain PCoder makes the error step negligible (~B*K/sqrt(C) smaller) and
  batch-size dependent.
  Preprocessing: Resize(224) + CenterCrop(224) + ImageNet Normalize, as in
  Predify (calculate_corr_dists.py L75-79), not torchvision's 256 -> 224.
  Noise convention (predify2021 manifold_projection/calculate_corr_dists.py
  L75-79, L128, commit e9684141): N(0, sigma) added AFTER ImageNet Normalize
  (normalized-image units), no clamping. Indirect evidence: the repo has no
  Gaussian-noise accuracy script. Use our own seeded generator (upstream
  reseeds the global RNG every batch).
- Gate 2: decoder beats the object-mask baseline, seeds agree, and on
  clean images score(t = 10) is close to score(t = 0). If not -> Plan B1:
  image-level multi-label affordance prediction (mAP).
## Environment
- Develop on MacBook (Apple Silicon: mps or cpu); long runs on a Windows desktop
  with an RTX 3060 (cuda), synced through git (push on Mac, pull and run on
  Windows); Kaggle as overflow.
- Device-agnostic: pick cuda > mps > cpu automatically, overridable. No hardcoded
  .cuda(). Use pathlib, no Linux-only shell assumptions.
- PC error correction needs autograd at inference: watch GPU memory, start with
  small batches. One PC step costs ~8x a VGG16 forward.
- Compute rule: one backbone pass per (image, condition, PC variant); all
  decoder seeds are read out from the same pass.
- Predify repos live in external/ (git-ignored). Do not edit them; wrap them in
  src/. Record the commit hash used in README.
- Data in data/, weights in weights/, both git-ignored. Paths come from configs/.
## Repo layout
configs/   experiment and path configs (yaml)
src/       modules (load_predify, datasets, degrade, decoder, train_decoder,
           evaluate, baselines, analysis)
notebooks/ visual checks
results/   one folder per run: config + metrics + figures
notes.md   my own notes (do not edit unless asked)
 
## Working rules
- One module per task. Plan first, then implement, then show me how to verify it.
- Keep functions small, typed and commented, but not overly-commented.  State tensor shapes in comments.
- Fixed seeds everywhere. Every run saves its config next to its results.
- Every script has an `if __name__ == "__main__":` guard (DataLoader workers
  use spawn on both macOS and Windows).
- Never choose hyperparameters, timesteps or thresholds on the test set. Use a
  validation split held out from training (IIT-AFF has none).
- Add a small, hand-checkable test for every metric.
- When unsure about Predify internals, read the code in external/ and cite the
  file and line instead of guessing.
## Key references
- Choksi et al. 2021, Predify, arXiv:2106.02749 (Eq. 2, Sec. 2.2, Appendix A.1-A.2,
  A.4, A.7 Table S2, Table S4).
- Alamia et al. 2021, feedback and predictive coding, arXiv:2106.04225.
- Tang et al. 2018, recurrent pattern completion, arXiv:1706.02240.
- Nguyen et al. 2017, IIT-AFF; Myers et al. 2015, UMD Part Affordance.
 