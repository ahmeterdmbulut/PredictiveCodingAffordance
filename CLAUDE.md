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
  small batches.
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
- Never choose hyperparameters, timesteps or thresholds on the test set. Use a
  validation split held out from training (IIT-AFF has none).
- Add a small, hand-checkable test for every metric.
- When unsure about Predify internals, read the code in external/ and cite the
  file and line instead of guessing.
## Key references
- Choksi et al. 2021, Predify, arXiv:2106.02749 (Eq. 2, Appendix A.1-A.2).
- Alamia et al. 2021, feedback and predictive coding, arXiv:2106.04225.
- Tang et al. 2018, recurrent pattern completion, arXiv:1706.02240.
- Nguyen et al. 2017, IIT-AFF; Myers et al. 2015, UMD Part Affordance.
 