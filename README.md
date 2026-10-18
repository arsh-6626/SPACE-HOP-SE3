# SPACE-HOP

**Spacecraft 6-DoF Pose Estimation using Embedding-Predictive Pretraining and Hopf Map**

SPACE-HOP estimates the 6-DoF pose of an uncooperative spacecraft from a single monocular image — no PnP solver, no CAD model, no test-time adaptation.

[[Paper](https://openaccess.thecvf.com/content/CVPR2026W/AI4Space/papers/Dutta_SPACE-HOP_Spacecraft_6-DoF_Pose_Estimation_using_Embedding-Predictive_Pretraining_and_Hopf_CVPRW_2026_paper.pdf)] [[BibTeX](#citation)] [[Project Page](spacehop.github.io)]

## How it works

Training happens in two stages. First, an encoder/predictor ViT pair is pretrained with embedding-predictive learning (`ijepa/`): a context encoder sees the visible patches of an image, a target encoder sees masked patches, and a lightweight predictor learns to guess the target embeddings from the context embeddings in latent space, with no pixel reconstruction. Masks aren't random — they're sampled around projected spacecraft keypoints (corners, antennas, panel edges) so the encoder is pushed to learn rigid-body geometry rather than surface texture.

Second, this pretrained backbone is dropped into the pose model (`src/model.py`). The `[CLS]` token feeds three heads: a translation head that regresses a normalized 3D offset later converted to metric translation via the camera intrinsics; a rotation classifier that cross-attends over patch tokens to pick the closest anchor from a precomputed Hopf-fibration grid of `SO(3)` rotations (`src/hopf_grid.py`); and an offset head that predicts a small Lie-algebra correction for the chosen anchor, which is mapped back onto the manifold with the exponential map to recover a continuous, sub-degree-accurate rotation. The whole thing is a single forward pass — no PnP solver, no CAD model, no iterative refinement at inference time.

Evaluated on [SPEED+](https://taehajeffpark.com/speedplus). See the [paper](#) for the full architecture, loss formulation, and results across domains.

## Repository structure

```
SPACE-HOP/
├── configs/
│   └── speedplus.yaml          # main experiment config (data, model, training)
├── ijepa/                       # embedding-predictive pretraining stage (fork of facebookresearch/ijepa)
│   ├── main.py / main_distributed.py
│   ├── configs/                 # *_vitb16_ep200.yaml configs
│   └── src/                     # ViT models, datasets, masks, augmentations, train.py
└── src/                          # pose estimation stage
    ├── model.py                  # FastPoseViT: JEPA backbone + Hopf rotation head (pass 0)
    ├── hopf_grid.py              # Hopf fibration SO(3) anchor grid + closest-anchor search
    ├── train.py / val.py         # training / evaluation loops
    ├── box.py                    # 3D wireframe overlay visualization
    ├── datasets/speedplus.py     # SPEED+ dataset loader
    ├── isro_data.py              # ISRO GSAT-12R dataset loader + oracle crop
    ├── refine.py                 # Refiner ViT (diff/siamese patch embed, attention pooling)
    ├── hopf_se3.py               # SE(3) Hopf-fibred iterative refiner (canonical-warp fibre removal)
    ├── nested_se3.py             # retrieval-free nested coarse-to-fine SE(3) bin classifier
    ├── spacehop_nested.py        # nested refiner folded into SPACE-HOP: one shared ViT, pass 0 + nested passes
    ├── so3_nested.py             # strict-hierarchy Hopf-HEALPix SO(3) distribution grid (SYMSOL)
    ├── helper/jensen_audit.py    # symmetry-instability (Jensen Gain) diagnostic
    └── utils/                    # geometry, augmentation, image, plotting utilities
```

## Setup

```bash
git clone https://github.com/arsh-6626/SPACE-HOP-SE3.git
cd SPACE-HOP-SE3
pip install torch torchvision opencv-python pandas scipy numpy tqdm wandb pyyaml
```

You'll also need the [SPEED+](https://taehajeffpark.com/speedplus) dataset and the Tango spacecraft keypoint file (`tangoPoints.mat`).

## Usage

**1. Pretrain the encoder.** Follow `ijepa/README.md` — clone upstream [i-JEPA](https://github.com/facebookresearch/ijepa), drop in the contents of `ijepa/`, point a config at your SPEED+ root, and run:

```bash
python main.py --fname configs/speedplus_vitb16_ep200.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3
```

**2. Configure.** Edit `configs/speedplus.yaml` with your dataset paths, keypoints file, and the pretrained JEPA checkpoint path (`model.jepa_path`).

**3. Train the pose head.**

```bash
python src/train.py
```

**4. Evaluate.**

```bash
python src/val.py
```

Reports translation error `E_T` and geodesic rotation error `E_R` across the Synthetic, Sunlamp, and Lightbox domains.

**5. Symmetry diagnostics.** `src/helper/jensen_audit.py` rotates each input through `K` in-plane rotations, canonicalizes the predictions, and reports the Jensen Gain — a measure of orientation-dependent instability useful for diagnosing axial-symmetry failure modes.

## ISRO GSAT-12R dataset

The dataset ships 2D landmarks and poses but no camera matrix or 3D keypoints. `src/tools/calibrate_isro.py` recovers both by triangulating the landmarks across views (median reprojection error < 0.01 px) and writes `configs/isro_camera.json`: `fx = fy ≈ 938.7`, `(cx, cy) = (960, 600)`, quaternions are `(w, x, y, z)` with `X_cam = R·X_body + t`.

```bash
python src/tools/calibrate_isro.py --root /path/to/ISROGSAT12RDataset --out configs/isro_camera.json   # already committed
PYTHON=python bash scripts/run_isro.sh   # JEPA pretraining -> pose training -> val/test eval
```

- Pretraining (`ijepa/configs/isro_vitb16.yaml`) and pose training (`configs/isro.yaml`) use only the train split of `cv_splits4` fold 1. Model selection uses the fold-1 val split. `images/test` is evaluated once at the end.
- Crops are squares centred on the projected landmarks, with side 1.6× their extent. They are derived from ground truth (oracle detection), as in the SPEED+ protocol. Train-time jitter is relative to the crop size.
- `src/val.py` reports E_T (m), E_T/|t|, geodesic E_R (°) and E_T/|t| + E_R[rad]. It also writes per-image errors.

## SE(3) Hopf refinement and nested distributions

This fork extends SPACE-HOP from a single forward pass to a **coarse-to-fine SE(3) estimator**, and towards a
**distribution** over pose rather than a point estimate. The base pose model is unchanged — it becomes *pass 0*.

**Canonical-warp fibre removal (`src/hopf_se3.py`).** Camera-relative `SE(3)` is split into a 2-DoF base point (the
*swing*: which side of the object faces the camera) and a 4-DoF fibre (viewing ray, in-plane roll, range). The fibre
is removed *exactly* by warping the full image through a homography into a virtual camera pointed along the estimated
ray and rolled by the estimated twist, with focal length set so the object fills a fixed-size crop. In that canonical
view the only remaining error is a small residual, which a `[q, q − r]` ViT (`src/refine.py`) predicts and applies as
an exact `SE(3)` update. Iterating re-warps at each step. (The ray/roll warp is exact; the range term is an
approximate zoom, re-corrected each iteration.)

**Retrieval-free nested classifier (`src/nested_se3.py`).** Instead of regressing the residual, classify it on local
grids in the fibre coordinates — tilt (2-D), roll (1-D), ray offset (2-D), range (1-D) — with spacing shrinking by 3×
each level, decoded with a sub-cell parabola fit. Gaussian-anchored bins give the classifier a regression-strength
gradient while still representing skewed or multi-modal (ambiguous) residuals. No exemplar retrieval at test time.

**One shared backbone (`src/spacehop_nested.py`).** The nested head is folded into SPACE-HOP so a *single* ViT serves
both pass 0 (global Hopf grid) and the nested refinement passes (canonical warp), trained jointly with a
frozen-backbone warm-up for the new head.

**Distribution benchmarks (`src/so3_nested.py`).** For SYMSOL / SYMSOL-T, the overlapping refinement grid is replaced
by the strict-hierarchy, equal-volume Hopf-HEALPix grid of [Implicit-PDF](https://implicit-pdf.github.io/)
(72 cells at level 0, ×8 per level). A distribution is a categorical over level-0 cells and an 8-way categorical at
each expanded node; the exact log-likelihood of a rotation is the summed log-probability along its cell path.
`python src/so3_nested.py selftest` verifies the grid (exact cell round-trip; a uniform predictor reaches the
`−ln(π²) ≈ −2.29` nats log-likelihood floor at every level).

> These components are validated on an internal spacecraft dataset; public-benchmark (SYMSOL, T-LESS, YCB-V) training
> and evaluation are in progress.

## Citation

```bibtex
@inproceedings{dutta2026spacehop,
  title     = {SPACE-HOP: Spacecraft 6-DoF Pose Estimation using Embedding-Predictive Pretraining and Hopf Map},
  author    = {Dutta, Arnabi and Naqvi, Arsh Abbas and Singh, Kavinder and Gautam, Lavendra and Parihar, Anil Singh and Bangare, Shakti and Lagisetty, Ravi Kumar},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2026}
}
```

## Acknowledgements

Supported by ISRO under the RESPOND project [RES-URSC-2023-023]. Thanks to the FAE Team, AMD India, for GPU resources. Pretraining builds on [I-JEPA](https://github.com/facebookresearch/ijepa) and is inspired by the keypoint-anchored masking strategy of RoboPEPP.
