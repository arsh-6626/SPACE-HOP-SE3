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
    ├── model.py                  # FastPoseViT: JEPA backbone + Hopf rotation head
    ├── hopf_grid.py              # Hopf fibration SO(3) anchor grid + closest-anchor search
    ├── train.py / val.py         # training / evaluation loops
    ├── box.py                    # 3D wireframe overlay visualization
    ├── datasets/speedplus.py     # SPEED+ dataset loader
    ├── helper/jensen_audit.py    # symmetry-instability (Jensen Gain) diagnostic
    └── utils/                    # geometry, augmentation, image, plotting utilities
```

## Setup

```bash
git clone https://github.com/Team-M3OW/SPACE-HOP.git
cd SPACE-HOP
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
