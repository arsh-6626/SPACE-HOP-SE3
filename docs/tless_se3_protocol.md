# T-LESS SE(3) protocol (to match SpyroPose, Haugaard et al. 2023)

Source: SpyroPose arXiv:2303.05308 v2, Sec 3.2 + Tables 5/6. Baseline for `tab:tless`.

## Translation box (object-dependent, metres)
`A = [[d, 0, t̂x], [0, d, t̂y], [0, 0, t̂z]]`, `g ∈ [-1/2, 1/2]^3`, `t = t̂ + A g`.
- `d` = object diameter (models_eval/models_info.json, mm → m)
- `t̂` = coarse position estimate (detector at test; metres)
- `det(A) = d^2 · t̂z`; total SE(3) volume = `π^2 · det(A)`
- In-plane bound ±d/2; depth bound down to half distance (size ~doubled).

## Rotation measure
HealPix-SO(3) (Yershova), `V(SO3) = π^2` — IDENTICAL to our se3_head V_SO3. ✓

## Uniform log-likelihood (acceptance test — REPRODUCED)
`uniform_LL = mean_i [ -ln(π^2 · d^2 · t̂z,i) ]` over evaluated test instances.
Reproduced from real diameters + GT depths on BOP19 test targets (jepa env):
  obj 1  3.484 (rep 3.5) | obj14 3.249 (3.3) | obj25 2.421 (2.4) | obj27 1.733 (1.7)
  avg 2.722 (rep 2.7). All within ±0.05 (residual = detector-t̂ vs GT-t_z).

## Protocol facts that shape the loader
- Train on **PBR** (train_pbr), test on **real** (test_primesense), BOP19 targets.
- t̂ from a **detector** at test. At train, simulate detector noise: sample
  t̂ = t_GT + A·ẽ, ẽ ~ truncated N(0, σI, ||ẽ||≤1/2), so GT lands at a varying
  offset inside the box (else g≡0 and translation is untrained — degenerate).
- Crop: centred on projected t̂, tight-ish (bbox from scene_gt_info.json), 224.
- **LL scored at the single annotated GT pose** — NOT symmetry-averaged. The model
  learns the multimodal density; we evaluate density at the one GT. So NO
  symmetry-equivalent expansion for this table (unlike SYMSOL-T test path).
- Objects 1/14/25/27; train-crop counts 16k/19k/23k/24k. Per-object models in
  SpyroPose; we can do per-object or one object-conditioned joint model.
- Data on disk: /media/kavinder/hdd2/liepose_data/bop_datasets/tless (33 GB,
  train_pbr + test_primesense + models complete). Read in place; write our
  crops/geometry to hdd3 (6.9 TB free), never the SSD or hdd2.
