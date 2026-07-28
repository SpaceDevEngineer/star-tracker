# Detailed Results

End-to-end lost-in-space attitude determination on real TESS imagery, no `--use-gt`
hints, no initial pose, no IMU prior. All attitude errors use physical body roll
about camera +x, not the FITS CD-matrix angle.

## Full evaluation — 120 real TESS frames

The held-out evaluation contains 120 frames across multiple TESS sectors. Thirteen
labels lack WCS camera calibration and are excluded as unsolvable data failures.
The remaining 107 frames are the algorithm denominator.

| Metric | Result |
|---|---:|
| Valid / total frames | 107 / 120 |
| Correctly solved | **104 / 107 (97.2%)** |
| Algorithm refusals | 3 |
| False locks | **0** |
| Median attitude error | **8.98″ (0.44 px)** |
| 90th / 95th percentile | **19.41″ / 22.15″** |
| Minimum / maximum | 1.10″ / 42.51″ |
| Cross-boresight median / p90 | **2.74″ / 4.85″** |
| Roll median / p90 | **8.33″ / 19.27″** |
| Below one pixel (20.57″) | **95 / 104** |
| Below one arcminute | **104 / 104** |

The three refusals are counted against solve rate rather than silently removed.
Machine-readable values and provenance are stored in
[`Results/full_test_metrics.json`](Results/full_test_metrics.json); the status,
error, match count, RANSAC iterations, and timing for every frame are in
[`Results/full_test_per_frame.csv`](Results/full_test_per_frame.csv).

### Attitude-convention correction

Early artefacts used the FITS CD angle as the field named `roll`. With this camera
axis convention (`pixel x → -body y`), physical roll is:

```text
physical_roll = (180° - CD_angle) mod 360°
```

The current pipeline factors CD rotation out of the camera calibration, initializes
physical roll from the blind RANSAC/Wahba rotation, and applies the equation above
only when converting legacy evaluation artefacts. Recomputing quaternion errors in
the physical convention changed the full-test median from the previously quoted
7.35″ to the corrected **8.98″**. Solve/refusal decisions and pixel-space fits are
unchanged.

## Bundled reproducible demo — 16 TESS images

Test set: 16 NASA TESS TICA Full-Frame-Images (sector s1751), each 2136 × 2078 px,
plate scale 20.57 arcsec/px. This compact subset is included so Replay and batch
evaluation work from a normal Git clone.

| Metric | Before fixes | **After SIP correction + plate-solve + quality gate** |
|---|---|---|
| Frames with sufficient detections | 16/16 | 16/16 |
| Attitude solve rate | 0% (all `id_failed`) | **93.8% (15/16)** |
| Median angular error | — | **6.63″ (0.32 px)** |
| 90th percentile | — | 18.13″ (0.88 px) |
| Max error (among solved) | — | 26.57″ (1.29 px) |
| False locks | — | **0** (1 honest refusal) |
| Best single image | — | 2.31″ (0.11 px) |

### Per-image breakdown

```
cam1/ccd1   4.88″       cam2/ccd1   26.57″      cam3/ccd1   14.68″      cam4/ccd1   4.88″
cam1/ccd2  20.44″       cam2/ccd2    8.26″      cam3/ccd2   FAIL ✋     cam4/ccd2   8.40″
cam1/ccd3   6.11″       cam2/ccd3    6.63″      cam3/ccd3    2.82″      cam4/ccd3   6.91″
cam1/ccd4   6.34″       cam2/ccd4    2.31″      cam3/ccd4    2.45″      cam4/ccd4  12.97″
```

14 of 15 solved frames are sub-pixel. `cam3/ccd2` is rejected via quality gate
(median Euclidean residual 20.4 px,
only 17 matches) — the pipeline honestly declined to emit a wrong attitude rather
than reporting a 632 000″ false lock.

![Summary](docs/images/summary.png)
*Angular error per image (log scale, left), accuracy vs match count (centre),
per-stage timing (right).*

## Centroid-quality benchmark — 120 test images, `--use-gt` mode

`--use-gt` synthesises reference vectors from the ground-truth quaternion so that
only the detector's centroid noise contributes. This isolates the deep-learning
model from the catalog-matching geometry.

| Detector | Parameters | Solve rate | Median error | 90th pct | Max |
|---|---:|---:|---:|---:|---:|
| **U-Net**  (run 3) | 7,762,465 | 100% | **4.6″** | 9.5″ | 13.5″ |
| **HRNet** (run 1) | 3,987,393 | 100% | **4.8″** | 9.7″ | 13.3″ |

The observed median gap is only 0.2″ while HRNet uses roughly half the parameters.
This makes HRNet a promising embedded alternative, but a paired bootstrap confidence
interval is still required before claiming statistical equivalence.

![Example overlay](docs/images/example_overlay.png)
*Red = U-Net detections, cyan = projected catalog after plate-solve, green lines
= matched pairs. Final attitude error 4.88″.*

## Per-star residual distribution

For a single image, residuals between observed pixel positions and catalog stars
projected through the final attitude:

![Residual map](docs/images/example_residuals.png)
*Left: each matched star coloured by its pixel-space residual. Right: histogram
of residuals across all 55 matches. Median residual = 4.79″, max 20.66″.*

Edge stars (top-left, bottom-right) carry slightly larger residuals — that's the
remaining attitude-free SIP residual visible. Factory CD-shear calibration would
reduce this further.

## Timing — single image, single-thread CPU

| Stage | Latency |
|---|---|
| U-Net detection (16 tiles) | ~0.5 s |
| RANSAC triangle ID | seconds … several minutes (depends on convergence) |
| Plate-solve refinement | ~1–3 s |
| **Median total** | **~2 min** |

The implementation now uses binary pair-angle lookup, vectorized third-star search,
and a 12° FOV-cone cull during verification. RANSAC sampling remains the dominant
and most variable CPU cost.

## Pipeline architecture

```
PNG (2136 × 2078)
   │
   ▼  STAGE 1 — Deep-learning star detection
U-Net / HRNet on 16 non-overlapping 512 × 512 tiles
   │  ~450 – 480 sub-pixel centroids + flux per centroid
   ▼  STAGE 2 — Body vectors
SIP-polynomial-corrected gnomonic projection (astropy.wcs.sip_pix2foc)
   │  body unit vectors in camera-fixed frame, attitude-independent
   │  top-60 brightest passed to triangle search
   ▼  STAGE 3 — RANSAC triangle star ID (Wahba SVD #1)
Random pair → Hipparcos pair-DB lookup (5.3 M pairs, binary search)
   → random 3rd detection → matching 3rd catalog star
   → chirality filter rejects mirror triplets
   → Wahba SVD on 3 pairs → candidate R
   → verify against all detections at 200″ tolerance
   │  ≤ 1500 iterations, keep best by verified count
   ▼  STAGE 4 — Iterative WCS refinement (Wahba SVD #2 via plate-solve)
For tol in (30 px, 5 px, 1.5 px):
    build WCS from current pose
    → project full mag<7.5 catalog
    → snap detections to nearest projection
    → scipy.optimize.least_squares over (RA, Dec, roll)
   ▼  STAGE 5 — Quality gate
Reject if median Euclidean per-star residual > 2 px or fewer than 8 matches survive
   │
   ▼
Attitude quaternion + (RA, Dec, roll)
```

## Models trained

| Model | Architecture | Params | Loss | Notes |
|---|---|---:|---|---|
| **U-Net** | 4-level encoder-decoder, skip connections | 7,762,465 | Weighted MSE (star × 20) | Run 3, best F1 = 0.636 at threshold 0.55 |
| **HRNet** | Multi-resolution parallel branches | 3,987,393 | Identical hyperparameters | Run 1 |

Training conditions held identical between the two models:

- 50 epochs, Adam @ lr = 1 e-3 + cosine annealing → 1 e-6
- Heatmap target: Gaussian σ = 1.5 px, peak normalised to 1.0
- Split 70 / 15 / 15 by image (560 / 120 / 120), seed = 42 (no tile-level leakage)
- Augmentation: random fliplr + flipud + rot90 (train only)
- Best checkpoint selected by validation F1 (not by loss — loss is a poor proxy for
  detection quality on highly sparse targets)
