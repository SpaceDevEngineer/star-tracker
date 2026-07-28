# Deep-Learning-Based Star Tracker — Spacecraft Attitude Determination

*Master's Thesis Project, 2025–2026*

---

## One-line summary (CV project entry)

**Deep Learning Star Tracker — Spacecraft Attitude Determination from Real Satellite Imagery**

Built an end-to-end "lost-in-space" attitude determination pipeline that recovers a
spacecraft's 3-axis orientation from a single raw star-field image, achieving sub-pixel
accuracy (median 8.98″ ≈ 0.44 px) on real TESS satellite data. Combined a CNN star
detector (U-Net / HRNet) with RANSAC triangle star identification, Wahba SVD, and
nonlinear plate-solve refinement.

---

## Portfolio description

**Deep-Learning-Based Star Tracker for Spacecraft Attitude Determination**

Designed and implemented a complete star-tracker pipeline that determines a spacecraft's
orientation (attitude quaternion) from a single image of the star field, with no prior
knowledge of pointing direction ("lost-in-space" mode). Validated on **real TESS
satellite imagery** (NASA's Transiting Exoplanet Survey Satellite), not simulations.

Pipeline: raw FITS image → CNN star detection → camera-frame body vectors → RANSAC
triangle identification against the Hipparcos star catalog → Wahba's problem (SVD) →
iterative plate-solve refinement → attitude quaternion.

Key contributions:

- Trained and compared two CNN architectures (U-Net, 7.7M params; HRNet, 4.0M params)
  for star centroid detection via heatmap regression — observed a 0.2″ median gap
  while HRNet used roughly half the model size; formal equivalence testing remains
  future work.
- Diagnosed a critical geometric failure: wide-FOV TESS optics introduce 6th-order SIP
  polynomial distortion that a naïve pinhole projection cannot model (200–1000″ errors
  at frame corners). Re-architected the geometry layer around SIP-corrected body vectors.
- Implemented a two-pass refinement: RANSAC triangle lock → nonlinear least-squares
  plate-solve over (RA, Dec, roll) → sub-pixel attitude.
- Added a quality gate that rejects false constellation locks, converting silent
  catastrophic errors into honest "no-solution" returns.
- Built diagnostic visualizations and an interactive Streamlit walkthrough of every
  pipeline stage.

Result: 104/107 valid frames solved (97.2%), 0 false locks, **median attitude error
8.98″ (~0.44 px at 20.57″/px plate scale)** in full lost-in-space mode.

---

## Detailed technical description

### Problem

A star tracker is a spacecraft sensor that determines orientation by photographing stars
and matching them to a catalog. The "lost-in-space" problem — recovering attitude with
zero prior information — is the hardest case. This thesis tackled it on **real TESS
satellite images** rather than synthetic data.

### Approach

1. **Data pipeline** — Converted ~800 TESS TICA Full-Frame-Image FITS files
   (sectors 27–101) into a training dataset: background-subtracted 8-bit PNGs,
   512×512 non-overlapping tiles, JSON labels with star positions, Hipparcos catalog
   matches, and the full WCS solution including SIP distortion coefficients.

2. **Star detection (deep learning)** — Trained CNNs for heatmap regression: each star
   rendered as a Gaussian blob; the model predicts a per-pixel probability map.
   Weighted-MSE loss (star pixels ×20), cosine LR schedule, best checkpoint selected by
   detection F1. Compared **U-Net** (7.76M parameters) and **HRNet** (3.99M parameters)
   under identical training conditions.

3. **Camera geometry** — Converted detected pixel centroids to 3-D body unit vectors.
   Identified that linear pinhole projection fails on TESS's 12° field of view due to
   optical distortion; replaced it with **SIP-polynomial-corrected gnomonic projection**,
   making body-vector angles consistent with catalog angles. Explicitly factored the
   rotational part out of the FITS CD matrix and initialized physical roll from the
   blind RANSAC/Wahba rotation rather than from ground-truth WCS orientation.

4. **Star identification** — RANSAC-based triangle matching: random detection triplets
   matched against a pre-computed Hipparcos pair-angle database (5.3M pairs,
   binary-search lookup), with chirality filtering to reject mirror solutions, then
   Wahba SVD for a candidate rotation, verified against all detections.

5. **Attitude refinement** — Two-pass plate-solve: rough Wahba rotation → synthesize a
   candidate WCS → re-project the full catalog → snap matches at tightening pixel
   tolerances (30→5→1.5 px) → nonlinear least-squares fit of (RA, Dec, roll) minimizing
   pixel residuals.

6. **Reliability** — A quality gate on the post-solve pixel residual rejects
   self-consistent-but-wrong constellation locks.

### Results

Real TESS test imagery, lost-in-space mode, no ground-truth hints:

| Metric | Value |
|---|---|
| Valid evaluation frames | 107 / 120 (13 lack WCS calibration) |
| Attitude solve rate | 97.2% (104/107), 0 false locks |
| Median attitude error | 8.98″ (0.44 px) |
| 90th percentile | 19.41″ (0.94 px) |
| Cross-boresight / roll median | 2.74″ / 8.33″ |
| Centroid-limited benchmark (U-Net / HRNet) | 4.6″ / 4.8″ median |
| Best single image | 1.10″ (0.05 px) |

The small observed detector gap suggests that architecture is not the dominant
accuracy bottleneck. Residual optical distortion is the stronger limitation,
identifying camera factory-calibration as the path to sub-arcsecond accuracy.

---

## Technologies

**Languages & ML:** Python, PyTorch (CNN architecture design, training), NumPy,
SciPy (SVD, nonlinear least-squares optimization)

**Domain libraries:** Astropy (WCS, SIP distortion, FITS), Pandas, Matplotlib, Streamlit

**Computer vision:** heatmap regression, CNN encoder-decoder architectures
(U-Net, HRNet), sub-pixel centroid extraction

**Algorithms:** RANSAC, Wahba's problem / SVD attitude solver, gnomonic projection,
plate-solving, quaternion mathematics

**Astronomy / aerospace:** star tracker design, ICRS coordinate frames, attitude
determination, Hipparcos catalog, TESS mission data

**Infrastructure:** SLURM cluster, NVIDIA RTX 2080 Ti GPU training, Git, Linux

---

## Skills demonstrated

- End-to-end ML system design (data pipeline → model → classical post-processing → evaluation)
- Deep learning: CNN architecture comparison, heatmap-regression training, hyperparameter tuning
- Classical computer vision & numerical methods: RANSAC, SVD, nonlinear optimization
- Root-cause debugging of a complex multi-stage system (diagnosed an optical-distortion
  failure invisible in synthetic benchmarks)
- Working with real, noisy scientific data rather than simulations
- Scientific software engineering and reproducible experiments
- Building diagnostic tooling and interactive demos
