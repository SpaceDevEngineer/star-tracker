# Star Tracker — Spacecraft Attitude Determination from Real Satellite Imagery

**Sub-pixel "lost-in-space" attitude estimation on real NASA TESS imagery, built with deep
learning + classical geometry.** Recovers a spacecraft's 3-axis orientation from a single
PNG of the star field with no prior pose information.

![Pipeline overview](docs/images/pipeline_cam1-ccd1.png)
*One image, six pipeline stages: from raw TESS frame to attitude quaternion.*

---

## Results in one table

End-to-end lost-in-space attitude on real TESS test imagery (no ground-truth correspondences):

| Metric | Value |
|---|---|
| Attitude solve rate | **15 / 16 (94%)** |
| Median angular error | **6.83″** (0.33 px at 20.6″/px plate scale) |
| 90th percentile | 18.22″ (0.89 px) |
| Maximum error | 27.01″ (1.31 px) |
| False locks | **0** — one image honestly refused via quality gate |
| Best single image | 2.27″ (0.11 px) |

Detector benchmark (centroid quality, `--use-gt` mode, 120 test images):

| Detector | Parameters | Median error |
|---|---:|---:|
| U-Net  | 7.76 M | 4.6″ |
| HRNet  | **3.99 M** | 4.8″ |

HRNet matches U-Net at half the parameter budget — **the detector architecture is
not the accuracy bottleneck**; the optical-distortion calibration is.

---

## What this is

A complete star-tracker pipeline that converts a single raw image of the sky into
the camera's attitude quaternion. Validated on **real TESS satellite imagery** (not
simulations), in true *lost-in-space* mode — no initial guess, no IMU prior, no
catalog correspondences.

```
        PNG image (2136 × 2078)
                │
                ▼
   ┌────────────────────────────┐
   │  Stage 1 — CNN detection   │   U-Net / HRNet → ~480 sub-pixel centroids
   └────────────────────────────┘
                │
                ▼
   ┌────────────────────────────┐
   │  Stage 2 — Body vectors    │   SIP-corrected gnomonic projection
   └────────────────────────────┘
                │
                ▼
   ┌────────────────────────────┐
   │  Stage 3 — Triangle ID     │   RANSAC + Hipparcos pair-DB + Wahba SVD
   └────────────────────────────┘
                │
                ▼
   ┌────────────────────────────┐
   │  Stage 4 — Plate-solve     │   scipy least-squares over (RA, Dec, roll)
   └────────────────────────────┘
                │
                ▼
   ┌────────────────────────────┐
   │  Stage 5 — Quality gate    │   honest "no-solution" if residual > 2 px
   └────────────────────────────┘
                │
                ▼
       Attitude quaternion
```

---

## Why I think this is interesting

1. **Real satellite data, not synthetic.** Most public star-tracker projects train
   and evaluate on procedurally generated star fields. TESS images carry real noise,
   real optical distortion, real diffraction spikes, and saturated bright stars.

2. **Diagnosed a non-obvious failure mode.** Initial linear pinhole geometry hid a
   200–1000″ residual at the field corners (TESS uses a 6th-order SIP polynomial for
   its 12° FOV). The synthetic `--use-gt` benchmark looked perfect (4.6″ median) but
   the catalog-based end-to-end pipeline silently failed on every image. The fix was
   to apply SIP correction via `astropy.wcs.sip_pix2foc` before the projection — a
   small change with a large effect.

3. **Two-pass refinement.** RANSAC Wahba lock + scipy plate-solve. The first finds a
   rough constellation; the second turns that into sub-pixel attitude by minimizing
   pixel residuals through the SIP forward projection.

4. **A quality gate.** Wrong locks fail loudly rather than silently. Production star
   trackers can't return wildly wrong attitudes; this pipeline refuses to answer
   when the post-solve median pixel residual exceeds 2 px.

5. **Interactive walkthrough.** Streamlit app runs every stage live with real numbers,
   not just plots.

---

## Try it

### Interactive Streamlit walkthrough

```bash
git clone https://github.com/SpaceDevEngineer/star-tracker.git
cd star-tracker
pip install -r requirements.txt
streamlit run Code/Streamlit_app/pipeline_app.py
```

Opens at <http://localhost:8501>. Pick any of the 16 demo TESS images in the sidebar
and press *Run pipeline*. Each stage materialises with the actual body vectors,
triangle-angle table, plate-solve iterations, and final pose comparison.

### End-to-end batch evaluation

```bash
python Code/Star_ID/inference_full.py \
    --data-dir Data/dataset_tess_test \
    --model    Results/unet_run3/best_model.pt \
    --catalog  Data/hybrid/catalog_hipparcos_full.csv \
    --mag-limit 7.5 \
    --out-dir  Results/star_id_run
```

Writes per-image JSON artefacts and a summary table.

### Generate the visualisations

```bash
python Code/Star_ID/visualize_inference.py \
    --data-dir Data/dataset_tess_test \
    --run-dir  Results/star_id_run \
    --out-dir  Results/star_id_run/viz
```

Produces per-image overlays (detections + projected catalog + match lines), per-star
residual maps, and a project summary chart.

---

## Repository layout

```
star-tracker/
├── README.md              ← this file
├── PROJECT_DESCRIPTION.md ← longer technical write-up
├── RESULTS.md             ← detailed results tables
├── requirements.txt
├── Code/
│   ├── Star_ID/
│   │   ├── inference_full.py       ← main lost-in-space pipeline
│   │   ├── triangle_id.py          ← RANSAC star identification
│   │   ├── visualize_inference.py  ← per-image diagnostic plots
│   │   └── visualize_pipeline.py   ← 6-panel pipeline trace
│   ├── Model_train_code/train.py   ← U-Net architecture + training
│   ├── HRNet_train/                ← HRNet architecture + training + inference
│   ├── Tess_Dataset/process_tess.py ← FITS → PNG + JSON labels (with SIP WCS)
│   └── Streamlit_app/pipeline_app.py
├── Data/
│   ├── dataset_tess_test/  ← 16 TESS demo images + labels (28 MB)
│   └── hybrid/             ← Hipparcos catalog (5 MB)
└── Results/
    └── unet_run3/best_model.pt   ← trained U-Net weights (30 MB)
```

The trained HRNet weights, full 800-image training set, and FITS archives live with the
research project; this repo carries only what's needed for the demo and to read the code.

---

## Tech stack

**Languages & ML:** Python, PyTorch (U-Net + HRNet, heatmap regression), NumPy, SciPy
(SVD + nonlinear least-squares).
**Astronomy / geometry:** Astropy (FITS, WCS, SIP polynomial), gnomonic projection,
quaternion algebra.
**Algorithms:** RANSAC, Wahba's problem, chirality-filtered triangle matching,
two-pass plate-solving.
**Tooling:** Streamlit (interactive demo), Matplotlib (visualisations), SLURM (training).

---

## Limitations and honest caveats

- **CD-matrix shear comes from the TESS WCS.** In a true on-orbit star tracker the
  per-CCD distortion model (SIP + CD shear) would be factory-calibrated and burned
  into firmware. This pipeline currently reads it from the FITS header. The principle
  is the same — these are camera intrinsics, not attitude — but the cleanest
  demonstration would be a one-time per-CCD calibration step instead.
- **Best-fit residual floor ≈ 196″** when the SIP polynomial is applied with an
  identity CD matrix (truly attitude-free). This is what limits the median error to
  6.8″ rather than approaching arc-second. A factory CD-shear calibration would push
  this down further.
- **RANSAC latency varies from 17 s to ~12 min** per image on CPU. On a GPU and with
  pair-DB indexing this is much smaller; not yet optimised.
- **A single image of the 16 was honestly refused** by the quality gate (false-lock
  candidate; the gate did its job).

---

## Acknowledgements

NASA TESS mission for the public TICA Full-Frame-Image data. The Hipparcos catalog
team for the reference star positions used in catalog matching.

---

## Author

Master's thesis project, 2025–2026.
[Email](mailto:timka02qochqorov@gmail.com) · [LinkedIn](https://www.linkedin.com/in/temurkuchkorov/)

## License

MIT — see [LICENSE](LICENSE).
