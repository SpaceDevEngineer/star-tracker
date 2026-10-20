# Deep-Learning Star Tracker: Spacecraft Attitude from Real Satellite Imagery

*Master's thesis project, 2025–2026*

---

## One-line summary (CV entry)

Built a lost-in-space star tracker that recovers a spacecraft's 3-axis attitude from a
single image of real NASA TESS data. It solved **248 of 248 frames with zero wrong
answers**, at a median error of **7.8″ (0.38 px)**, and kept the same accuracy on 8
unseen sky sectors. The design pairs a U-Net star detector with RANSAC triangle
identification, Wahba's SVD solver and nonlinear plate solving.

## Short portfolio description

A star tracker finds a spacecraft's orientation by photographing the stars and
recognising them. I built the complete pipeline: a U-Net that detects stars with
sub-pixel precision, triangle-pattern matching against the Hipparcos catalogue with
RANSAC, attitude from Wahba's problem, plate-solve refinement, and a quality gate that
refuses instead of guessing. On 248 real images taken in orbit by NASA's TESS
satellite, every frame was solved with zero wrong answers, at a median error of about
a third of a pixel. Along the way I diagnosed a rare failure mode in the dense Milky
Way and fixed it with a statistical stopping rule, and made the identification step
27× faster with bit-identical results.

---

## Key contributions

- **End-to-end pipeline on real data.** TESS FITS → background-subtracted PNG and labels
  → U-Net heatmap detection → SIP-corrected body vectors → RANSAC triangle
  identification → Wahba SVD → plate solve → quality gate → refusal-retry → quaternion.
- **Learned detection that measurably helps.** With known star matches, the U-Net
  improves attitude error over classical peak finding from 6.0″ to 4.6″ (23%), through
  centroid precision. HRNet comes within 0.2″ with half the parameters.
- **Root-caused a real-world optics failure.** TESS's 12° field needs a 6th-order SIP
  distortion model. A pinhole model left 200–1000″ residuals at the corners and made
  blind identification fail on every frame. Rebuilt the geometry around SIP-corrected
  vectors.
- **Diagnosed and fixed the only remaining failure mode.** 7 of 248 frames locked onto
  a chance pattern in Carina, the densest bright-star field in the sky. The quality
  gate refused all 7, so none were wrong. Showed that raw match counts overlap between
  true and false locks but significance (matches / matches expected by chance) does
  not (≥ 7.4× vs ≤ 3.6×), and added a significance-aware retry that recovers all 7.
- **Performance engineering.** Profiled the pipeline, found that the third-star
  catalogue search dominated, and replaced it with precomputed neighbour lists: the
  median frame time went from 211 s to 9.4 s, with identical results on all 248 frames.
- **Honest evaluation.** Targets fixed before the final runs (≥ 95% solved, < 1 px, 0 false locks), a
  generalisation test on 8 unseen sectors, Wilson confidence intervals, an ablation of
  the textbook Pyramid check (inert on clean data), and per-frame result tables in the
  repository.
- **Reproducible software.** Unit tests that tie the scored attitude error to WCS
  geometry, result-table contract tests, Streamlit replay tests, GitHub Actions CI, and
  an interactive demo of every pipeline stage.

## Results

| | Test split | Unseen sectors |
|---|---:|---:|
| Solved | 120 / 120 | 128 / 128 |
| False locks | 0 | 0 |
| Median error | 7.79″ (0.38 px) | 7.33″ (0.36 px) |
| 90th percentile / worst | 13.87″ / 20.39″ | 15.02″ / 25.35″ |
| Below one pixel | 120 / 120 | 125 / 128 |
| Median pointing / roll | 2.78″ / 7.10″ | 2.61″ / 6.50″ |

Typical frame time: 9.4 s (U-Net on GPU, star identification on CPU).

---

## Detailed technical description

### Problem

A star tracker determines attitude by matching imaged stars against a catalogue. The
*lost-in-space* case, with no prior pointing knowledge, is the hardest. Most published
learning-based trackers are evaluated on simulated star fields and report detection
metrics. This project evaluates the final attitude on real images taken in orbit, with
real noise, stray light, saturated stars and lens distortion.

### Approach

1. **Data.** About 800 TESS TICA full-frame images from sectors 27–101 were converted
   into background-subtracted 8-bit PNGs with JSON labels: star positions matched to
   Hipparcos and the full WCS solution including SIP distortion. The split is 70/15/15
   by image, and 8 further sectors are held out as an unseen-sky test.
2. **Star detection.** A U-Net (7.76 M parameters) regresses a Gaussian heatmap per
   star on 512×512 tiles, trained with weighted MSE (star pixels × 20) and selected by
   validation F1 (0.636 at threshold 0.55). Centroids come from intensity-weighted
   connected components.
3. **Camera geometry.** Pixel centroids are mapped to body unit vectors with the SIP
   distortion removed (`astropy.wcs.sip_pix2foc`) and a gnomonic projection. Only camera
   calibration (SIP, CRPIX and the attitude-free scale/shear part of the CD matrix) is
   read from the label. The attitude, including roll, comes from the blind solution.
4. **Star identification.** RANSAC over triangles of the 60 brightest detections,
   matched against a Hipparcos pair-angle database (V < 7.5) with a chirality check. A
   3-star Wahba solution is verified against all detections, and the third catalogue
   star is searched only among precomputed neighbours.
5. **Refinement.** Wahba refit on every verified star, then nonlinear least squares over
   (RA, Dec, roll) in pixel space through the SIP forward model, with match tolerances
   tightened from 30 px to 5 px to 1.5 px.
6. **Reliability.** A quality gate refuses any solution whose median post-fit residual
   exceeds 2 px. After a refusal, the search is repeated with the refused sky region
   excluded and early stopping allowed only for candidates at least 5× above the
   chance-match level of their sky region.

### Lessons

- Detection F1 is a misleading proxy: the classical detector scores F1 = 0.10 yet gives
  usable attitudes, because its "false positives" are mostly real faint stars. The
  end-to-end attitude error is the honest metric.
- On real data the hard cases are statistical (dense sky regions), not algorithmic.
  Comparing against chance is what separates true from false locks.
- Roll about the boresight is the weakly observable axis for a single frame: about 7″,
  against about 3″ for pointing.

---

## Technologies

**ML:** Python, PyTorch (U-Net, HRNet, heatmap regression), NumPy, SciPy.
**Astronomy / aerospace:** Astropy (FITS, WCS, SIP), Hipparcos, ICRS frames,
quaternions, attitude determination, TESS mission data.
**Algorithms:** RANSAC, Wahba's problem / SVD, gnomonic projection, plate solving,
nonlinear least squares, significance testing.
**Engineering:** pytest, GitHub Actions, Streamlit, Matplotlib, SLURM GPU cluster
(RTX 2080 Ti), profiling and performance optimisation, Git, Linux.

## Skills demonstrated

- End-to-end ML system design: data pipeline, model, classical post-processing,
  evaluation
- Deep learning for precise localisation (sub-pixel heatmap regression)
- Classical computer vision and numerical methods: RANSAC, SVD, nonlinear optimisation
- Root-cause analysis of failures in a multi-stage system on real, noisy data
- Performance engineering with verified output equivalence
- Rigorous, honest evaluation: generalisation tests, ablations, confidence intervals
- Scientific software engineering: tests, CI, reproducible result tables, interactive demo
