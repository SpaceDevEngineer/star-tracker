"""
pipeline_app.py — Interactive walkthrough of the full lost-in-space star tracker
pipeline. Shows each stage as it happens with the actual numbers and figures.

Run:
    streamlit run Code/Streamlit_app/pipeline_app.py

Pipeline stages displayed:
  1. U-Net detection         — heatmap + extracted centroids
  2. Brightest 60 + body vec — top-N selection + SIP-aware unit vectors
  3. RANSAC triangle search  — the chosen triangle with its 3 angles
  4. Wahba rotation          — 3-point R + matrix display
  5. Pass-1 verification     — initial matches at wide tolerance
  6. Pass-2/3 plate-solve    — iterative refinement (3 inner iterations)
  7. Quality gate + result   — final attitude with GT comparison
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so it works on Streamlit Cloud,
# Codespaces, local clones, etc.)
# ---------------------------------------------------------------------------
PROJECT_DIR  = Path(__file__).resolve().parents[2]
MODEL_PATH   = PROJECT_DIR / "Results" / "unet_run3" / "best_model.pt"
CATALOG_CSV  = PROJECT_DIR / "Data" / "hybrid" / "catalog_hipparcos_full.csv"
TEST_DIR     = PROJECT_DIR / "Data" / "dataset_tess_test"
ARTEFACT_DIR = PROJECT_DIR / "Results" / "star_id_run"

sys.path.insert(0, str(PROJECT_DIR / "Code" / "Model_train_code"))
sys.path.insert(0, str(PROJECT_DIR / "Code" / "Star_ID"))

from train import UNet, extract_centroids
from triangle_id import StarIdentifier
from inference_full import (
    detect_unet, pixels_to_body_vecs, load_intrinsics_wcs,
    extract_camera_intrinsics, build_pose_wcs, plate_solve,
    refine_matches_by_projection, _R_from_pose, rot_to_quat,
    angular_error_arcsec,
)

TILE_SIZE = 512
ANGLE_TOL_ARCSEC      = 120.0
VERIFY_TOL_ARCSEC     = 200.0


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading U-Net model...")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = UNet(base_ch=32).to(device)
    m.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
    m.eval()
    return m, device


@st.cache_resource(show_spinner="Loading Hipparcos catalog + pattern DB (~5s first time)...")
def load_identifier():
    return StarIdentifier(str(CATALOG_CSV), mag_limit=7.5)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Star Tracker — Pipeline Walkthrough",
    page_icon="🌟", layout="wide",
)

st.title("🌟 Star Tracker Pipeline — Interactive Walkthrough")
st.caption("Lost-in-space attitude determination: PNG → quaternion. "
           "Each stage runs live with the actual numbers.")

# ---------------------------------------------------------------------------
# Sidebar — input + speed
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚡ Execution mode")
    has_artefacts = ARTEFACT_DIR.exists() and any(ARTEFACT_DIR.glob("*.json"))
    exec_default  = "Replay (instant)" if has_artefacts else "Live (slow, runs RANSAC)"
    exec_mode = st.radio(
        "How to run", ["Replay (instant)", "Live (slow, runs RANSAC)"],
        index=0 if exec_default == "Replay (instant)" else 1,
        help=("Replay loads pre-computed artefacts and renders every stage "
              "instantly. Live actually runs the pipeline (CPU: 1–10 min per "
              "image). The numbers are identical — Replay just skips the wait."),
    )
    is_replay = exec_mode.startswith("Replay")

    st.markdown("---")
    st.header("📂 Input")
    test_images = sorted((TEST_DIR / "images").glob("*.png")) if TEST_DIR.exists() else []
    mode = st.radio("Source", ["From test set", "Upload PNG"], index=0,
                     disabled=is_replay,
                     help="Upload is disabled in Replay mode — only "
                          "pre-computed images are available.")
    if is_replay:
        mode = "From test set"

    img_path = lbl_path = uploaded_img = uploaded_lbl = None
    if mode == "From test set":
        if test_images:
            choice = st.selectbox(
                f"Pick image ({len(test_images)} available)",
                test_images, format_func=lambda p: p.stem[-30:].replace("_tess_v01_img", ""),
                index=0,
            )
            img_path = choice
            lbl_path = TEST_DIR / "labels" / (choice.stem + ".json")
        else:
            st.warning("No test images found at " + str(TEST_DIR))
    else:
        uploaded_img = st.file_uploader("PNG image", type=["png"])
        uploaded_lbl = st.file_uploader("JSON label (needed for catalog WCS + GT)",
                                         type=["json"])

    st.markdown("---")
    st.header("⚙️ Settings")
    unet_thr = st.slider("U-Net detection threshold", 0.10, 0.95, 0.55, 0.05,
                          disabled=is_replay,
                          help="Used only in Live mode.")
    delay = st.slider("Animation delay between stages (s)", 0.0, 3.0, 0.6, 0.2,
                       help="Slow it down to watch each stage materialise.")

    st.markdown("---")
    run_btn = st.button("▶️ Run pipeline", type="primary",
                         use_container_width=True)


# ---------------------------------------------------------------------------
# Image + label load
# ---------------------------------------------------------------------------
img_np = label = None
if mode == "From test set" and img_path is not None and img_path.exists():
    img_np = np.array(Image.open(img_path), dtype=np.float32) / 255.0
    if lbl_path and lbl_path.exists():
        label = json.load(open(lbl_path))
elif mode == "Upload PNG" and uploaded_img is not None and uploaded_lbl is not None:
    img_np = np.array(Image.open(uploaded_img), dtype=np.float32) / 255.0
    label = json.load(uploaded_lbl)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def show_image_with_overlay(ax, img, vmin, vmax, title=""):
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11)


def fig_one(figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def pose_to_unit_vec(ra_deg, dec_deg):
    r = np.radians(ra_deg); d = np.radians(dec_deg)
    return np.array([np.cos(d)*np.cos(r), np.cos(d)*np.sin(r), np.sin(d)])


# ---------------------------------------------------------------------------
# Input preview
# ---------------------------------------------------------------------------
if img_np is None:
    st.info("👈 Pick a test image (or upload PNG + matching JSON) in the sidebar, then press **Run pipeline**.")
    st.markdown(
        "**Why a JSON?** The label JSON contains the FITS WCS (incl. SIP polynomial) — "
        "we need the camera intrinsics to convert pixels into body vectors."
    )
    st.stop()

vmin, vmax = np.percentile(img_np, [1, 99.7])
img_h, img_w = img_np.shape

st.subheader("📷 Input image")
c1, c2 = st.columns([3, 1])
with c1:
    fig, ax = plt.subplots(figsize=(8, 8))
    show_image_with_overlay(ax, img_np, vmin, vmax,
        f"{img_w}×{img_h}  •  TESS  ({img_path.name if img_path else 'uploaded'})")
    st.pyplot(fig, use_container_width=True)
with c2:
    st.metric("Image size", f"{img_w}×{img_h}")
    if label:
        st.metric("Plate scale", f"{label['pose']['plate_scale_arcsec_per_pix']:.2f}″/px")
        st.metric("Boresight RA", f"{label['pose']['boresight_ra_deg']:.3f}°")
        st.metric("Boresight Dec", f"{label['pose']['boresight_dec_deg']:.3f}°")
        st.metric("GT stars (in label)", len(label.get("stars", [])))

if not run_btn:
    st.stop()

# ---------------------------------------------------------------------------
# Pipeline execution — stage by stage
# ---------------------------------------------------------------------------
pose      = label["pose"]
ps_arcsec = pose["plate_scale_arcsec_per_pix"]
ps_rad    = np.radians(ps_arcsec / 3600.0)
wcs_full  = load_intrinsics_wcs(pose["wcs_header"])
intr_keys, cd_base, cd_base_roll = extract_camera_intrinsics(pose["wcs_header"])

# Heavy loaders happen only in Live mode.
model      = device = identifier = None
artefact   = None
if is_replay:
    art_path = ARTEFACT_DIR / f"{img_path.stem}.json" if img_path else None
    if art_path is None or not art_path.exists():
        st.error(f"Replay mode requires a pre-computed artefact at "
                  f"`{art_path}`, but it's missing. Pick another image or "
                  f"switch to Live mode.")
        st.stop()
    artefact = json.load(open(art_path))
    if artefact.get("failed"):
        st.warning("This image was rejected by the quality gate during the "
                   "original batch run — Replay will end at the gate stage.")
else:
    model, device = load_model()
    identifier    = load_identifier()

# ============================ STAGE 1 ===================================
st.markdown("## Stage 1 — U-Net detection")
st.markdown(
    "We slice the 2136×2078 PNG into 16 tiles of 512×512 px, run U-Net on each, "
    "extract local maxima from the heatmap above threshold, and reassemble back into image coords."
)

with st.status("🔁 Running U-Net on 16 tiles..." if not is_replay
               else "📼 Loading pre-computed detections...",
               expanded=True) as s1:
    if is_replay:
        det_xy = np.asarray(artefact["det_xy"])
        t_detect = float(artefact.get("time_detect_s", 0.0))
        det_b = None  # used only for flux sorting; in Replay we use det_xy_top directly
        st.write(f"**{len(det_xy)} centroids** loaded from cached run "
                  f"(original run-time: {t_detect:.1f}s on the laptop GPU).")
    else:
        t0 = time.time()
        det_xy, det_b = detect_unet(model, img_np, device, threshold=unet_thr)
        t_detect = time.time() - t0
        st.write(f"**{len(det_xy)} centroids** found in {t_detect:.1f}s "
                  f"(threshold={unet_thr:.2f})")

    fig, ax = plt.subplots(figsize=(9, 9))
    show_image_with_overlay(ax, img_np, vmin, vmax,
        f"Stage 1: {len(det_xy)} detections from U-Net")
    ax.scatter(det_xy[:, 0], det_xy[:, 1], facecolors="none", edgecolors="red",
                s=18, linewidths=0.6, label=f"U-Net detections ({len(det_xy)})")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    st.pyplot(fig, use_container_width=True)

    if is_replay:
        st.caption(
            "Live mode also shows the raw U-Net heatmap on the centre tile; "
            "Replay skips that one model forward pass for instant playback."
        )
    else:
        st.markdown("**What the U-Net actually outputs** (centre tile, raw heatmap):")
        row0 = (img_h - TILE_SIZE) // 2
        col0 = (img_w - TILE_SIZE) // 2
        tile = img_np[row0:row0+TILE_SIZE, col0:col0+TILE_SIZE]
        with torch.no_grad():
            heatmap = model(torch.tensor(tile[None, None]).to(device)).cpu().numpy().squeeze()
        centroids_local = np.asarray(extract_centroids(heatmap, threshold=unet_thr))

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6))
        axL.imshow(tile, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
        axL.set_title(f"Input tile ({TILE_SIZE}×{TILE_SIZE})")
        axL.set_xticks([]); axL.set_yticks([])
        axR.imshow(tile, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
        axR.imshow(heatmap, cmap="hot", alpha=0.55, vmin=0, vmax=1, origin="upper")
        if len(centroids_local) > 0:
            axR.scatter(centroids_local[:, 0], centroids_local[:, 1],
                        facecolors="none", edgecolors="cyan", s=22, linewidths=0.7,
                        label=f"{len(centroids_local)} peaks")
        axR.legend(loc="upper right", fontsize=9)
        axR.set_title("U-Net heatmap + extracted centroids")
        axR.set_xticks([]); axR.set_yticks([])
        st.pyplot(fig, use_container_width=True)
    s1.update(label=f"✅ Stage 1 — {len(det_xy)} detections", state="complete")

time.sleep(delay)

if len(det_xy) < 10:
    st.error("Too few detections — pipeline aborted.")
    st.stop()

# ============================ STAGE 2 ===================================
st.markdown("## Stage 2 — Top-60 brightest + body vectors")
st.markdown(
    "We sort detections by flux (sum of pixel values in a 7×7 window around the centroid), "
    "keep the **60 brightest** for the RANSAC search, and convert all detections to "
    "**body unit vectors** via SIP-corrected gnomonic projection."
)

with st.status("🔁 Computing body vectors...", expanded=True) as s2:
    if is_replay:
        det_xy_top = np.asarray(artefact["det_xy_top"])
        # Re-derive the top-N positional indices into det_xy for the table view.
        top_idx = []
        for tx, ty in det_xy_top:
            dist = np.hypot(det_xy[:, 0] - tx, det_xy[:, 1] - ty)
            top_idx.append(int(np.argmin(dist)))
        top_idx = np.asarray(top_idx)
        det_b = np.zeros(len(det_xy), dtype=np.float32)  # not used past this point in Replay
    else:
        bright_order = np.argsort(-det_b)
        top_idx      = bright_order[:60]
        det_xy_top   = det_xy[top_idx]
    body_vecs     = pixels_to_body_vecs(np.asarray(det_xy_top), wcs_full, ps_rad)
    body_vecs_all = pixels_to_body_vecs(np.asarray(det_xy),     wcs_full, ps_rad)
    st.write(f"Selected **top-{len(det_xy_top)}** brightest detections.")
    st.write(f"Body vectors are unit vectors in the camera frame, "
              f"corrected for TESS optical distortion (SIP polynomial of order 6).")

    fig, ax = plt.subplots(figsize=(9, 9))
    show_image_with_overlay(ax, img_np, vmin, vmax, "Stage 2: top-60 brightest")
    ax.scatter(det_xy[:, 0], det_xy[:, 1], facecolors="none", edgecolors="red",
                s=15, linewidths=0.3, alpha=0.5, label="all detections")
    ax.scatter(det_xy_top[:, 0], det_xy_top[:, 1], facecolors="none",
                edgecolors="yellow", s=40, linewidths=0.9, label="top-60 brightest")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    st.pyplot(fig, use_container_width=True)

    st.markdown(
        "Each row below is one detected star shown in **two coordinate systems**: "
        "its 2-D pixel position in the image, and its 3-D unit vector in the "
        "camera body frame (after SIP correction). The 3-D form is what the "
        "star-ID algorithm consumes."
    )
    sample = []
    for i in range(min(5, len(body_vecs))):
        sample.append({
            "★ brightness rank": i + 1,
            "detection id":      int(top_idx[i]),
            "pixel x":  f"{det_xy_top[i, 0]:.1f}",
            "pixel y":  f"{det_xy_top[i, 1]:.1f}",
            "body x":   f"{body_vecs[i, 0]:.5f}",
            "body y":   f"{body_vecs[i, 1]:.5f}",
            "body z":   f"{body_vecs[i, 2]:.5f}",
            "flux":     f"{det_b[top_idx[i]]:.1f}",
        })
    st.dataframe(pd.DataFrame(sample), use_container_width=True, hide_index=True)
    s2.update(label="✅ Stage 2 — 60 body vectors ready", state="complete")

time.sleep(delay)

# ============================ STAGE 3 ===================================
st.markdown("## Stage 3 — RANSAC triangle search")
st.markdown(
    "We need to find **3 detections** whose pairwise angles match **3 catalog stars**. "
    "RANSAC: pick 2 random detections, look up the catalog pattern DB for pairs at the same angular distance, "
    "then probe a 3rd detection and find the 3rd catalog star that closes the triangle (with matching chirality). "
    "Compute Wahba rotation, verify on all detections."
)

with st.status(
    "📼 Loading cached triangle from previous RANSAC run..." if is_replay
    else "🔁 RANSAC searching for a valid triangle (up to 1500 iterations)...",
    expanded=True,
) as s3:
    if is_replay:
        di, dj, dk, ci, cj, ck = artefact["triangle"]
        iterations = artefact.get("iterations", 0)
        n_matched_id = artefact.get("n_matched_id", 0)
        t_identify = float(artefact.get("time_identify_s", 0.0))

        st.write(f"**Cached triangle was found at iteration {iterations}** "
                  f"(original RANSAC took {t_identify:.1f}s).")
    else:
        t1 = time.time()
        id_result = identifier.identify(
            body_vecs, verify_body_vecs=body_vecs_all,
            tol_arcsec=ANGLE_TOL_ARCSEC,
            verify_tol_arcsec=VERIFY_TOL_ARCSEC,
            verbose=False,
        )
        t_identify = time.time() - t1
        if id_result is None:
            s3.update(label="❌ RANSAC failed — no valid triangle found", state="error")
            st.error("Pipeline aborted: no triangle ≥ 10 verified matches.")
            st.stop()

        di, dj, dk, ci, cj, ck = id_result.triangle
        iterations   = id_result.iterations
        n_matched_id = id_result.score
        st.write(f"**Found a valid triangle after {iterations} iterations** in {t_identify:.1f}s.")

    st.write(f"Detection vertices (indices into top-60): **A={di}**, **B={dj}**, **C={dk}**")
    if not is_replay:
        st.write(f"Catalog matches (Hipparcos IDs): "
                  f"**A→{identifier.db.star_ids[ci]}**, "
                  f"**B→{identifier.db.star_ids[cj]}**, "
                  f"**C→{identifier.db.star_ids[ck]}**")
    else:
        st.write(f"Catalog matches (pattern-DB indices): "
                  f"**A→{ci}**, **B→{cj}**, **C→{ck}**")
    st.write(f"Pass-1 verified matches at 200″ tolerance: **{n_matched_id}**")

    bi, bj, bk = body_vecs[di], body_vecs[dj], body_vecs[dk]
    θ_ij = np.degrees(np.arccos(np.clip(np.dot(bi, bj), -1, 1))) * 3600
    θ_ik = np.degrees(np.arccos(np.clip(np.dot(bi, bk), -1, 1))) * 3600
    θ_jk = np.degrees(np.arccos(np.clip(np.dot(bj, bk), -1, 1))) * 3600

    if is_replay:
        Θ_ij = Θ_ik = Θ_jk = None
    else:
        vi, vj, vk = identifier.db.star_vecs[ci], identifier.db.star_vecs[cj], identifier.db.star_vecs[ck]
        Θ_ij = np.degrees(np.arccos(np.clip(np.dot(vi, vj), -1, 1))) * 3600
        Θ_ik = np.degrees(np.arccos(np.clip(np.dot(vi, vk), -1, 1))) * 3600
    Θ_jk = np.degrees(np.arccos(np.clip(np.dot(vj, vk), -1, 1))) * 3600

    if is_replay:
        st.markdown("**Pairwise angles** — body-frame triangle from cached centroids:")
        angle_table = pd.DataFrame({
            "pair":          ["A–B", "A–C", "B–C"],
            "body (arcsec)": [f"{θ_ij:.2f}", f"{θ_ik:.2f}", f"{θ_jk:.2f}"],
        })
        st.dataframe(angle_table, use_container_width=True, hide_index=True)
        st.caption("Live mode also shows the catalog-side angles and their delta; "
                   "Replay omits them to avoid loading the 82 MB pattern DB.")
    else:
        Θ_jk = np.degrees(np.arccos(np.clip(np.dot(vj, vk), -1, 1))) * 3600
        st.markdown("**Pairwise angles** — detection triangle vs catalog triangle:")
        angle_table = pd.DataFrame({
            "pair":             ["A–B", "A–C", "B–C"],
            "body (arcsec)":    [f"{θ_ij:.2f}", f"{θ_ik:.2f}", f"{θ_jk:.2f}"],
            "catalog (arcsec)": [f"{Θ_ij:.2f}", f"{Θ_ik:.2f}", f"{Θ_jk:.2f}"],
            "Δ (arcsec)":       [f"{θ_ij-Θ_ij:+.2f}",
                                 f"{θ_ik-Θ_ik:+.2f}",
                                 f"{θ_jk-Θ_jk:+.2f}"],
        })
        st.dataframe(angle_table, use_container_width=True, hide_index=True)

    tri_pix = det_xy_top[[di, dj, dk]]
    fig, ax = plt.subplots(figsize=(9, 9))
    show_image_with_overlay(ax, img_np, vmin, vmax,
        f"Stage 3: RANSAC triangle (iteration {iterations})")
    ax.scatter(det_xy_top[:, 0], det_xy_top[:, 1], facecolors="none",
                edgecolors="yellow", s=25, linewidths=0.6, alpha=0.5)
    tri_poly = Polygon(tri_pix, closed=True, fill=False, edgecolor="lime",
                        linewidth=2.5)
    ax.add_patch(tri_poly)
    ax.scatter(tri_pix[:, 0], tri_pix[:, 1], c="lime", s=180, marker="*",
                edgecolors="black", linewidths=0.8, zorder=5)
    for idx_local, lbl_letter in zip([di, dj, dk], "ABC"):
        ax.annotate(lbl_letter, det_xy_top[idx_local], xytext=(10, -10),
                    textcoords="offset points", color="lime",
                    fontsize=14, fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    s3.update(label=f"✅ Stage 3 — triangle found in {iterations} iters", state="complete")

time.sleep(delay)

# ============================ STAGE 4 ===================================
st.markdown("## Stage 4 — Wahba SVD: first rotation")
st.markdown(
    "From the 3 body↔catalog pairs we solve **Wahba's problem** "
    "(find R minimizing ‖R·b – r‖²) via SVD. This is the **first rough rotation** — "
    "good enough to verify against the rest of the detections."
)

with st.status("🔁 Solving Wahba SVD on 3 pairs + verifying...", expanded=True) as s4:
    if is_replay:
        ra_pred, dec_pred, roll_pred = artefact["pose_pred"]
        R_pred = _R_from_pose(ra_pred, dec_pred, roll_pred)
        R_id = R_pred
        st.caption("Replay shows the final refined rotation matrix — the rough "
                   "3-point R is not separately saved in the artefact. The "
                   "numerical structure is identical.")
    else:
        R_id = id_result.R
    st.write("**Rotation matrix R (body → ICRS):**")
    st.code(np.array2string(R_id, formatter={'float': lambda x: f"{x:+.5f}"},
                              prefix="R = "), language="text")

    boresight = R_id @ np.array([1.0, 0, 0])
    ra_est  = np.degrees(np.arctan2(boresight[1], boresight[0])) % 360.0
    dec_est = np.degrees(np.arcsin(np.clip(boresight[2], -1, 1)))
    st.write(f"**Decoded approximate boresight**: RA ≈ {ra_est:.3f}°, Dec ≈ {dec_est:.3f}°")
    st.write(f"**Verified on all {len(body_vecs_all)} detections**: "
              f"**{n_matched_id} matches** within {VERIFY_TOL_ARCSEC:.0f}″ tolerance")

    s4.update(label=f"✅ Stage 4 — rough R found ({n_matched_id} initial matches)",
              state="complete")

time.sleep(delay)

# ============================ STAGE 5 ===================================
st.markdown("## Stage 5 — Pass 2/3: iterative plate-solve refinement")
st.markdown(
    "The rough R from Stage 4 has ~30 px residuals at the edges (SIP correction has limits when used without the true CD matrix). "
    "We **plate-solve**: scipy fits (RA, Dec, roll) to minimize pixel residuals between projected catalog and observed centroids. "
    "We iterate — project catalog, snap detections to nearest projection at progressively tighter tolerance (30→5→1.5 px), re-solve."
)

with st.status("🔁 Plate-solve refinement (3 inner iterations)..." if not is_replay
               else "📼 Loading cached plate-solve result...",
               expanded=True) as s5:
    if is_replay:
        pose_pred = tuple(artefact["pose_pred"])
        median_px_res = float(artefact.get("median_px_residual", float("nan")))
        n_matched = int(artefact.get("n_matched", len(artefact.get("matched_pairs", []))))
        history = [{
            "iter":          "cached final",
            "tol_px":        "1.5",
            "matches":       n_matched,
            "median_res_px": f"{median_px_res:.3f}",
            "RA":            f"{pose_pred[0]:.4f}",
            "Dec":           f"{pose_pred[1]:.4f}",
            "roll":          f"{pose_pred[2]:.4f}",
        }]
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
        st.caption("Live mode shows the four inner iterations of the plate-solve "
                   "loop tightening from 30 px to 1.5 px tolerance; the cached "
                   "artefact only stores the final converged result.")
    else:
        def matches_to_pairs(matches):
            di_list = list(matches.keys()); ci_list = list(matches.values())
            v = identifier.db.star_vecs[ci_list]
            dec = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
            ra  = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360
            return np.asarray(det_xy)[di_list], np.stack([ra, dec], axis=-1)

        R = R_id
        verify_tol_rad = np.radians(VERIFY_TOL_ARCSEC / 3600.0)
        all_matches = id_result.matches
        for _ in range(2):
            body_in_icrs = body_vecs_all @ R.T
            dots = body_in_icrs @ identifier.db.star_vecs.T
            cos_tol = np.cos(verify_tol_rad)
            best_cat = np.argmax(dots, axis=1)
            best_score = dots[np.arange(len(body_vecs_all)), best_cat]
            order = np.argsort(-best_score)
            all_matches = {}; used_cat = set()
            for d_idx in order:
                c_idx = int(best_cat[d_idx])
                if best_score[d_idx] >= cos_tol and c_idx not in used_cat:
                    all_matches[int(d_idx)] = c_idx
                    used_cat.add(c_idx)
            if len(all_matches) < 4:
                break
            body_p = [body_vecs_all[d_idx] for d_idx in all_matches.keys()]
            ref_p  = [identifier.db.star_vecs[c_idx] for c_idx in all_matches.values()]
            B = sum(np.outer(rr, bb) for bb, rr in zip(body_p, ref_p))
            U, _, Vt = np.linalg.svd(B)
            ds = np.linalg.det(U) * np.linalg.det(Vt)
            R = U @ np.diag([1., 1., ds]) @ Vt

        # Seed the plate-solver at the direction R sends CRPIX (not the image centre).
        crpix = np.array([[float(pose["wcs_header"]["CRPIX1"]),
                           float(pose["wcs_header"]["CRPIX2"])]])
        body_crpix = pixels_to_body_vecs(crpix, wcs_full, ps_rad)[0]
        icrs_crpix = R @ body_crpix
        ra_init  = float(np.degrees(np.arctan2(icrs_crpix[1], icrs_crpix[0]))) % 360.0
        dec_init = float(np.degrees(np.arcsin(np.clip(icrs_crpix[2], -1, 1))))
        pose_pred = (ra_init, dec_init, cd_base_roll)

        pix_in, rdc_in = matches_to_pairs(all_matches)
        history = []
        if len(pix_in) >= 4:
            ra, dec, roll, cost, final_res = plate_solve(
                pix_in, rdc_in, intr_keys, cd_base, cd_base_roll, pose_pred)
            pose_pred = (ra, dec, roll)
            history.append({"iter": "init", "tol_px": "—", "matches": len(all_matches),
                            "median_res_px": f"{float(np.median(np.abs(final_res))):.3f}",
                            "RA": f"{ra:.4f}", "Dec": f"{dec:.4f}", "roll": f"{roll:.4f}"})

            for tol_pix in (30.0, 5.0, 1.5):
                w_cand = build_pose_wcs(*pose_pred, intr_keys, cd_base, cd_base_roll)
                new_matches = refine_matches_by_projection(
                    w_cand, det_xy, identifier, img_w, img_h, tol_pix=tol_pix)
                if len(new_matches) < 4:
                    break
                pix_in, rdc_in = matches_to_pairs(new_matches)
                ra, dec, roll, cost, final_res = plate_solve(
                    pix_in, rdc_in, intr_keys, cd_base, cd_base_roll, pose_pred,
                    bounds_deg=0.5, roll_bounds_deg=5.0)
                pose_pred = (ra, dec, roll)
                all_matches = new_matches
                history.append({"iter": f"tol={tol_pix} px",
                                "tol_px": f"{tol_pix:.1f}",
                                "matches": len(all_matches),
                                "median_res_px": f"{float(np.median(np.abs(final_res))):.3f}",
                                "RA": f"{ra:.4f}", "Dec": f"{dec:.4f}", "roll": f"{roll:.4f}"})

        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
        median_px_res = float(np.median(np.abs(final_res)))
        n_matched = len(all_matches)
    s5.update(label=f"✅ Stage 5 — plate-solved ({n_matched} matches, "
                    f"{median_px_res:.3f} px residual)", state="complete")

time.sleep(delay)

# ============================ STAGE 6 ===================================
st.markdown("## Stage 6 — Quality gate")
quality_ok = median_px_res < 2.0 and n_matched >= 8
if quality_ok:
    st.success(f"✅ **PASS** — median residual {median_px_res:.3f} px < 2 px, "
                f"{n_matched} matches ≥ 8.")
else:
    st.error(f"❌ **FAIL** — residual {median_px_res:.2f} px or only {n_matched} matches. "
              f"Would mark this image as a 'no lock' rather than emit a wrong attitude.")

time.sleep(delay)

# ============================ STAGE 7 ===================================
st.markdown("## Stage 7 — Final attitude + GT comparison")

ra_gt   = float(pose["wcs_header"]["CRVAL1"])
dec_gt  = float(pose["wcs_header"]["CRVAL2"])
roll_gt = cd_base_roll
R_gt    = _R_from_pose(ra_gt, dec_gt, roll_gt)
R_pred  = _R_from_pose(*pose_pred)
q_gt    = rot_to_quat(R_gt);   q_gt   /= np.linalg.norm(q_gt)
q_pred  = rot_to_quat(R_pred); q_pred /= np.linalg.norm(q_pred)
err     = angular_error_arcsec(q_pred, q_gt)

c1, c2 = st.columns(2)
with c1:
    st.metric("Angular error", f"{err:.2f}″", f"{err/ps_arcsec:.3f} px")
    st.metric("Verified matches", n_matched)
with c2:
    st.metric("Median pixel residual", f"{median_px_res:.3f} px")
    st.metric("Pipeline time",
              f"{t_detect + t_identify:.1f}s",
              help="U-Net detect + RANSAC + plate-solve (original run)")

st.markdown("**Pose comparison (predicted vs ground-truth):**")
st.dataframe(pd.DataFrame({
    "parameter":      ["RA (deg)", "Dec (deg)", "Roll (deg)"],
    "predicted":      [f"{pose_pred[0]:.5f}", f"{pose_pred[1]:.5f}", f"{pose_pred[2]:.5f}"],
    "GT (CRVAL/CD)":  [f"{ra_gt:.5f}",         f"{dec_gt:.5f}",       f"{roll_gt:.5f}"],
    "Δ":              [f"{pose_pred[0]-ra_gt:+.5f}",
                       f"{pose_pred[1]-dec_gt:+.5f}",
                       f"{pose_pred[2]-roll_gt:+.5f}"],
}), use_container_width=True, hide_index=True)

st.markdown("**Final overlay**: red = U-Net detections, cyan = projected catalog "
            "via predicted attitude, lime lines = matched pairs.")
wcs_pred = build_pose_wcs(*pose_pred, intr_keys, cd_base, cd_base_roll)

if is_replay:
    mp = artefact.get("matched_pairs", [])
    cat_ra  = np.array([m["ra_deg"]  for m in mp])
    cat_dec = np.array([m["dec_deg"] for m in mp])
    mp_px   = np.array([(m["px"], m["py"]) for m in mp])
else:
    cat_idx_list = list(all_matches.values())
    cat_v = identifier.db.star_vecs[cat_idx_list]
    cat_ra  = np.degrees(np.arctan2(cat_v[:, 1], cat_v[:, 0])) % 360
    cat_dec = np.degrees(np.arcsin(np.clip(cat_v[:, 2], -1, 1)))
    mp_px   = np.asarray(det_xy)[list(all_matches.keys())]
px_proj, py_proj = wcs_pred.all_world2pix(cat_ra, cat_dec, 0)

fig, ax = plt.subplots(figsize=(10, 10))
show_image_with_overlay(ax, img_np, vmin, vmax,
    f"Final result — angular error {err:.2f}″ ({err/ps_arcsec:.2f} px)")
ax.scatter(det_xy[:, 0], det_xy[:, 1], facecolors="none", edgecolors="red",
            s=15, linewidths=0.4, alpha=0.5)
for (mx, my), (pxp, pyp) in zip(mp_px, zip(px_proj, py_proj)):
    ax.plot([mx, pxp], [my, pyp], "-", c="lime", linewidth=0.7)
ax.scatter(mp_px[:, 0], mp_px[:, 1], facecolors="none", edgecolors="red",
            s=22, linewidths=0.8, label="matched detection")
ax.scatter(px_proj, py_proj, marker="+", c="cyan", s=70, linewidths=1.2,
            label="projected catalog")
ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
st.pyplot(fig, use_container_width=True)

st.success(f"🎯 Pipeline complete. Lost-in-space attitude determined to "
            f"**{err:.2f}″** ({err/ps_arcsec:.2f} px) on a {img_w}×{img_h} TESS image.")
