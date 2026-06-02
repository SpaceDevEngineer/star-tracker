"""
process_tess.py — TESS FFI → Star Tracker training dataset pipeline.

Pipeline per FITS file:
    1. Load TESS FFI (robust to data living in hdul[0] or hdul[1]).
    2. Normalize image (5–95 percentile clipping + log scaling).
    3. Save normalized image as PNG to dataset/images/.
    4. Obtain WCS:
        - --wcs-source header  → use WCS baked into the FITS header (fast; TESS TICA FFIs ship with SIP WCS).
        - --wcs-source astrometry → upload PNG to Astrometry.net via astroquery and wait for a solve.
    5. Match Hipparcos catalog stars that fall inside the image FOV, project (RA, Dec) → (x, y).
    6. Save JSON label file to dataset/labels/ with one entry per star.

Usage:
    python process_tess.py \
        --fits-dir /Users/timon/Desktop/Thesis/Data/Image_TESS/cam1-ccd1 \
        --catalog  /Users/timon/Desktop/Thesis/Data/hybrid/catalog_hipparcos.csv \
        --out-dir  /Users/timon/Desktop/Thesis/Code/dataset \
        --wcs-source header

    # Or with Astrometry.net (requires ASTROMETRY_API_KEY env var or --api-key):
    python process_tess.py --wcs-source astrometry --api-key XXXXXXXXXXXX ...

Notes:
    - The provided Hipparcos CSV has columns (ra_deg, dec_deg, mag) but no HIP identifier,
      so the 0-based CSV row index is used as `hipparcos_id` (deterministic within this catalog).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from PIL import Image

warnings.filterwarnings("ignore", category=FITSFixedWarning)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("process_tess")
# Separate progress logger — always prints to stdout cleanly
_progress = logging.getLogger("progress")
_progress.setLevel(logging.INFO)
if not _progress.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    _progress.addHandler(_h)
    _progress.propagate = False


# ---------------------------------------------------------------------------
# 1. FITS loading
# ---------------------------------------------------------------------------
def load_fits_image(fits_path: Path) -> Tuple[np.ndarray, fits.Header]:
    """Return (image_2d, header) from a TESS FFI.

    The user's brief says data usually lives in ``hdul[1].data``. For TESS TICA
    FFIs it actually lives in ``hdul[0].data`` and ``hdul[1]`` is a binary
    table. We therefore pick the first HDU whose data is a 2D float array.
    """
    with fits.open(fits_path, memmap=False) as hdul:
        for i, hdu in enumerate(hdul):
            data = hdu.data
            if data is not None and getattr(data, "ndim", 0) == 2:
                log.debug("  FITS image read from HDU[%d], shape=%s, dtype=%s",
                         i, data.shape, data.dtype)
                return np.asarray(data, dtype=np.float32), hdu.header
    raise ValueError(f"No 2D image HDU found in {fits_path.name}")


# ---------------------------------------------------------------------------
# 2. Normalization
# ---------------------------------------------------------------------------
def _subtract_background(image: np.ndarray) -> np.ndarray:
    """Fast large-scale background estimation via downsample + Gaussian smooth + upsample.

    Replaces photutils Background2D and scipy median_filter, both of which are
    extremely slow on 2078×2136 images. This approach runs in < 0.5s on any CPU:
      1. Downsample image to ~128px on the short axis
      2. Gaussian-smooth the tiny image (captures large-scale scattered light)
      3. Upsample back to original size and subtract
    """
    from scipy.ndimage import gaussian_filter, zoom

    factor = max(1, min(image.shape) // 128)
    small  = zoom(image, 1.0 / factor, order=1, prefilter=False)
    bg_small = gaussian_filter(small.astype(np.float64), sigma=5)
    bg = zoom(bg_small, factor, order=1, prefilter=False)
    bg = bg[:image.shape[0], :image.shape[1]]
    return (image - bg).astype(np.float32)


def normalize_image(
    image: np.ndarray,
    mode: str = "startracker",
    p_low: float = 50.0,
    p_high: float = 99.7,
) -> np.ndarray:
    """Rescale a TESS FFI to [0, 1] float32 for display / plate-solving.

    Star fields have ~99% background and ~1% stars, so classic 5–95 percentile
    clipping saturates the sky and makes stars invisible. This function offers
    several presets:

    mode:
        "startracker" — (default) subtract 2D scattered-light background,
                        hard-clip below sky and above p_high, linear stretch.
                        Gives the classic "black sky / bright stars" look that
                        is ideal for star-tracker training and for
                        Astrometry.net source extraction.
        "zscale"      — ZScaleInterval + AsinhStretch (DS9/IRAF default).
                        Great for science display; too grey for star trackers.
        "log"         — aggressive percentile clip (p_low..p_high) + log1p.
                        Kept for reproducibility.
        "zscore"      — standard score then min–max.
    """
    img = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if mode == "startracker":
        sub = _subtract_background(img)
        # After background subtraction the sky scatters around 0. Stars are
        # positive outliers. Clip bottom at a few sigma above zero to force a
        # truly black sky, top at p99.7 to avoid a single bright pixel
        # dominating the range.
        lo = float(np.percentile(sub, p_low))
        hi = float(np.percentile(sub, p_high))
        if hi <= lo:
            hi = lo + 1.0
        out = np.clip((sub - lo) / (hi - lo), 0.0, 1.0)
        return out.astype(np.float32)

    if mode == "zscale":
        from astropy.visualization import (
            AsinhStretch,
            ImageNormalize,
            ZScaleInterval,
        )
        norm = ImageNormalize(
            img,
            interval=ZScaleInterval(contrast=0.15),
            stretch=AsinhStretch(a=0.1),
            clip=True,
        )
        out = norm(img)
        out = np.ma.filled(out, 0.0).astype(np.float32)
        return np.clip(out, 0.0, 1.0)

    lo, hi = np.percentile(img, (p_low, p_high))
    if hi <= lo:
        hi = lo + 1.0
    img = np.clip(img, lo, hi)

    if mode == "log":
        img = img - lo + 1.0
        img = np.log1p(img)
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    elif mode == "zscore":
        img = (img - img.mean()) / (img.std() + 1e-12)
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    else:
        raise ValueError(f"unknown normalization mode: {mode}")

    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. PNG save
# ---------------------------------------------------------------------------
def save_png(image_01: np.ndarray, png_path: Path) -> None:
    """Save a [0, 1] float image as an 8-bit grayscale PNG."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image_01 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(png_path, format="PNG")
    log.debug("  PNG saved → %s", png_path)


# ---------------------------------------------------------------------------
# 4a. WCS from FITS header (fast path — TESS TICA ships SIP WCS)
# ---------------------------------------------------------------------------
def wcs_from_header(header: fits.Header) -> Optional[WCS]:
    try:
        w = WCS(header)
        if w.has_celestial:
            log.debug("  WCS obtained from FITS header (CTYPE1=%s)", header.get("CTYPE1"))
            return w
    except Exception as exc:
        log.warning("  WCS(header) failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# 4b. WCS from Astrometry.net
# ---------------------------------------------------------------------------
def wcs_from_astrometry_net(
    png_path: Path,
    api_key: str,
    header_hint: Optional[fits.Header] = None,
    timeout_s: int = 300,
    retries: int = 1,
) -> Optional[WCS]:
    """Upload PNG to Astrometry.net and wait for a plate-solve.

    `header_hint` (FITS header) lets us pass SC_RA/SC_DEC as a search hint,
    which dramatically reduces solve time. Retries the call once on transient
    network errors (RemoteDisconnected, timeouts).
    """
    from astroquery.astrometry_net import AstrometryNet

    ast = AstrometryNet()
    ast.api_key = api_key

    kw = dict(
        publicly_visible="n",
        allow_commercial_use="n",
        allow_modifications="n",
        solve_timeout=timeout_s,
    )
    if header_hint is not None:
        ra_hint = header_hint.get("SC_RA") or header_hint.get("CRVAL1")
        dec_hint = header_hint.get("SC_DEC") or header_hint.get("CRVAL2")
        if ra_hint is not None and dec_hint is not None:
            kw.update(
                center_ra=float(ra_hint),
                center_dec=float(dec_hint),
                radius=15.0,                     # TESS camera FOV ≈ 24°; 15° is a tight, safe hint
                scale_units="degwidth",
                scale_type="ul",
                scale_lower=10.0,
                scale_upper=30.0,
            )

    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 2):
        log.debug("  Uploading to Astrometry.net (attempt %d/%d, timeout %ds)…",
                 attempt, retries + 1, timeout_s)
        t0 = time.time()
        try:
            wcs_header = ast.solve_from_image(str(png_path), **kw)
            if wcs_header:
                log.debug("  Astrometry.net solved in %.0fs", time.time() - t0)
                return WCS(wcs_header)
            log.warning("  empty WCS returned after %.0fs", time.time() - t0)
        except Exception as exc:
            last_exc = exc
            log.warning("  attempt %d failed after %.0fs: %s",
                        attempt, time.time() - t0, exc)

    log.error("  Astrometry.net solve gave up: %s", last_exc)
    return None


# ---------------------------------------------------------------------------
# 5. Label generation from Hipparcos catalog
# ---------------------------------------------------------------------------
def compute_pose(wcs: WCS, image_shape: Tuple[int, int]) -> dict:
    """Derive star-tracker ground-truth pose from a WCS solution.

    Returns:
        boresight_ra_deg, boresight_dec_deg : direction the optical axis points to
        roll_deg                            : rotation of the image about the boresight
        quaternion_xyzw                     : ECI/ICRS attitude as a unit quaternion
        plate_scale_arcsec_per_pix          : average pixel scale
        fov_deg                             : (width, height) field of view
    """
    h, w = image_shape

    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    ra_arr, dec_arr = wcs.wcs_pix2world([cx], [cy], 0)
    ra_c, dec_c = float(np.asarray(ra_arr).ravel()[0]), float(np.asarray(dec_arr).ravel()[0])

    cd = wcs.pixel_scale_matrix          # 2x2, deg/pixel
    roll_rad = float(np.arctan2(cd[0, 1], cd[0, 0]))
    roll_deg = float(np.rad2deg(roll_rad))

    px_scale_deg = float(np.sqrt(np.abs(np.linalg.det(cd))))
    plate_scale_arcsec = px_scale_deg * 3600.0
    fov_w = w * px_scale_deg
    fov_h = h * px_scale_deg

    # ICRS attitude quaternion from (RA, Dec, roll), Z-Y-X intrinsic convention:
    #   R = Rz(RA) · Ry(-Dec) · Rx(roll)
    a = np.deg2rad(ra_c) / 2.0
    d = np.deg2rad(-dec_c) / 2.0
    r = np.deg2rad(roll_deg) / 2.0
    qz = np.array([0, 0, np.sin(a), np.cos(a)])
    qy = np.array([0, np.sin(d), 0, np.cos(d)])
    qx = np.array([np.sin(r), 0, 0, np.cos(r)])

    def qmul(p, q):
        x1, y1, z1, w1 = p
        x2, y2, z2, w2 = q
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ])

    q = qmul(qmul(qz, qy), qx)
    q /= np.linalg.norm(q)

    # Serialize the WCS (incl. SIP polynomial) so downstream inference can
    # apply the correct nonlinear pixel<->sky mapping. Only the keys astropy
    # needs to reconstruct a celestial WCS are kept; everything else (DATE-OBS,
    # SC_*, etc.) lives in `spacecraft`/elsewhere.
    wcs_hdr = wcs.to_header(relax=True)  # relax=True preserves SIP A_*/B_*
    wcs_dict = {}
    for k in wcs_hdr.keys():
        v = wcs_hdr[k]
        if isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        wcs_dict[k] = v

    return {
        "boresight_ra_deg": ra_c,
        "boresight_dec_deg": dec_c,
        "roll_deg": roll_deg,
        "quaternion_xyzw": [float(v) for v in q],
        "plate_scale_arcsec_per_pix": plate_scale_arcsec,
        "fov_deg": [fov_w, fov_h],
        "wcs_header": wcs_dict,
    }


def _angular_prefilter(
    ra: np.ndarray, dec: np.ndarray, center_ra: float, center_dec: float, radius_deg: float
) -> np.ndarray:
    """Boolean mask of points within `radius_deg` of (center_ra, center_dec)."""
    ra_r = np.deg2rad(ra)
    dec_r = np.deg2rad(dec)
    c_ra_r = np.deg2rad(center_ra)
    c_dec_r = np.deg2rad(center_dec)
    cos_d = (np.sin(dec_r) * np.sin(c_dec_r)
             + np.cos(dec_r) * np.cos(c_dec_r) * np.cos(ra_r - c_ra_r))
    cos_d = np.clip(cos_d, -1.0, 1.0)
    return np.arccos(cos_d) <= np.deg2rad(radius_deg)


def generate_labels(
    wcs: WCS,
    image_shape: Tuple[int, int],
    catalog: pd.DataFrame,
    mag_limit: Optional[float] = None,
) -> list[dict]:
    """Project catalog stars into pixel space and keep the ones inside the FOV.

    Returns a list of dicts: {x, y, magnitude, hipparcos_id}.
    """
    h, w = image_shape

    ra = catalog["ra_deg"].to_numpy(dtype=np.float64)
    dec = catalog["dec_deg"].to_numpy(dtype=np.float64)
    mag = catalog["mag"].to_numpy(dtype=np.float64)
    hip = catalog.get("hipparcos_id", pd.Series(catalog.index, name="hipparcos_id")).to_numpy()

    if mag_limit is not None:
        keep = mag <= mag_limit
        ra, dec, mag, hip = ra[keep], dec[keep], mag[keep], hip[keep]

    # Coarse angular pre-filter: drop stars far from the plate center before the
    # iterative SIP inverse transform (which otherwise diverges for distant points).
    try:
        (center_ra,), (center_dec,) = wcs.wcs_pix2world([w / 2.0], [h / 2.0], 0)
        # Half-diagonal of the plate in degrees, with a generous 1.5× safety factor.
        corners_x = np.array([0, w, 0, w], dtype=np.float64)
        corners_y = np.array([0, 0, h, h], dtype=np.float64)
        c_ra, c_dec = wcs.wcs_pix2world(corners_x, corners_y, 0)
        radius_deg = 1.5 * np.max(
            np.rad2deg(np.arccos(np.clip(
                np.sin(np.deg2rad(c_dec)) * np.sin(np.deg2rad(center_dec))
                + np.cos(np.deg2rad(c_dec)) * np.cos(np.deg2rad(center_dec))
                * np.cos(np.deg2rad(c_ra - center_ra)),
                -1.0, 1.0,
            )))
        )
        mask = _angular_prefilter(ra, dec, float(center_ra), float(center_dec), float(radius_deg))
        ra, dec, mag, hip = ra[mask], dec[mask], mag[mask], hip[mask]
        log.debug("  Pre-filter around (%.3f, %.3f)° r=%.2f°: %d candidates",
                 float(center_ra), float(center_dec), float(radius_deg), len(ra))
    except Exception as exc:
        log.warning("  angular pre-filter skipped: %s", exc)

    if len(ra) == 0:
        return []

    try:
        x, y = wcs.all_world2pix(ra, dec, 0, quiet=True)
    except Exception:
        # Fall back to linear (no SIP) transform if the iterative solver still fails.
        x, y = wcs.wcs_world2pix(ra, dec, 0)

    margin = 0
    inside = (
        np.isfinite(x) & np.isfinite(y)
        & (x >= margin) & (x < w - margin)
        & (y >= margin) & (y < h - margin)
    )
    x, y, mag, hip = x[inside], y[inside], mag[inside], hip[inside]

    log.debug("  Catalog stars projected: %d inside FOV", len(x))

    labels = [
        {
            "x": float(px),
            "y": float(py),
            "magnitude": float(m),
            "hipparcos_id": int(hid) if np.issubdtype(type(hid), np.integer) else str(hid),
        }
        for px, py, m, hid in zip(x, y, mag, hip)
    ]
    return labels


# ---------------------------------------------------------------------------
# 6. Per-file driver
# ---------------------------------------------------------------------------
def process_one(
    fits_path: Path,
    out_images: Path,
    out_labels: Path,
    catalog: pd.DataFrame,
    wcs_source: str,
    api_key: Optional[str],
    mag_limit: Optional[float],
    norm_mode: str,
    overview_dir: Optional[Path] = None,
) -> bool:
    stem = fits_path.stem
    log.debug("[%s]", stem)

    try:
        image, header = load_fits_image(fits_path)
    except Exception as exc:
        log.error("  load failed: %s", exc)
        return False

    norm = normalize_image(image, mode=norm_mode)
    png_path = out_images / f"{stem}.png"
    save_png(norm, png_path)

    if wcs_source == "header":
        wcs = wcs_from_header(header)
    elif wcs_source == "astrometry":
        if not api_key:
            raise SystemExit("Astrometry.net requires --api-key or $ASTROMETRY_API_KEY")
        wcs = wcs_from_astrometry_net(png_path, api_key, header_hint=header)
    else:
        raise ValueError(f"unknown wcs-source: {wcs_source}")

    if wcs is None:
        log.error("  no WCS available; skipping label generation for %s", stem)
        return False

    labels = generate_labels(wcs, image.shape, catalog, mag_limit=mag_limit)

    label_path = out_labels / f"{stem}.json"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    pose = compute_pose(wcs, image.shape)

    spacecraft = {}
    for k_fits, k_out in (("SC_RA", "sc_ra_deg"), ("SC_DEC", "sc_dec_deg"),
                          ("SC_ROLL", "sc_roll_deg"),
                          ("SC_QUATX", "sc_qx"), ("SC_QUATY", "sc_qy"),
                          ("SC_QUATZ", "sc_qz"), ("SC_QUATQ", "sc_qw"),
                          ("MIDTJD", "mid_tjd"), ("CADENCE", "cadence"),
                          ("CAMNUM", "camera"), ("CCDNUM", "ccd")):
        v = header.get(k_fits)
        if v is not None:
            spacecraft[k_out] = float(v) if isinstance(v, (int, float)) else v

    payload = {
        "image": f"images/{png_path.name}",
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "wcs_source": wcs_source,
        "num_stars": len(labels),
        "mag_limit": mag_limit,
        "pose": pose,
        "spacecraft": spacecraft,
        "stars": labels,
    }
    with open(label_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.debug("  labels saved → %s  (%d stars)", label_path, len(labels))

    if overview_dir is not None and labels:
        overview_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(norm, cmap="gray", origin="lower")
        xs = [s["x"] for s in labels]
        ys = [s["y"] for s in labels]
        ax.scatter(xs, ys, s=8, facecolors="none", edgecolors="red", linewidths=0.5)
        ax.set_title(f"{stem}  ({len(labels)} stars)")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(overview_dir / f"{stem}_overlay.png", dpi=120)
        plt.close(fig)

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="TESS FFI → Star Tracker dataset pipeline")
    p.add_argument("--fits-dir", type=Path,
                   default=Path("/Users/timon/Desktop/Thesis/Data/Image_TESS_demo"))
    p.add_argument("--catalog",  type=Path,
                   default=Path("/Users/timon/Desktop/Thesis/Data/hybrid/catalog_hipparcos_full.csv"),
                   help="Hipparcos CSV; must have ra_deg, dec_deg, mag (and ideally hipparcos_id).")
    p.add_argument("--ids-csv", type=Path, default=None,
                   help="Optional companion CSV aligned row-by-row with --catalog, "
                        "contributing a 'hipparcos_id' column. Only needed for legacy "
                        "split catalogs (new catalog_hipparcos_full.csv already has IDs).")
    p.add_argument("--out-dir",  type=Path,
                   default=Path("/Users/timon/Desktop/Thesis/Code/dataset"))
    p.add_argument("--wcs-source", choices=("header", "astrometry"), default="header",
                   help="'header' uses WCS from FITS (fast, works for TESS). "
                        "'astrometry' uploads PNG to Astrometry.net.")
    p.add_argument("--api-key", default=os.environ.get("ASTROMETRY_API_KEY"),
                   help="Astrometry.net API key (or set ASTROMETRY_API_KEY env var).")
    p.add_argument("--mag-limit", type=float, default=12.0,
                   help="Drop catalog stars fainter than this magnitude. None = keep all.")
    p.add_argument("--norm", choices=("startracker", "zscale", "log", "zscore"),
                   default="startracker",
                   help="Normalization preset. 'startracker' (default) subtracts "
                        "scattered-light background and gives black sky / bright "
                        "stars — best for star-tracker training and "
                        "Astrometry.net. 'zscale' is DS9-style grey display. "
                        "'log' / 'zscore' are legacy.")
    p.add_argument("--glob", default="*.fits", help="File pattern under --fits-dir.")
    p.add_argument("--overview", action="store_true",
                   help="Also save <stem>_overlay.png with star markers into dataset/overview/.")
    args = p.parse_args()

    out_images = args.out_dir / "images"
    out_labels = args.out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    overview_dir = (args.out_dir / "overview") if args.overview else None

    _progress.info("Pipeline: norm=%s  wcs_source=%s  mag_limit=%s",
                   args.norm, args.wcs_source, args.mag_limit)

    catalog = pd.read_csv(args.catalog)
    required = {"ra_deg", "dec_deg", "mag"}
    if not required.issubset(catalog.columns):
        raise SystemExit(f"Catalog must contain columns {required}; got {list(catalog.columns)}")

    if "hipparcos_id" not in catalog.columns and args.ids_csv and args.ids_csv.exists():
        ids_df = pd.read_csv(args.ids_csv)
        if "hipparcos_id" not in ids_df.columns:
            raise SystemExit(f"{args.ids_csv} must contain 'hipparcos_id' column")
        if len(ids_df) != len(catalog):
            raise SystemExit(
                f"--ids-csv has {len(ids_df)} rows but --catalog has {len(catalog)}; "
                "they must be aligned row-by-row."
            )
        catalog = catalog.reset_index(drop=True)
        catalog["hipparcos_id"] = ids_df["hipparcos_id"].to_numpy()
        if "mag" in ids_df.columns:
            catalog["hp_mag"] = ids_df["mag"].to_numpy()

    if "hipparcos_id" not in catalog.columns:
        catalog = catalog.reset_index(drop=True)
        catalog["hipparcos_id"] = catalog.index

    _progress.info("Catalog: %d stars loaded", len(catalog))

    fits_files = sorted(args.fits_dir.glob(args.glob))
    if not fits_files:
        raise SystemExit(f"No FITS files matched {args.fits_dir}/{args.glob}")
    _progress.info("Found %d FITS files — starting processing...", len(fits_files))

    ok = 0
    log_every = max(1, len(fits_files) // 20)   # print progress ~20 times total

    for i, path in enumerate(fits_files, 1):
        if process_one(
            fits_path=path,
            out_images=out_images,
            out_labels=out_labels,
            catalog=catalog,
            wcs_source=args.wcs_source,
            api_key=args.api_key,
            mag_limit=args.mag_limit,
            norm_mode=args.norm,
            overview_dir=overview_dir,
        ):
            ok += 1

        if i % log_every == 0 or i == len(fits_files):
            _progress.info("Progress: %d / %d  (%.0f%%)", i, len(fits_files), 100*i/len(fits_files))

    _progress.info("Done: %d/%d processed  →  images: %s", ok, len(fits_files), out_images)
    return 0 if ok == len(fits_files) else 1


if __name__ == "__main__":
    sys.exit(main())
