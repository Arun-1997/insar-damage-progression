"""
Phase 1 — Step 4: Build [T, H, W, 3] numpy stacks
from SNAP-processed GeoTIFF outputs.

Channels per timestep:
    [0] coherence         — float32, range [0, 1]
    [1] amplitude ratio   — float32, log-scaled, pre/post amplitude ratio
    [2] wrapped phase     — float32, range [-π, π]

Output:
    ./data/stacks/stack.npy         — shape [T, H, W, 3]  (full AOI)
    ./data/stacks/tiles/            — shape [T, 256, 256, 3] per tile
    ./data/stacks/metadata.json     — scene dates, band stats, tile index

Requirements:
    pip install rasterio numpy scipy tqdm
"""

import os
import json
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import zoom
from tqdm import tqdm
from datetime import datetime


# ── Config ─────────────────────────────────────────────────────────────────────

PROCESSED_DIR = "./data/processed"
OUTPUT_DIR    = "./data/stacks"
TILE_DIR      = os.path.join(OUTPUT_DIR, "tiles")
TILE_SIZE     = 256       # pixels — reduce to 128 if memory constrained
TILE_OVERLAP  = 32        # pixel overlap between tiles (reduces edge artifacts)
MIN_VALID     = 0.5       # minimum fraction of valid pixels per tile

# Scene order — must match your processed output directories
SCENE_ORDER = [
    "post_01",  # T=0  ~2023-02-07  (day after quake)
    "post_02",  # T=1  ~2023-02-18
    "post_03",  # T=2  ~2023-03-02
    "post_04",  # T=3  ~2023-03-14
    "post_05",  # T=4  ~2023-03-26
    "post_06",  # T=5  ~2023-04-07
    "post_07",  # T=6  ~2023-04-19
    "post_08",  # T=7  ~2023-05-01
]

# Band names as written by SNAP to the GeoTIFF
# These may vary slightly — check with: rio info --verbose your_file.tif
BAND_NAMES = {
    "coherence":  "coh_VV",        # coherence band
    "amplitude":  "Intensity_VV",  # post-event amplitude (master amplitude also in file)
    "phase":      "Phase_ifg_VV",  # wrapped interferometric phase
}

# Pre-event amplitude — used to compute amplitude ratio
# Extracted from the first processed scene (same master for all pairs)
PRE_AMPLITUDE_BAND = "Intensity_mst_VV"


# ── Utilities ──────────────────────────────────────────────────────────────────

def find_geotiff(scene_dir: str) -> str:
    """Find the output GeoTIFF in a processed scene directory."""
    patterns = ["*.tif", "*.tiff"]
    for p in patterns:
        files = glob.glob(os.path.join(scene_dir, p))
        if files:
            return files[0]
    raise FileNotFoundError(f"No GeoTIFF found in {scene_dir}")


def get_band_index(src: rasterio.DatasetReader, band_name: str) -> int:
    """Find the rasterio band index (1-based) by name."""
    desc = [d.lower() if d else "" for d in src.descriptions]
    for i, d in enumerate(desc):
        if band_name.lower() in d:
            return i + 1  # rasterio is 1-indexed
    # Fallback: print available bands to help debug
    print(f"  Available bands: {src.descriptions}")
    raise ValueError(f"Band '{band_name}' not found. See available bands above.")


def read_band(src: rasterio.DatasetReader, band_name: str,
              target_shape: tuple = None) -> np.ndarray:
    """Read a named band, optionally resampling to target_shape (H, W)."""
    idx = get_band_index(src, band_name)
    data = src.read(idx, resampling=Resampling.bilinear)

    if target_shape is not None and data.shape != target_shape:
        scale = (target_shape[0] / data.shape[0],
                 target_shape[1] / data.shape[1])
        data = zoom(data, scale, order=1)

    return data.astype(np.float32)


def compute_amplitude_ratio(post_amp: np.ndarray,
                            pre_amp: np.ndarray,
                            eps: float = 1e-6) -> np.ndarray:
    """
    Log amplitude change ratio — standard damage proxy metric.
    Values near 0 = no change. Large negative = amplitude decrease (damage).
    """
    ratio = np.log10((post_amp + eps) / (pre_amp + eps))
    return ratio.astype(np.float32)


def normalize_coherence(coh: np.ndarray) -> np.ndarray:
    """Coherence is already in [0,1]. Clip any numerical noise."""
    return np.clip(coh, 0.0, 1.0)


def normalize_phase(phase: np.ndarray) -> np.ndarray:
    """Wrapped phase to [-π, π]. SNAP outputs in radians, just clip for safety."""
    return np.clip(phase, -np.pi, np.pi)


def normalize_amplitude_ratio(ratio: np.ndarray,
                               clip_range: tuple = (-3.0, 3.0)) -> np.ndarray:
    """Clip log amplitude ratio to reasonable range, then scale to [-1, 1]."""
    ratio = np.clip(ratio, clip_range[0], clip_range[1])
    ratio = ratio / clip_range[1]   # scale to [-1, 1]
    return ratio


# ── Per-scene feature extraction ───────────────────────────────────────────────

def extract_features(scene_name: str, target_shape: tuple = None) -> np.ndarray:
    """
    Load a processed GeoTIFF and extract [H, W, 3] feature array.
    
    Returns:
        np.ndarray of shape [H, W, 3]:
            [..., 0] = coherence        [0, 1]
            [..., 1] = amplitude ratio  [-1, 1]
            [..., 2] = wrapped phase    [-π, π]
    """
    scene_dir = os.path.join(PROCESSED_DIR, scene_name)
    tif_path  = find_geotiff(scene_dir)

    print(f"  Loading: {tif_path}")

    with rasterio.open(tif_path) as src:
        shape = target_shape or (src.height, src.width)

        # Read bands
        coh      = read_band(src, BAND_NAMES["coherence"],  shape)
        post_amp = read_band(src, BAND_NAMES["amplitude"],  shape)
        phase    = read_band(src, BAND_NAMES["phase"],      shape)
        pre_amp  = read_band(src, PRE_AMPLITUDE_BAND,       shape)

        # Replace NoData values with NaN
        nodata = src.nodata
        if nodata is not None:
            for arr in [coh, post_amp, phase, pre_amp]:
                arr[arr == nodata] = np.nan

    # Compute and normalize features
    amp_ratio = compute_amplitude_ratio(post_amp, pre_amp)

    coh       = normalize_coherence(coh)
    amp_ratio = normalize_amplitude_ratio(amp_ratio)
    phase     = normalize_phase(phase)

    # Stack to [H, W, 3]
    features = np.stack([coh, amp_ratio, phase], axis=-1)
    return features


# ── Tiling ─────────────────────────────────────────────────────────────────────

def generate_tile_indices(H: int, W: int,
                          tile_size: int,
                          overlap: int) -> list:
    """Generate (row_start, col_start) for all tiles covering [H, W]."""
    stride = tile_size - overlap
    indices = []
    for r in range(0, H - tile_size + 1, stride):
        for c in range(0, W - tile_size + 1, stride):
            indices.append((r, c))
    return indices


def is_valid_tile(tile: np.ndarray, min_valid: float) -> bool:
    """
    Reject tiles that are mostly NaN (ocean, edge effects, no-data).
    tile shape: [T, tile_size, tile_size, C]
    """
    valid_frac = np.isfinite(tile).mean()
    return valid_frac >= min_valid


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_stack() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TILE_DIR,   exist_ok=True)

    print("=" * 60)
    print("Phase 1 — Building [T, H, W, 3] feature stack")
    print("=" * 60)

    # ── 1. Extract features for each timestep ──────────────────
    print("\n[1/3] Extracting features per scene...")
    scene_features = []

    # Use first scene to determine reference shape
    ref_tif = find_geotiff(os.path.join(PROCESSED_DIR, SCENE_ORDER[0]))
    with rasterio.open(ref_tif) as src:
        target_shape = (src.height, src.width)
        profile      = src.profile
    print(f"  Reference shape: {target_shape}")

    for scene_name in tqdm(SCENE_ORDER):
        feats = extract_features(scene_name, target_shape=target_shape)
        scene_features.append(feats)

    # Stack to [T, H, W, 3]
    stack = np.stack(scene_features, axis=0)
    T, H, W, C = stack.shape
    print(f"\n  Full stack shape: {stack.shape}  ({T} timesteps)")

    # ── 2. Save full stack ─────────────────────────────────────
    print("\n[2/3] Saving full stack...")
    stack_path = os.path.join(OUTPUT_DIR, "stack.npy")
    np.save(stack_path, stack)
    print(f"  Saved: {stack_path}  ({stack.nbytes / 1e9:.2f} GB)")

    # ── 3. Generate tiles ──────────────────────────────────────
    print(f"\n[3/3] Tiling stack ({TILE_SIZE}×{TILE_SIZE}, overlap={TILE_OVERLAP})...")
    tile_indices = generate_tile_indices(H, W, TILE_SIZE, TILE_OVERLAP)
    print(f"  Candidate tiles: {len(tile_indices)}")

    saved_tiles  = 0
    skipped_tiles = 0
    tile_metadata = []

    for idx, (r, c) in enumerate(tqdm(tile_indices)):
        tile = stack[:, r:r+TILE_SIZE, c:c+TILE_SIZE, :]  # [T, 256, 256, 3]

        if not is_valid_tile(tile, MIN_VALID):
            skipped_tiles += 1
            continue

        # Replace NaN with 0 for model ingestion
        tile = np.nan_to_num(tile, nan=0.0)

        tile_path = os.path.join(TILE_DIR, f"tile_{idx:05d}.npy")
        np.save(tile_path, tile.astype(np.float32))

        tile_metadata.append({
            "tile_id":    idx,
            "row_start":  r,
            "col_start":  c,
            "row_end":    r + TILE_SIZE,
            "col_end":    c + TILE_SIZE,
            "path":       tile_path,
        })
        saved_tiles += 1

    # ── 4. Save metadata ───────────────────────────────────────
    metadata = {
        "created":      datetime.now().isoformat(),
        "scenes":       SCENE_ORDER,
        "stack_shape":  list(stack.shape),
        "tile_size":    TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "channels":     ["coherence", "amplitude_ratio", "wrapped_phase"],
        "tiles_saved":  saved_tiles,
        "tiles_skipped": skipped_tiles,
        "tiles":        tile_metadata,
    }
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Stack shape:      {stack.shape}")
    print(f"  Tiles saved:      {saved_tiles}")
    print(f"  Tiles skipped:    {skipped_tiles} (low valid pixel fraction)")
    print(f"  Output dir:       {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Metadata:         {meta_path}")
    print(f"\n  ✓ Phase 1 complete — ready for label generation (Phase 2)")
    print(f"{'='*60}")


# ── Quick sanity check ─────────────────────────────────────────────────────────

def sanity_check():
    """Load a random tile and print stats — run after build_stack()."""
    import random
    tiles = glob.glob(os.path.join(TILE_DIR, "*.npy"))
    if not tiles:
        print("No tiles found. Run build_stack() first.")
        return

    tile = np.load(random.choice(tiles))
    print(f"\nSanity check — random tile:")
    print(f"  Shape:    {tile.shape}  (T, H, W, C)")
    print(f"  dtype:    {tile.dtype}")
    print(f"  Coherence    min/max: {tile[:,:,:,0].min():.3f} / {tile[:,:,:,0].max():.3f}")
    print(f"  Amp ratio    min/max: {tile[:,:,:,1].min():.3f} / {tile[:,:,:,1].max():.3f}")
    print(f"  Wrapped phase min/max: {tile[:,:,:,2].min():.3f} / {tile[:,:,:,2].max():.3f}")
    print(f"  NaN count:  {np.isnan(tile).sum()}")


if __name__ == "__main__":
    build_stack()
    sanity_check()
