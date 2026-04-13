"""
Phase 1 - Step 1: Download Sentinel-1 SLC scenes
Türkiye 2023 Earthquake (Kahramanmaraş)
Earthquake date: 2023-02-06

Requires:
    pip install sentinelsat
    
Setup:
    Register at https://dataspace.copernicus.eu/
    Set env vars:
        export COPERNICUS_USER=your_username
        export COPERNICUS_PASSWORD=your_password
"""

import os
from datetime import date
from sentinelsat import SentinelAPI
from shapely.geometry import box

# ── credentials ────────────────────────────────────────────────────────────────
USER     = os.environ["COPERNICUS_USER"]
PASSWORD = os.environ["COPERNICUS_PASSWORD"]
API_URL  = "https://catalogue.dataspace.copernicus.eu/odata/v1"

# ── AOI: Kahramanmaraş urban core ──────────────────────────────────────────────
# Bounding box [min_lon, min_lat, max_lon, max_lat]
AOI = box(36.5, 37.2, 37.5, 38.0)

# ── Sentinel-1 acquisition parameters ──────────────────────────────────────────
PLATFORM      = "Sentinel-1"
PRODUCT_TYPE  = "SLC"          # Single Look Complex — required for InSAR
MODE          = "IW"           # Interferometric Wide swath
POLARISATION  = "VV"           # VV preferred for urban coherence

# ── Time windows ───────────────────────────────────────────────────────────────
# Sentinel-1 has 12-day repeat cycle over this region
# Pick scenes that are 12 days apart for best coherence

SCENES = {
    "pre_event": {
        "start": date(2023, 1, 13),   # one cycle before quake
        "end":   date(2023, 1, 26),
    },
    "post_01": {"start": date(2023, 2, 7),  "end": date(2023, 2, 8)},   # day after
    "post_02": {"start": date(2023, 2, 18), "end": date(2023, 2, 20)},  # +12 days
    "post_03": {"start": date(2023, 3, 2),  "end": date(2023, 3, 4)},
    "post_04": {"start": date(2023, 3, 14), "end": date(2023, 3, 16)},
    "post_05": {"start": date(2023, 3, 26), "end": date(2023, 3, 28)},
    "post_06": {"start": date(2023, 4, 7),  "end": date(2023, 4, 9)},
    "post_07": {"start": date(2023, 4, 19), "end": date(2023, 4, 21)},
    "post_08": {"start": date(2023, 5, 1),  "end": date(2023, 5, 3)},
}

# ── Download ────────────────────────────────────────────────────────────────────
def search_and_download(api: SentinelAPI, label: str, start: date, end: date, outdir: str):
    print(f"\n{'─'*60}")
    print(f"Searching: {label}  [{start} → {end}]")

    products = api.query(
        area=AOI.wkt,
        date=(start, end),
        platformname=PLATFORM,
        producttype=PRODUCT_TYPE,
        sensoroperationalmode=MODE,
        polarisationmode=POLARISATION,
    )

    if not products:
        print(f"  ⚠  No products found for {label}")
        return

    # Pick the product with smallest size if multiple found (same pass, different slices)
    df = api.to_dataframe(products).sort_values("size")
    chosen = df.iloc[0]
    print(f"  ✓  Found: {chosen['title']}")
    print(f"     Size:  {chosen['size']}")
    print(f"     Date:  {chosen['beginposition']}")

    scene_dir = os.path.join(outdir, label)
    os.makedirs(scene_dir, exist_ok=True)

    api.download(chosen.name, directory_path=scene_dir)
    print(f"  ✓  Downloaded to {scene_dir}")


def main():
    outdir = "./data/raw_slc"
    os.makedirs(outdir, exist_ok=True)

    print("Connecting to Copernicus Data Space...")
    api = SentinelAPI(USER, PASSWORD, API_URL)
    print("Connected.\n")

    for label, window in SCENES.items():
        search_and_download(api, label, window["start"], window["end"], outdir)

    print(f"\n{'='*60}")
    print("Download complete.")
    print(f"Scenes saved to: {os.path.abspath(outdir)}")
    print("\nNext step: Open SNAP and run preprocessing pipeline.")


if __name__ == "__main__":
    main()
