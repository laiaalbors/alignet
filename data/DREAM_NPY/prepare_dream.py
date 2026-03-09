import argparse
import os
from pathlib import Path

import numpy as np
import rasterio


def convert_one_tif_to_npy(
    tif_path: Path,
    out_path: Path,
    overwrite: bool,
    gdal_cachemax_mb: int,
) -> bool:
    """
    Reads band 1 from a tif with rasterio and saves as float32 .npy (H, W).
    Returns True if written, False if skipped.
    """
    if out_path.exists() and not overwrite:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Conservative caches + avoid dataset sharing/pooling surprises
    with rasterio.Env(GDAL_CACHEMAX=gdal_cachemax_mb, VSI_CACHE=False):
        with rasterio.open(tif_path, "r", sharing=False) as ds:
            arr = ds.read(1, out_dtype="float32")  # (H, W), float32

    # Ensure contiguous float32 (usually already is, but keep it safe)
    arr = np.ascontiguousarray(arr, dtype=np.float32)

    np.save(out_path, arr)  # saves as .npy
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert DREAM GeoTIFFs to float32 NumPy arrays (.npy) using rasterio (reads only band 1)."
    )
    parser.add_argument("--root", type=str, required=True, help="Path to DREAM root folder (contains zone1, zone2, ...)")
    parser.add_argument("--out", type=str, required=True, help="Output root folder for NPYs (mirrors the structure)")
    parser.add_argument(
        "--modalities",
        type=str,
        default="optique,radar,S1,S2,SRTM",
        help="Comma-separated list of modality folder names to include",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy files")
    parser.add_argument("--gdal-cachemax-mb", type=int, default=64, help="GDAL_CACHEMAX in MB (per process)")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]

    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    tif_paths = []
    for zone_dir in sorted(root.glob("zone*")):
        if not zone_dir.is_dir():
            continue
        for mod in modalities:
            mod_dir = zone_dir / mod
            if not mod_dir.exists():
                continue
            tif_paths.extend(sorted(mod_dir.rglob("*.tif")))
            tif_paths.extend(sorted(mod_dir.rglob("*.tiff")))

    total = len(tif_paths)
    print(f"[INFO] Found {total} GeoTIFFs under {root} for modalities={modalities}")

    written = 0
    skipped = 0
    failed = 0

    for i, tif_path in enumerate(tif_paths, 1):
        rel = tif_path.relative_to(root)
        out_path = (out_root / rel).with_suffix(".npy")

        if args.dry_run:
            print(f"[DRY] {tif_path} -> {out_path}")
            continue

        try:
            did_write = convert_one_tif_to_npy(
                tif_path=tif_path,
                out_path=out_path,
                overwrite=args.overwrite,
                gdal_cachemax_mb=args.gdal_cachemax_mb,
            )
            if did_write:
                written += 1
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] Failed on {tif_path}: {e}")

        if (i % 200) == 0 or i == total:
            print(f"[PROGRESS] {i}/{total} | written={written} skipped={skipped} failed={failed}")

    print(f"[DONE] written={written} skipped={skipped} failed={failed} | out={out_root}")


if __name__ == "__main__":
    # These help avoid some GDAL caching behavior especially on remote/fuse-backed filesystems.
    os.environ.setdefault("GDAL_MAX_DATASET_POOL_SIZE", "0")
    os.environ.setdefault("VSI_CACHE", "FALSE")
    os.environ.setdefault("VSI_CACHE_SIZE", "0")

    main()