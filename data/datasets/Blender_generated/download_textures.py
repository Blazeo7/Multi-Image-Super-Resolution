import os

# from datasets import load_dataset
#
# # ==========================================
# # CONFIGURATION
# # ==========================================
print("Script started. Importing libraries...", flush=True)

import argparse
from pathlib import Path
import json
import shutil

import datasets

datasets.logging.set_verbosity_info()
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset.")
    parser.add_argument(
        "--base_dir",
        default="./downloaded_textures",
        help="Directory to save the downloaded files."
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    base_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nSaving files to: {base_dir.resolve()}", flush=True)
    print("Connecting to Hugging Face...", flush=True)

    ds = load_dataset(
        "gvecchio/MatSynth",
        streaming=True,
    )

    image_cols = [
        "basecolor", "diffuse", "displacement", "specular",
        "height", "metallic", "normal", "opacity",
        "roughness", "blend_mask"
    ]

    for split in ds.keys():
        split_dir = base_dir / split

        downloaded_names = set()
        if split_dir.exists():
            for meta_path in split_dir.rglob("metadata.json"):
                downloaded_names.add(meta_path.parent.name)

        print(f"\n--- Starting split: {split} ---", flush=True)
        if downloaded_names:
            print(f"Found {len(downloaded_names)} fully downloaded items. Scanning stream...", flush=True)

        # 1. THE CRITICAL FIX: Turn off automatic image decoding!
        # This stops Hugging Face from using your CPU to decompress the PNGs just to yield the row.
        for col in image_cols:
            if col in ds[split].features:
                ds[split] = ds[split].cast_column(col, datasets.Image(decode=False))

        # We use a standard loop. Because decode=False, yielding items is now blazingly fast.
        for item in ds[split]:
            name = item["name"]

            if name in downloaded_names:
                continue

            dest_dir = base_dir / split / item["metadata"]["category"] / name
            dest_dir.mkdir(exist_ok=True, parents=True)

            # 2. RAW BYTE WRITING
            # Since we turned off decoding, `item[col]` is now a dictionary containing
            # the raw file bytes. We dump them to disk instantly without using PIL.
            for col in image_cols:
                if col in item and item[col] is not None:
                    # Depending on caching, HF returns either raw bytes or a local path
                    if "bytes" in item[col] and item[col]["bytes"] is not None:
                        with open(dest_dir / f"{col}.png", "wb") as f:
                            f.write(item[col]["bytes"])
                    elif "path" in item[col] and item[col]["path"] is not None:
                        shutil.copy(item[col]["path"], dest_dir / f"{col}.png")

            # SAVE METADATA LAST
            with open(dest_dir / "metadata.json", "w") as f:
                item["metadata"]["physical_size"] = str(item["metadata"]["physical_size"])
                json.dump(item["metadata"], f, indent=4)

            print(f"Successfully downloaded: {name}", flush=True)

            # Add to our set so we don't process it again if the stream acts weird
            downloaded_names.add(name)