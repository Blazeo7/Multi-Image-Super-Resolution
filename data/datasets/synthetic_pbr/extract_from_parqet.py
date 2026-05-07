import gc
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def sanitize_metadata(data):
    if isinstance(data, dict):
        return {k: sanitize_metadata(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_metadata(v) for v in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    return data


def extract_shard(shard_path, out_path):
    df = pd.read_parquet(shard_path)
    image_cols = ["basecolor", "normal", "roughness", "metallic", "height"]

    # Single progress bar for the materials inside the current shard
    # 'desc' is updated to the filename so you know where you are
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Processing {shard_path.name}"
    ):
        name = row["name"]
        meta = row["metadata"]
        category = meta.get("category", "unknown")

        dest_dir = out_path / "train" / category / name
        if (dest_dir / "metadata.json").exists():
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)

        for col in image_cols:
            if col in row and row[col] is not None:
                img_data = row[col]
                file_out = dest_dir / f"{col}.png"

                try:
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        image = Image.open(io.BytesIO(img_data["bytes"]))
                        image.save(file_out)
                    elif isinstance(img_data, bytes):
                        image = Image.open(io.BytesIO(img_data))
                        image.save(file_out)
                    elif hasattr(img_data, "save"):
                        img_data.save(file_out)
                except Exception:
                    pass

        safe_meta = sanitize_metadata(meta)
        with open(dest_dir / "metadata.json", "w") as f:
            json.dump(safe_meta, f, indent=4)


def main():
    input_dir = Path("./matsynth_train_subset/data")
    output_dir = Path("./extracted_textures")
    shards = sorted(list(input_dir.glob("train-*.parquet")))

    # Set your number of workers here
    # Start with 4 to see how your cluster's memory handles it
    num_workers = 16

    print(f"Starting extraction with {num_workers} workers...")

    # Use ProcessPoolExecutor to handle the file-by-file loop
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # We wrap the executor in tqdm to see overall shard progress
        list(
            tqdm(
                executor.map(extract_shard, shards, [output_dir] * len(shards)),
                total=len(shards),
                desc="Overall Shard Progress",
            )
        )


if __name__ == "__main__":
    main()
