import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple, TypeAlias

# Represents a single scene entry
SampleData: TypeAlias = Dict[str, Any]
# Represents a collection of scenes keyed by a unique identifier (ID or Path)
SampleGroup: TypeAlias = Dict[str, List[SampleData]]


def get_metadata_groups(input_dir: Path, split_by: str) -> SampleGroup:
    """Scans directory and groups metadata entries by texture_id or file path."""
    groups: SampleGroup = defaultdict(list)

    if not input_dir.exists():
        return groups

    texture_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    for tex_dir in texture_dirs:
        texture_id = tex_dir.name
        metadata_files = list(tex_dir.glob("*_metadata.json"))

        for meta_path in metadata_files:
            # Grouping logic
            group_key = texture_id if split_by == "texture" else str(meta_path)

            sample: SampleData = {
                "texture_id": texture_id,
                "base_dir": f"{input_dir.name}/{texture_id}/",
                "metadata_path": meta_path.name,
            }
            groups[group_key].append(sample)

    return groups


def calculate_split_indices(
    total_units: int, test_ratio: float, dev_ratio: float
) -> Tuple[int, int, int]:
    """Determines split sizes with a 1-unit minimum safety."""
    if total_units == 0:
        return 0, 0, 0

    n_test = max(1, int(total_units * test_ratio)) if test_ratio > 0 else 0
    n_dev = max(1, int(total_units * dev_ratio)) if dev_ratio > 0 else 0

    if (n_test + n_dev) >= total_units:
        n_train = 1 if total_units >= 3 else total_units
        n_test = 1 if total_units >= 2 else 0
        n_dev = 1 if total_units >= 3 else 0
    else:
        n_train = total_units - n_test - n_dev

    return n_train, n_test, n_dev


def partition_keys(
    keys: List[str], test_split: float, dev_split: float, seed: int
) -> Dict[str, List[str]]:
    """Shuffles and splits keys into train, test, and dev sets."""
    random.seed(seed)
    shuffled_keys = list(keys)
    random.shuffle(shuffled_keys)

    n_train, n_test, _ = calculate_split_indices(len(shuffled_keys), test_split, dev_split)

    return {
        "train": shuffled_keys[:n_train],
        "test": shuffled_keys[n_train : n_train + n_test],
        "dev": shuffled_keys[n_train + n_test :],
    }


def save_manifest(
    output_path: Path,
    samples: List[SampleData],
    args: argparse.Namespace,
    split_name: str,
    key_count: int,
):
    """Formats and writes the manifest to disk."""
    samples.sort(key=lambda x: (x["texture_id"], x["metadata_path"]))

    content = {
        "scale_factor": args.scale,
        "num_aligned_images": args.n_supporting,
        "split_method": args.split_by,
        "samples": samples,
    }

    with open(output_path, "w") as f:
        json.dump(content, f, indent=4)

    status_label = "textures" if args.split_by == "texture" else "samples"
    if args.split_by == "texture":
        print(f"[{split_name.upper()}] {key_count} {status_label} -> {len(samples)} total samples.")
    else:
        print(f"[{split_name.upper()}] {len(samples)} samples.")


def create_manifests(args: argparse.Namespace):
    """Main execution flow."""
    root = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather
    groups: SampleGroup = get_metadata_groups(root, args.split_by)
    if not groups:
        print(f"Error: No valid data found in {root}")
        return

    # Split
    split_map = partition_keys(list(groups.keys()), args.test_split, args.dev_split, args.seed)

    # Save
    for split_name, keys in split_map.items():
        if not keys:
            continue

        split_samples: List[SampleData] = [s for k in keys for s in groups[k]]
        save_manifest(
            out_dir / f"{split_name}_manifest.json", split_samples, args, split_name, len(keys)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleaned Texture Manifest Creator")
    parser.add_argument("--input_dir", type=str, default="./samples")
    parser.add_argument("--output_dir", type=str, default="./data/manifests")
    parser.add_argument("--split_by", type=str, choices=["texture", "sample"], default="texture")
    parser.add_argument("-n", "--n_supporting", type=int, default=4)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--dev_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    create_manifests(parser.parse_args())
