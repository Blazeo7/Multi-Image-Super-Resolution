import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def get_texture_groups(input_dir: Path) -> Dict[str, List[str]]:
    """
    Scans directory and groups metadata entries by texture folder name.
    Returns: { "texture_name": ["meta1.json", "meta2.json"] }
    """
    groups = {}

    if not input_dir.exists():
        return groups

    # Get all subdirectories (each represents a texture)
    for tex_dir in input_dir.iterdir():
        if not tex_dir.is_dir():
            continue

        texture_name = tex_dir.name
        metadata_files = sorted([f.name for f in tex_dir.glob("*_metadata.json")])

        if metadata_files:
            groups[texture_name] = metadata_files

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
    shuffled_keys = sorted(list(keys))
    random.shuffle(shuffled_keys)

    n_train, n_test, _ = calculate_split_indices(len(shuffled_keys), test_split, dev_split)

    return {
        "train": shuffled_keys[:n_train],
        "test": shuffled_keys[n_train : n_train + n_test],
        "dev": shuffled_keys[n_train + n_test :],
    }


def save_manifest(
    output_path: Path,
    texture_keys: List[str],
    all_groups: Dict[str, List[str]],
    args: argparse.Namespace,
    split_name: str,
):
    """Saves the textures and their scenes to the manifest."""

    # Transform the dict into a list of objects
    samples_list = []
    for k in sorted(texture_keys):
        samples_list.append({"texture_name": k, "scenes": all_groups[k]})

    total_scenes = sum(len(v["scenes"]) for v in samples_list)

    content = {
        "scale_factor": args.scale,
        "num_aligned_images": args.n_supporting,
        "data_dir": f"{Path(args.input_dir).name}/",
        "samples": samples_list,
    }

    with open(output_path, "w") as f:
        json.dump(content, f, indent=4)

    print(f"[{split_name.upper()}] {len(samples_list)} textures -> {total_scenes} total scenes.")


def create_manifests(args: argparse.Namespace):
    root = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather all data grouped by Texture ID
    groups = get_texture_groups(root)
    if not groups:
        print(f"Error: No valid metadata found in {root}")
        return

    # Split by Texture keys
    split_map = partition_keys(list(groups.keys()), args.test_split, args.dev_split, args.seed)

    # Save manifests
    for split_name, keys in split_map.items():
        if not keys:
            continue
        save_manifest(out_dir / f"{split_name}_manifest.json", keys, groups, args, split_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured Texture Manifest Creator")
    parser.add_argument("--input_dir", type=str, default="./samples")
    parser.add_argument("--output_dir", type=str, default="./data/manifests")
    parser.add_argument("-n", "--n_supporting", type=int, default=4)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--dev_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    create_manifests(parser.parse_args())
