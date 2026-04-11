import argparse
import json
import random
from pathlib import Path


def extract_id(filename):
    """
    Strips known suffixes to isolate the unique texture base name.
    Example: 'tex_0004_particle_board_angle_000_LQ_aligned.png' -> 'tex_0004_particle_board'
    """
    name = Path(filename).stem
    suffixes = ['_deadon', '_angle', '_LQ', '_LR', '_aligned', '_HQ']
    
    base = name
    for s in suffixes:
        if s in base:
            base = base.split(s)[0]
    return base

def create_manifests(args):
    # .resolve() turns the input path into an absolute path
    root = Path(args.input_dir).resolve()
    hr_dir = root / "hr"
    lr_dir = root / "lr"
    
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_map = {}

    if not hr_dir.exists():
        print(f"Error: HR directory not found at {hr_dir}")
        return

    for hr_file in hr_dir.glob("*.png"):
        tex_id = extract_id(hr_file.name)
        # Convert to absolute path string
        data_map[tex_id] = {
            "target": str(hr_file.resolve()),
            "reference": None,
            "supporting": []
        }

    if not lr_dir.exists():
        print(f"Error: LR directory not found at {lr_dir}")
        return

    for lr_file in lr_dir.glob("*.png"):
        tex_id = extract_id(lr_file.name)
        
        if tex_id not in data_map:
            continue

        # Convert to absolute path string
        abs_lr_path = str(lr_file.resolve())

        if "deadon_LR" in lr_file.name or "deadon_LR" in lr_file.name.upper():
            data_map[tex_id]["reference"] = abs_lr_path
        elif "aligned" in lr_file.name:
            data_map[tex_id]["supporting"].append(abs_lr_path)

    valid_ids = []
    for tex_id, paths in data_map.items():
        has_target = paths["target"] is not None
        has_ref = paths["reference"] is not None
        actual_count = len(paths["supporting"])
        
        if has_target and has_ref and actual_count == args.n_supporting:
            valid_ids.append(tex_id)
        else:
            if args.verbose:
                reason = "missing HR/LR" if not (has_target and has_ref) else f"had {actual_count} supporting"
                print(f"Dropping {tex_id}: {reason}")

    random.seed(42)
    random.shuffle(valid_ids)

    total = len(valid_ids)
    if total == 0:
        print(f"No samples matched criteria (N={args.n_supporting}). Check filenames or N value.")
        return

    train_end = int(total * args.train_split)
    test_end = train_end + int(total * args.test_split)

    splits = {
        "train": valid_ids[:train_end],
        "test": valid_ids[train_end:test_end],
        "dev": valid_ids[test_end:]
    }

    for split_name, ids in splits.items():
        if not ids: continue
        
        samples = []
        for tex_id in ids:
            entry = data_map[tex_id]
            samples.append({
                "id": tex_id,
                "reference": entry["reference"],
                "supporting": sorted(entry["supporting"]),
                "target": entry["target"]
            })

        manifest = {
            "scale_factor": args.scale,
            "num_aligned_images": args.n_supporting,
            "samples": samples
        }

        output_path = out_dir / f"{split_name}_manifest.json"
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=4)
        
        print(f"Created {output_path} with {len(samples)} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ML manifests from HR/LR texture pairs.")
    parser.add_argument("--input_dir", type=str, default="./", help="Root dir containing 'hr' and 'lr' folders")
    parser.add_argument("--output_dir", type=str, default="./manifests", help="Where to save JSON files")
    parser.add_argument("-n", "--n_supporting", type=int, default=4, help="Exact number of supporting images required")
    parser.add_argument("--scale", type=int, default=2, help="Scale factor for manifest header")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--dev_split", type=float, default=0.1)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print dropped IDs")

    args = parser.parse_args()
    create_manifests(args)
