import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

def create_splits(manifest_path, output_dir, train_ratio=0.8, val_ratio=0.1):
    # 1. Load the master manifest
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    scale_factor = data.get("scale_factor", 2)
    samples = data.get("samples", [])
    
    # 2. Group samples by Scene ID (the first part of the 'id' before the first '_')
    # Example: "000_s00..." -> "000"
    scene_groups = defaultdict(list)
    for s in samples:
        scene_id = s['id'].split('_')[0]
        scene_groups[scene_id].append(s)
    
    scene_ids = list(scene_groups.keys())
    random.seed(42) # For reproducibility
    random.shuffle(scene_ids)
    
    # 3. Calculate split indices
    total_scenes = len(scene_ids)
    train_end = int(total_scenes * train_ratio)
    val_end = train_end + int(total_scenes * val_ratio)
    
    splits = {
        "train": scene_ids[:train_end],
        "val": scene_ids[train_end:val_end],
        "test": scene_ids[val_end:]
    }
    
    # 4. Save the split manifests
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {}
    
    for name, ids in splits.items():
        split_samples = []
        for sid in ids:
            split_samples.extend(scene_groups[sid])
        
        split_manifest = {
            "scale_factor": scale_factor,
            "samples": split_samples
        }
        
        file_path = output_dir / f"{name}.json"
        with open(file_path, 'w') as f:
            json.dump(split_manifest, f, indent=4)
        
        summary[name] = {"scenes": len(ids), "samples": len(split_samples)}
        print(f"Created {file_path}: {len(ids)} scenes, {len(split_samples)} samples.")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split manifest into train/val/test by Scene ID.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to master manifest.json")
    parser.add_argument("--output-dir", type=str, default="splits", help="Where to save split jsons")
    parser.add_argument("--train", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.1, help="Val ratio")
    
    args = parser.parse_args()
    
    create_splits(args.manifest, args.output_dir, args.train, args.val)
