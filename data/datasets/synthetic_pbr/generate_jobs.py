import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path.cwd()
INPUT_BASE_DIR = BASE_DIR / "extracted_textures"
OUTPUT_DIR = BASE_DIR / "blender_dataset_renders"
JOBS_DIR = BASE_DIR / "jobs"

RENDERS_PER_TEXTURE = 10
NUM_SCENE_SETTINGS = 5

LOW_RES_PERCENTAGE = 100
REF_CAM_Z = 1.4

X_CAMERA_RANGE = (-0.625, 0.625)
Y_CAMERA_RANGE = (-0.625, 0.625)
Z_CAMERA_RANGE = (1.2, 1.6)

X_TRACKER_RANGE = (-0.15, 0.15)
Y_TRACKER_RANGE = (-0.15, 0.15)
Z_TRACKER_RANGE = (0.0, 0.0)

X_LIGHT_RANGE = (-9.0, 9.0)
Y_LIGHT_RANGE = (-9.0, 9.0)
Z_LIGHT_RANGE = (4.0, 9.0)

LIGHT_SIZE_RANGE = (0.001, 5)

LIGHT_POWER_MIN = 50
LIGHT_POWER_MAX = 250
LIGHT_POWER_SIGMA = 20

START_MEAN = 80
END_MEAN = 200


def estimate_texture_brightness(tex_dir: Path) -> float:
    for name in ("basecolor.png", "diffuse.png"):
        p = tex_dir / name
        if p.exists():
            img = np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
            img = img**2.2
            luminance = (
                0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
            )
            return float(luminance.mean())
    return 0.5


def sample_light_power(mean: float, sigma: float) -> float:
    while True:
        v = random.gauss(mean, sigma)
        if LIGHT_POWER_MIN <= v <= LIGHT_POWER_MAX:
            return v


def main():
    os.makedirs(JOBS_DIR, exist_ok=True)

    manifest_files = sorted(list(INPUT_BASE_DIR.rglob("metadata.json")), reverse=True)
    if not manifest_files:
        print(f"ERROR: No metadata.json files found in {INPUT_BASE_DIR}.")
        return

    print(f"Found {len(manifest_files)} textures. Generating job files...")

    for manifest_path in manifest_files:
        tex_dir = manifest_path.parent
        texture_name = tex_dir.name

        job_data = {
            "texture_name": texture_name,
            "texture_dir": str(tex_dir),
            "output_dir": str(OUTPUT_DIR / texture_name),
            "scenes": [],
        }

        for i_setting in range(NUM_SCENE_SETTINGS):
            t = i_setting / (NUM_SCENE_SETTINGS - 1)
            scene_mean = START_MEAN + (END_MEAN - START_MEAN) * t

            scene_data = {
                "setting_id": i_setting,
                "light_loc": [
                    random.uniform(*X_LIGHT_RANGE),
                    random.uniform(*Y_LIGHT_RANGE),
                    random.uniform(*Z_LIGHT_RANGE),
                ],
                "light_power": sample_light_power(scene_mean, LIGHT_POWER_SIGMA),
                "light_size": (
                    0.001
                    if random.uniform(0.0, 1.0) < 0.5
                    else random.uniform(*LIGHT_SIZE_RANGE)
                ),
                "renders": [],
            }

            scene_data["renders"].append(
                {
                    "type": "HQ",
                    "tracker_loc": [0.0, 0.0, 0.0],
                    "camera_loc": [0.0, 0.0, REF_CAM_Z],
                    "res_pct": 100,
                    "filename": f"tex_{texture_name.replace(' ', '_')}_{i_setting}_deadon.png",
                }
            )

            for i_render in range(RENDERS_PER_TEXTURE):
                scene_data["renders"].append(
                    {
                        "type": "LQ",
                        "render_id": i_render,
                        "tracker_loc": [
                            random.uniform(*X_TRACKER_RANGE),
                            random.uniform(*Y_TRACKER_RANGE),
                            random.uniform(*Z_TRACKER_RANGE),
                        ],
                        "camera_loc": [
                            random.uniform(*X_CAMERA_RANGE),
                            random.uniform(*Y_CAMERA_RANGE),
                            random.uniform(*Z_CAMERA_RANGE),
                        ],
                        "res_pct": LOW_RES_PERCENTAGE,
                        "filename": f"tex_{texture_name.replace(' ', '_')}_{i_setting}_angle_{i_render:03d}.png",
                    }
                )

            job_data["scenes"].append(scene_data)

        job_filepath = JOBS_DIR / f"job_{texture_name.replace(' ', '_')}.json"
        with open(job_filepath, "w") as f:
            json.dump(job_data, f, indent=4)

    print(f"\nSuccessfully generated {len(manifest_files)} job files in {JOBS_DIR}")


if __name__ == "__main__":
    main()
