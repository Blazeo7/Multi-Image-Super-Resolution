import json
import os
import random
from pathlib import Path

# --- CONFIGURATION (Matches your original script) ---
BASE_DIR = Path.cwd()
INPUT_BASE_DIR = BASE_DIR / "downloaded_textures"
OUTPUT_DIR = BASE_DIR / "blender_dataset_renders2"
JOBS_DIR = BASE_DIR / "jobs"

RENDERS_PER_TEXTURE = 5
NUM_SCENE_SETTINGS = 1
REF_CAM_Z = 1.4

X_CAMERA_RANGE = (-0.625, 0.625)
Y_CAMERA_RANGE = (-0.625, 0.625)
Z_CAMERA_RANGE = (1.2, 1.6)
X_TRACKER_RANGE = (-0.1, 0.1)
Y_TRACKER_RANGE = (-0.1, 0.1)
Z_TRACKER_RANGE = (0.0, 0.0)
X_LIGHT_RANGE = (-9.0, 9.0)
Y_LIGHT_RANGE = (-9.0, 9.0)
Z_LIGHT_RANGE = (4.0, 9.0)
LIGHT_POWER_RANGE = (45.0, 80.0)


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
            "scenes": []
        }

        for i_setting in range(NUM_SCENE_SETTINGS):
            scene_data = {
                "setting_id": i_setting,
                "light_loc": [random.uniform(*X_LIGHT_RANGE), random.uniform(*Y_LIGHT_RANGE),
                              random.uniform(*Z_LIGHT_RANGE)],
                "light_power": random.uniform(*LIGHT_POWER_RANGE),
                "renders": []
            }

            # 1. Dead-on HQ Render (Main Camera)
            scene_data["renders"].append({
                "type": "HQ",
                "tracker_loc": [0.0, 0.0, 0.0],
                "camera_loc": [0.0, 0.0, REF_CAM_Z],
                "res_pct": 100,
                "filename": f"tex_{texture_name.replace(' ', '_')}_{i_setting}_deadon_HQ.png"
            })

            # 2. Angled LQ Renders (Random Cameras)
            for i_render in range(RENDERS_PER_TEXTURE):
                scene_data["renders"].append({
                    "type": "LQ",
                    "render_id": i_render,
                    "tracker_loc": [random.uniform(*X_TRACKER_RANGE), random.uniform(*Y_TRACKER_RANGE),
                                    random.uniform(*Z_TRACKER_RANGE)],
                    "camera_loc": [random.uniform(*X_CAMERA_RANGE), random.uniform(*Y_CAMERA_RANGE),
                                   random.uniform(*Z_CAMERA_RANGE)],
                    "res_pct": 50,
                    "filename": f"tex_{texture_name.replace(' ', '_')}_{i_setting}_angle_{i_render:03d}_LQ.png"
                })

            job_data["scenes"].append(scene_data)

        # Save job to JSON
        job_filepath = JOBS_DIR / f"job_{texture_name.replace(' ', '_')}.json"
        with open(job_filepath, 'w') as f:
            json.dump(job_data, f, indent=4)

    print(f"Successfully generated {len(manifest_files)} job files in {JOBS_DIR}")


if __name__ == "__main__":
    main()