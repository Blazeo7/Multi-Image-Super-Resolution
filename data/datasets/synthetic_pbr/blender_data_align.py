import cv2
import json
import numpy as np
import os
from pathlib import Path

# --- CONFIGURATION ---
# Adjust these paths to match where your files are actually stored
JSON_FILE_PATH = "./homographies/matrices.json"
INPUT_DIR = Path("./homographies")   # Folder containing the LQ renders
OUTPUT_DIR = Path("./homographies/warped") # Folder to save the warped images

# The resolution of your Main/HQ camera (the target space for the homography)
TARGET_WIDTH = 1024
TARGET_HEIGHT = 1024

def main():
    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the homography matrices from the JSON file
    if not os.path.exists(JSON_FILE_PATH):
        print(f"Error: Could not find {JSON_FILE_PATH}")
        return

    with open(JSON_FILE_PATH, 'r') as f:
        matrix_data = json.load(f)

    print(f"Loaded {len(matrix_data)} transformation matrices. Starting warp...")

    # 2. Iterate through each image and matrix
    for filename, matrix_list in matrix_data.items():
        input_image_path = INPUT_DIR / filename
        output_image_path = OUTPUT_DIR / f"warped_{filename}"

        # Check if the image actually exists before trying to process it
        if not input_image_path.exists():
            print(f"Skipping {filename}: File not found in {INPUT_DIR}")
            continue

        # 3. Read the image
        # cv2.imread loads the image as a NumPy array in BGR color format
        img = cv2.imread(str(input_image_path))

        if img is None:
            print(f"Error: Failed to load image {input_image_path}")
            continue

        # 4. Convert the JSON list-of-lists into a 3x3 float32 NumPy array
        # OpenCV requires transformation matrices to be floating point numbers
        H_matrix = np.array(matrix_list, dtype=np.float32)

        # 5. Apply the Homography Warp
        # dsize is (width, height). We use the TARGET dimensions so the LQ image
        # is stretched and projected onto the exact same canvas as your HQ image.
        warped_img = cv2.warpPerspective(
            src=img,
            M=H_matrix,
            dsize=(TARGET_WIDTH, TARGET_HEIGHT),
            flags=cv2.INTER_LINEAR # Linear interpolation works best for standard textures
        )

        # 6. Save the result
        cv2.imwrite(str(output_image_path), warped_img)
        print(f"Successfully warped and saved: {output_image_path.name}")

    print("\nAll warping operations complete!")

if __name__ == "__main__":
    main()