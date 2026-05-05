import os
import json
import cv2
import numpy as np
import glob


def process_homography_transformations(directory='./homographies'):
    # Find all JSON files in the specified directory
    json_files = glob.glob(os.path.join(directory, '*.json'))

    if not json_files:
        print("No JSON files found in the directory.")
        return

    for json_file in json_files:
        print(f"Processing {json_file}...")

        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {json_file}. Skipping.")
                continue

        for lq_filename, info in data.items():
            hq_filename = info.get('target_hq_file')
            matrix = info.get('homography_matrix')

            if not matrix:
                print(f"No homography matrix found for {lq_filename}. Skipping.")
                continue

            # Convert the 3x3 list into a numpy array of floats
            H = np.array(matrix, dtype=np.float32)

            # Load the source (LQ) image
            lq_path = os.path.join(directory, lq_filename)
            lq_img = cv2.imread(lq_path)

            if lq_img is None:
                print(f"Warning: Could not find or load {lq_filename}. Skipping.")
                continue

            # Attempt to load the target (HQ) image to get the exact output dimensions.
            # If the HQ image isn't in the folder, it falls back to the LQ image's dimensions.
            hq_path = os.path.join(directory, hq_filename)
            hq_img = cv2.imread(hq_path)

            if hq_img is not None:
                h, w = hq_img.shape[:2]
            else:
                h, w = lq_img.shape[:2]

            # Apply the perspective transformation
            transformed_img = cv2.warpPerspective(lq_img, H, (w, h))

            # Save the transformed image with a new prefix
            output_filename = f"transformed_{lq_filename}"
            output_path = os.path.join(directory, output_filename)
            cv2.imwrite(output_path, transformed_img)

            print(f"  -> Successfully generated: {output_filename}")


if __name__ == "__main__":
    # Run the script in the current directory
    process_homography_transformations()