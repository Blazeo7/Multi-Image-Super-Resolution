import os

from huggingface_hub import HfApi, snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def download_flat_train_subset(n_files=20, local_dir="./matsynth_train_subset"):
    api = HfApi()

    print("Fetching file list from MatSynth...")
    all_files = api.list_repo_files(repo_id="gvecchio/MatSynth", repo_type="dataset")

    prefix = "val"
    # Filter for files starting with 'train-' in a flat structure
    parquets = sorted([f for f in all_files if prefix in f and f.endswith(".parquet")])

    if not parquets:
        print(f"Error: Could not find any files starting with '{prefix}-'.")
        return

    # Select the first N shards
    files_to_download = parquets[:n_files]

    print(f"Found {len(parquets)} total train shards.")
    print(f"Downloading the first {len(files_to_download)} shards to {local_dir}...")

    # snapshot_download will respect the specific file list
    snapshot_download(
        repo_id="gvecchio/MatSynth",
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=files_to_download,
        max_workers=5,  # Crucial for avoiding thread/ulimit crashes
        resume_download=True,
    )

    print("\nDownload complete.")


if __name__ == "__main__":
    download_flat_train_subset(n_files=100)
