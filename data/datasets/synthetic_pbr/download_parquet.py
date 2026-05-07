import os

from huggingface_hub import HfApi, snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def download_flat_train_subset(n_files=20, local_dir="./matsynth_train_subset"):
    api = HfApi()

    print("Fetching file list from MatSynth...")
    all_files = api.list_repo_files(repo_id="gvecchio/MatSynth", repo_type="dataset")

    # Filter for files starting with 'train-' in a flat structure
    train_parquets = sorted(
        [f for f in all_files if "train" in f and f.endswith(".parquet")]
    )

    if not train_parquets:
        print("Error: Could not find any files starting with 'train-'.")
        return

    # Select the first N shards
    files_to_download = train_parquets[:n_files]

    print(f"Found {len(train_parquets)} total train shards.")
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
