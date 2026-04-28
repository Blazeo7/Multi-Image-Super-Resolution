import os
import subprocess
import argparse
from pathlib import Path
from multiprocessing import Pool

# --- DEFAULTS ---
DEFAULT_BLEND_FILE = "Texture_gen_scene.blend"
DEFAULT_SCRIPT = "render_job.py"


def run_blender_job(args):
    job_json_path, gpu_id, blend_file, script_file = args
    print(f"--> [GPU {gpu_id}] Starting {job_json_path.name}")

    # Copy environment and mask GPU
    env = os.environ.copy()
    # Masking for NVIDIA (CUDA/OptiX) and AMD (HIP)
    gpu_str = str(gpu_id)
    env["CUDA_VISIBLE_DEVICES"] = gpu_str

    command = [
        "blender", "-b", blend_file,
        "-P", script_file,
        "--", str(job_json_path)
    ]

    # Run Blender
    subprocess.run(command, env=env)  # stdout=DEVNULL keeps terminal clean
    print(f"--- [GPU {gpu_id}] Finished {job_json_path.name}")


if __name__ == '__main__':
    # 1. Setup Argparse
    parser = argparse.ArgumentParser(description="Blender Multi-GPU Render Cluster")

    parser.add_argument("-g", "--gpus", type=int, default=1,
                        help="Number of GPUs to use (default: 1)")
    parser.add_argument("-b", "--blenders", type=int, default=4,
                        help="Number of Blender instances per GPU (default: 4)")
    parser.add_argument("-f", "--file", type=str, default=DEFAULT_BLEND_FILE,
                        help=f"Path to .blend file (default: {DEFAULT_BLEND_FILE})")
    parser.add_argument("-s", "--script", type=str, default=DEFAULT_SCRIPT,
                        help=f"Path to python script (default: {DEFAULT_SCRIPT})")
    parser.add_argument("-j", "--jobs_dir", type=str, default="jobs",
                        help="Directory containing job JSONs (default: 'jobs')")

    args = parser.parse_args()

    # 2. Gather Jobs
    jobs_path = Path(args.jobs_dir)
    if not jobs_path.exists():
        print(f"Error: Jobs directory '{args.jobs_dir}' not found.")
        exit(1)

    jobs = list(jobs_path.glob("*.json"))
    if not jobs:
        print("No job JSON files found. Nothing to do.")
        exit(0)

    # 3. Prepare Arguments for Pool
    # We pass the GPU ID, blend file, and script path to each worker
    job_args = [
        (job, i % args.gpus, args.file, args.script)
        for i, job in enumerate(jobs)
    ]

    total_workers = args.gpus * args.blenders

    print("========================================")
    print(f"CLUSTER STARTED")
    print(f"Total GPUs:       {args.gpus}")
    print(f"Instances/GPU:    {args.blenders}")
    print(f"Total Workers:    {total_workers}")
    print(f"Total Jobs:       {len(jobs)}")
    print("========================================")

    # 4. Execute
    with Pool(total_workers) as p:
        p.map(run_blender_job, job_args)

    print("\nAll jobs completed successfully.")