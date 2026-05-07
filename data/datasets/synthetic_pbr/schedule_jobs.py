import argparse
import os
import subprocess
import time
from multiprocessing import Process, Queue
from pathlib import Path

# ==========================================
# CONSTANTS
# ==========================================
BLENDER_BIN = "/pub/users/xnguye28/blender-5.1.1-linux-x64/blender"
DEFAULT_BLEND_FILE = "Texture_gen_scene.blend"
DEFAULT_SCRIPT = "render_job.py"


def gpu_worker(gpu_id, job_queue, blend_file, script_file):
    """
    One worker process per GPU slot.
    Pulls jobs from the shared queue until it is empty.
    CUDA_VISIBLE_DEVICES is set to the single gpu_id this worker owns.
    """
    while True:
        try:
            job_json_path = job_queue.get_nowait()
        except Exception:
            break  # Queue is empty — worker exits cleanly

        print(f"--> [GPU {gpu_id}] Starting {job_json_path.name}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        command = [
            BLENDER_BIN,
            "-b",
            blend_file,
            "-P",
            script_file,
            "--",
            str(job_json_path),
        ]

        subprocess.run(command, env=env)
        print(f"--- [GPU {gpu_id}] Finished {job_json_path.name}")
        time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Multi-GPU Render Cluster")
    parser.add_argument(
        "-b",
        "--blenders",
        type=int,
        default=4,
        help="Number of Blender instances per GPU (default: 4)",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=DEFAULT_BLEND_FILE,
        help=f"Path to .blend file (default: {DEFAULT_BLEND_FILE})",
    )
    parser.add_argument(
        "-s",
        "--script",
        type=str,
        default=DEFAULT_SCRIPT,
        help=f"Path to render python script (default: {DEFAULT_SCRIPT})",
    )
    parser.add_argument(
        "-j",
        "--jobs_dir",
        type=str,
        default="jobs",
        help="Directory containing job JSON files (default: 'jobs')",
    )
    args = parser.parse_args()

    # --- Read GPU list from CUDA_VISIBLE_DEVICES ---
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_env:
        print("ERROR: CUDA_VISIBLE_DEVICES is not set.")
        print("Example: CUDA_VISIBLE_DEVICES=0,1,2 python render_cluster.py -b 4")
        exit(1)

    gpu_list = [int(x.strip()) for x in cuda_env.split(",") if x.strip().isdigit()]
    if not gpu_list:
        print(
            f"ERROR: Could not parse any GPU IDs from CUDA_VISIBLE_DEVICES='{cuda_env}'"
        )
        exit(1)

    # --- Gather Jobs ---
    jobs_path = Path(args.jobs_dir)
    if not jobs_path.exists():
        print(f"Error: Jobs directory '{args.jobs_dir}' not found.")
        exit(1)

    jobs = list(jobs_path.glob("*.json"))
    if not jobs:
        print("No job JSON files found. Nothing to do.")
        exit(0)

    # --- Build Shared Queue ---
    job_queue = Queue()
    for job in jobs:
        job_queue.put(job)

    total_workers = len(gpu_list) * args.blenders

    print("========================================")
    print("CLUSTER STARTED")
    print(f"  Blend file:     {args.file}")
    print(f"  Script:         {args.script}")
    print(f"  Jobs dir:       {args.jobs_dir}")
    print(f"  GPUs:           {gpu_list}")
    print(f"  Instances/GPU:  {args.blenders}")
    print(f"  Total workers:  {total_workers}")
    print(f"  Total jobs:     {len(jobs)}")
    print("========================================")

    # --- Spawn Workers ---
    processes = []
    for gpu_id in gpu_list:
        for _ in range(args.blenders):
            p = Process(
                target=gpu_worker,
                args=(gpu_id, job_queue, args.file, args.script),
            )
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    print("\nAll jobs completed successfully.")
