import os
import subprocess
from pathlib import Path
from multiprocessing import Pool

# --- CLUSTER CONFIGURATION ---
NUM_GPUS = 1  # Total number of GPUs on this machine
BLENDERS_PER_GPU = 4  # Increase if your textures are small and VRAM is mostly empty

BLEND_FILE = "Texture_gen_scene.blend"

def run_blender_job(args):
    job_json_path, gpu_id = args
    print(f"Starting {job_json_path.name} on GPU {gpu_id}")

    # 1. Copy the current OS environment variables
    env = os.environ.copy()

    # 2. Mask the GPUs! Blender will now think only this single GPU exists in the machine.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Use env["HIP_VISIBLE_DEVICES"] = str(gpu_id) if you are on AMD GPUs
    # Use env["OPTIX_VISIBLE_DEVICES"] = str(gpu_id) if using OptiX backend

    command = [
        "blender", "-b", BLEND_FILE,
        "-P", "render_job.py",
        "--", str(job_json_path)
    ]

    # Pass the modified environment to the subprocess
    subprocess.run(command, env=env)
    print(f"Finished {job_json_path.name}")


if __name__ == '__main__':
    jobs = list(Path("jobs").glob("*.json"))

    # Create a list of tuples: (Job Path, Target GPU ID)
    # This assigns Job 0 -> GPU 0, Job 1 -> GPU 1, ..., Job 8 -> GPU 0
    job_args = [(job, i % NUM_GPUS) for i, job in enumerate(jobs)]

    total_workers = NUM_GPUS * BLENDERS_PER_GPU

    with Pool(total_workers) as p:
        p.map(run_blender_job, job_args)