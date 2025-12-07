#!/usr/bin/env python3
"""
Automated sbatch job submission for hybrid LoRA parallel training experiments.
New approach: Submit 3 jobs (one per GPU count), each running all 15 strategy×chunk_size combinations.
Total: 3 jobs × (3 strategies × 5 chunk sizes) = 45 experiments.
"""

import subprocess
import sys
import time

# Configuration: Set to True to enable dependency chaining between jobs
USE_DEPENDENCIES = True


def submit_job(num_gpus, dependency_job_id=None):
    """Submit a single batch job for all experiments with the given GPU count."""

    # Construct sbatch command
    cmd = [
        'sbatch',
        f'--gres=gpu:h100-80:{num_gpus}',
        f'--job-name=hybrid_par_g{num_gpus}'
    ]

    # Add dependency if specified
    if dependency_job_id:
        cmd.extend(['--dependency', f'afterany:{dependency_job_id}'])

    # Add the script and arguments
    cmd.extend([
        'scripts/run.sh',
        '--num-gpus', str(num_gpus)
    ])

    try:
        # Print the full command for debugging
        print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Extract job ID from sbatch output (format: "Submitted batch job 12345")
        job_id = result.stdout.strip().split()[-1]

        experiments_per_job = 15  # 3 strategies × 5 chunk sizes
        print(f"✓ Submitted job {job_id}: {num_gpus} GPU(s), {experiments_per_job} experiments")
        return job_id

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to submit job for {num_gpus} GPU(s)")
        print(f"  Error: {e.stderr}")
        return None


def main():
    """Submit batch jobs - one per GPU count."""

    # Experimental parameters
    gpu_counts = [1, 2, 4]
    strategies = ['ddp', 'fsdp', 'fsdp_dtensor']
    chunk_sizes = [1024, 2048, 4096, 8192, 16384]

    experiments_per_job = len(strategies) * len(chunk_sizes)
    total_experiments = len(gpu_counts) * experiments_per_job

    print("="*60)
    print("HYBRID LORA PARALLEL TRAINING - BATCH JOB SUBMISSION")
    print("="*60)
    print(f"Job structure: {len(gpu_counts)} jobs")
    print(f"  - Each job runs {experiments_per_job} experiments")
    print(f"  - Strategies: {', '.join(strategies)}")
    print(f"  - Chunk sizes: {', '.join(map(str, chunk_sizes))}")
    print(f"Total experiments: {total_experiments}")
    print("="*60)

    # Submit jobs
    previous_job_id = None
    submitted_jobs = []

    for i, num_gpus in enumerate(gpu_counts):
        print(f"\nSubmitting batch job {i+1}/{len(gpu_counts)}")

        job_id = submit_job(num_gpus, previous_job_id)

        if job_id:
            submitted_jobs.append({
                'job_id': job_id,
                'num_gpus': num_gpus,
                'experiments': experiments_per_job
            })
            if USE_DEPENDENCIES:
                previous_job_id = job_id
        else:
            print("  Continuing with remaining jobs despite failure...")

        # Small delay to avoid overwhelming the scheduler
        time.sleep(0.5)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUBMISSION SUMMARY")
    print(f"{'='*60}")
    print(f"Jobs submitted: {len(submitted_jobs)}/{len(gpu_counts)}")

    if submitted_jobs:
        print(f"\nSubmitted job IDs:")
        for job in submitted_jobs:
            print(f"  - Job {job['job_id']}: {job['num_gpus']} GPU(s), {job['experiments']} experiments")

        print(f"\nTo monitor job queue: squeue -u $USER")
        print(f"To check specific job: squeue -j <job_id>")
        print(f"To cancel all jobs: scancel -u $USER")

    print(f"\nLogs will be generated in:")
    print(f"  - Experiment logs: logs/<run_name>/")
    print(f"  - SLURM logs: logs/slurm/slurm-<job_id>.out")
    print(f"\nRun analyzer.py after jobs complete to collect results.")


if __name__ == "__main__":
    main()
