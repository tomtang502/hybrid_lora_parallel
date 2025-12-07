#!/usr/bin/env python3
"""Analyzer for hybrid LoRA parallel training experiment results."""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def parse_perf_file(file_path: Path) -> Dict:
    """Parse perf file and extract metrics."""
    try:
        content = file_path.read_text()

        if "CUDA out of memory" in content or "OutOfMemoryError" in content:
            return {"status": "OOM"}
        if "ERROR" in content or "FAILED" in content:
            return {"status": "ERROR"}

        metrics = {"status": "SUCCESS"}

        if m := re.search(r'Latency \(s\)\s+\|\s+([\d.]+)\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+([\d.]+)', content):
            metrics['latency_mean'] = float(m.group(1))
            metrics['latency_std'] = float(m.group(2))

        if m := re.search(r'Throughput \(tok/s\)\s+\|\s+([\d.]+)\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+([\d.]+)', content):
            metrics['throughput_mean'] = float(m.group(1))
            metrics['throughput_std'] = float(m.group(2))

        return metrics
    except Exception:
        return {"status": "PARSE_ERROR"}


def parse_log_dir_name(dir_name: str) -> Optional[Dict]:
    """Parse directory name to extract experiment parameters."""
    if m := re.match(r'(.+?)_mha\+embednorms_lora_clength(\d+)_(\d+)_b(\d+)_s(\d+)', dir_name):
        return {'strategy': m.group(1), 'chunk_size': int(m.group(2)), 'num_gpus': int(m.group(3))}
    return None


def format_metric(mean, std):
    """Format metric as 'mean ± std' or 'FAILED'."""
    return f"{mean:.2f} ± {std:.2f}" if mean and std else "FAILED"


def collect_results() -> List[Dict]:
    """Collect all experiment results from log directories."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("Error: logs directory not found!")
        return []

    results = []
    log_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
    print(f"Found {len(log_dirs)} log directories")

    for log_dir in log_dirs:
        if not (params := parse_log_dir_name(log_dir.name)):
            continue

        rank_metrics = []
        for rank in range(params['num_gpus']):
            perf_file = log_dir / f"perf_r{rank}.txt"
            metrics = parse_perf_file(perf_file) if perf_file.exists() else {"status": "MISSING"}

            if metrics.get('status') == 'SUCCESS':
                rank_metrics.append({
                    'rank': rank,
                    'latency_mean': metrics.get('latency_mean'),
                    'latency_std': metrics.get('latency_std'),
                    'throughput_mean': metrics.get('throughput_mean'),
                    'throughput_std': metrics.get('throughput_std'),
                })
            else:
                rank_metrics.append({'rank': rank, 'status': metrics.get('status')})

        results.append({
            **params,
            'rank_metrics': rank_metrics,
            'status': 'SUCCESS' if all('latency_mean' in m for m in rank_metrics) else 'INCOMPLETE'
        })

    return results


def create_results_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Create wide-format DataFrame with per-rank columns."""
    strategies = ['ddp', 'fsdp', 'fsdp_dtensor']
    chunk_sizes = [1024, 2048, 4096, 8192, 16384]

    # Pre-populate all combinations
    config_data = {(s, c): {'Strategy': s, 'Chunk_Size': c}
                   for s in strategies for c in chunk_sizes}

    # Fill in actual data
    for result in results:
        key = (result['strategy'], result['chunk_size'])
        for rank_data in result.get('rank_metrics', []):
            rank, n = rank_data.get('rank', -1), result['num_gpus']

            config_data[key][f"Throughput (n={n}, P{rank}) (tok/s)"] = format_metric(
                rank_data.get('throughput_mean'), rank_data.get('throughput_std'))
            config_data[key][f"Latency (n={n}, P{rank}) (s)"] = format_metric(
                rank_data.get('latency_mean'), rank_data.get('latency_std'))

    df = pd.DataFrame(list(config_data.values()))

    # Sort and reorder
    df['_order'] = df['Strategy'].map({'ddp': 0, 'fsdp': 1, 'fsdp_dtensor': 2})
    df = df.sort_values(['_order', 'Chunk_Size']).drop('_order', axis=1)

    cols = ['Strategy', 'Chunk_Size']
    cols += sorted([c for c in df.columns if c.startswith('Throughput')])
    cols += sorted([c for c in df.columns if c.startswith('Latency')])

    return df[cols].fillna("FAILED")


def main():
    """Main analyzer function."""
    print("Collecting experiment results...")
    results = collect_results()

    if not results:
        print("No results found!")
        return

    # Summary
    successful = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(results)} experiments | {successful} successful | {len(results)-successful} incomplete")
    print(f"{'='*80}")

    # Show incomplete
    for r in (r for r in results if r['status'] == 'INCOMPLETE'):
        failed = [m['rank'] for m in r['rank_metrics'] if m.get('status')]
        print(f"  {r['strategy']}, chunk={r['chunk_size']}, gpus={r['num_gpus']}, failed ranks={failed}")

    # Create and save results
    df = create_results_dataframe(results)
    print("\n" + "="*150)
    print("EXPERIMENT RESULTS")
    print("="*150)
    print(df.to_string(index=False))
    print("="*150)

    df.to_csv("experiment_results.csv", index=False)
    print(f"\nResults saved to: experiment_results.csv")


if __name__ == "__main__":
    main()
