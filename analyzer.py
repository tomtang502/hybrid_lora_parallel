#!/usr/bin/env python3
"""Analyzer for hybrid LoRA parallel training experiment results."""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def parse_perf_file(file_path: Path) -> Optional[Dict]:
    """Parse perf file and extract metrics."""
    try:
        content = file_path.read_text()
        if "CUDA out of memory" in content or "ERROR" in content or "FAILED" in content:
            return None

        metrics = {}
        if m := re.search(r'Latency \(s\)\s+\|\s+([\d.]+)\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+([\d.]+)', content):
            metrics['lat_mean'], metrics['lat_std'] = float(m.group(1)), float(m.group(2))
        if m := re.search(r'Throughput \(tok/s\)\s+\|\s+([\d.]+)\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+([\d.]+)', content):
            metrics['thr_mean'], metrics['thr_std'] = float(m.group(1)), float(m.group(2))
        if m := re.search(r'Peak VRAM Usage:\s+([\d.]+)\s+MB', content):
            metrics['vram'] = float(m.group(1))
        return metrics if metrics else None
    except:
        return None


def parse_dir_name(name: str) -> Optional[Dict]:
    """Parse directory name to extract parameters."""
    if m := re.match(r'(.+?)_mha\+embednorms_lora_clength(\d+)_(\d+)_b\d+_s\d+', name):
        return {'strategy': m.group(1), 'chunk': int(m.group(2)), 'ngpus': int(m.group(3))}
    return None


def collect_results() -> Dict:
    """Collect experiment results grouped by (strategy, chunk, ngpus)."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("Error: logs directory not found!")
        return {}

    results = {}
    for log_dir in logs_dir.iterdir():
        if not log_dir.is_dir() or not (params := parse_dir_name(log_dir.name)):
            continue

        key = (params['strategy'], params['chunk'], params['ngpus'])
        metrics = [parse_perf_file(log_dir / f"perf_r{r}.txt") for r in range(params['ngpus'])]
        metrics = [m for m in metrics if m]  # Filter out None

        if metrics:
            results[key] = metrics

    return results


def aggregate(metrics: List[Dict], field_prefix: str) -> tuple:
    """Aggregate metrics: mean of means ± propagated std."""
    means = [m[f'{field_prefix}_mean'] for m in metrics if f'{field_prefix}_mean' in m]
    stds = [m[f'{field_prefix}_std'] for m in metrics if f'{field_prefix}_std' in m]
    if not means or not stds:
        return None, None
    return np.mean(means), np.sqrt(np.sum(np.array(stds) ** 2)) / len(stds)


def create_table(results: Dict, metric: str) -> pd.DataFrame:
    """Create results table for a specific metric."""
    strategies = ['ddp', 'fsdp', 'fsdp_dtensor']
    chunks = [1024, 2048, 4096, 8192, 16384]

    data = {(s, c): {'Strategy': s, 'Chunk_Size': c} for s in strategies for c in chunks}

    for (strategy, chunk, ngpus), metrics in results.items():
        key = (strategy, chunk)
        if metric in ['thr', 'lat']:
            mean, std = aggregate(metrics, metric)
            data[key][f"n={ngpus}"] = f"{mean:.2f} ± {std:.2f}" if mean else "FAILED"
        else:  # vram
            vrams = [m['vram'] for m in metrics if 'vram' in m]
            data[key][f"n={ngpus}"] = f"{max(vrams):.2f}" if vrams else "FAILED"

    df = pd.DataFrame(list(data.values()))
    df['_ord'] = df['Strategy'].map({'ddp': 0, 'fsdp': 1, 'fsdp_dtensor': 2})
    df = df.sort_values(['_ord', 'Chunk_Size']).drop('_ord', axis=1)
    cols = ['Strategy', 'Chunk_Size'] + sorted([c for c in df.columns if c.startswith('n=')])
    return df[cols].fillna("FAILED")


def main():
    """Main analyzer function."""
    print("Collecting experiment results...")
    results = collect_results()

    if not results:
        print("No results found!")
        return

    print(f"Found {len(results)} experiments")

    # Create tables
    df_thr = create_table(results, 'thr')
    df_lat = create_table(results, 'lat')
    df_vram = create_table(results, 'vram')

    # Display
    for title, df in [
        ("THROUGHPUT (tok/s) - Average across ranks", df_thr),
        ("LATENCY (s) - Average across ranks", df_lat),
        ("PEAK VRAM (MB) - Maximum across ranks", df_vram)
    ]:
        print(f"\n{'='*100}\n{title}\n{'='*100}")
        print(df.to_string(index=False))
        print("="*100)

    # Save to CSV
    with open("experiment_results.csv", 'w', encoding='utf-8') as f:
        for title, df in [
            ("THROUGHPUT (tok/s) - Average across ranks", df_thr),
            ("LATENCY (s) - Average across ranks", df_lat),
            ("PEAK VRAM (MB) - Maximum across ranks", df_vram)
        ]:
            f.write(f"{title}\n")
            df.to_csv(f, index=False)
            f.write("\n\n")

    print("\nResults saved to: experiment_results.csv")


if __name__ == "__main__":
    main()
