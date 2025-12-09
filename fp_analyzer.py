#!/usr/bin/env python3
"""Analyzer for full precision (FP) training experiment results."""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def parse_report_file(file_path: Path) -> Optional[Dict]:
    """Parse a report file with Rank 0 and Rank 1 data."""
    try:
        content = file_path.read_text()

        # Parse Rank 0 and Rank 1 sections
        ranks_data = {}
        for rank in [0, 1]:
            pattern = f'=== Report for Rank {rank} ===(.*?)(?==== Report for Rank|$)'
            match = re.search(pattern, content, re.DOTALL)
            if not match:
                return None

            rank_content = match.group(1)

            # Parse latency
            lat_match = re.search(r'Latency \(s\)\s+\|\s+([\d.]+)\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+([\d.]+)', rank_content)
            # Parse throughput
            thr_match = re.search(r'Throughput \(tok/s\)\s+\|\s+([\d.]+)\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+([\d.]+)', rank_content)
            # Parse VRAM
            vram_match = re.search(r'Peak VRAM Usage:\s+([\d.]+)\s+MB', rank_content)

            if not (lat_match and thr_match and vram_match):
                return None

            ranks_data[rank] = {
                'lat_mean': float(lat_match.group(1)),
                'lat_std': float(lat_match.group(2)),
                'thr_mean': float(thr_match.group(1)),
                'thr_std': float(thr_match.group(2)),
                'vram': float(vram_match.group(1))
            }

        return ranks_data
    except:
        return None


def aggregate_ranks(ranks_data: Dict) -> Dict:
    """Aggregate data from two ranks.

    Args:
        ranks_data: Dict with keys 0 and 1 containing rank data
    """
    r0, r1 = ranks_data[0], ranks_data[1]

    # Mean of means
    lat_mean = (r0['lat_mean'] + r1['lat_mean']) / 2
    thr_mean = (r0['thr_mean'] + r1['thr_mean']) / 2

    # Std of mean: sqrt(sigma_1^2 + sigma_2^2) / 2
    lat_std = np.sqrt(r0['lat_std']**2 + r1['lat_std']**2) / 2
    thr_std = np.sqrt(r0['thr_std']**2 + r1['thr_std']**2) / 2

    # VRAM metrics
    vram_max = max(r0['vram'], r1['vram'])
    vram_diff = abs(r0['vram'] - r1['vram'])
    vram_sum = r0['vram'] + r1['vram']

    return {
        'lat_mean': lat_mean,
        'lat_std': lat_std,
        'thr_mean': thr_mean,
        'thr_std': thr_std,
        'vram_max': vram_max,
        'vram_diff': vram_diff,
        'vram_sum': vram_sum
    }


def collect_pp_ablation() -> Dict:
    """Collect results from pp_ablation folder."""
    data_dir = Path("fp_data/pp_ablation")
    if not data_dir.exists():
        print(f"Error: {data_dir} not found!")
        return {}

    results = {}
    valid_chunksizes = [1024, 2048, 3172, 4096]

    for file_path in data_dir.glob("pp_split*_sl*_mbs*_report.txt"):
        # Parse filename: pp_split{split}_sl{sl}_mbs{mbs}_report.txt
        match = re.match(r'pp_split(\d+)_sl(\d+)_mbs(\d+)_report\.txt', file_path.name)
        if not match:
            continue

        split = int(match.group(1))
        sl = int(match.group(2))
        mbs = int(match.group(3))

        # Filter chunksizes
        if sl not in valid_chunksizes:
            continue

        ranks_data = parse_report_file(file_path)
        if not ranks_data:
            continue

        # Aggregate all metrics (includes vram_max, vram_diff, vram_sum)
        agg_data = aggregate_ranks(ranks_data)

        # Normalize PP latency by dividing by 2 (microbatch definition difference)
        agg_data['lat_mean'] = agg_data['lat_mean'] / 2
        agg_data['lat_std'] = agg_data['lat_std'] / 2

        key = (split, sl, mbs)
        results[key] = agg_data

    return results


def collect_toy_fsdp_dtensor() -> Dict:
    """Collect results from toy_fsdp_dtensor folder (only ac0)."""
    data_dir = Path("fp_data/toy_fsdp_dtensor")
    if not data_dir.exists():
        print(f"Error: {data_dir} not found!")
        return {}

    results = {}
    valid_chunksizes = [1024, 2048, 3172, 4096]

    for file_path in data_dir.glob("fsdp_ac0_sl*_mbs*_report.txt"):
        # Parse filename: fsdp_ac0_sl{sl}_mbs{mbs}_report.txt
        match = re.match(r'fsdp_ac0_sl(\d+)_mbs(\d+)_report\.txt', file_path.name)
        if not match:
            continue

        sl = int(match.group(1))
        mbs = int(match.group(2))

        # Filter chunksizes
        if sl not in valid_chunksizes:
            continue

        ranks_data = parse_report_file(file_path)
        if not ranks_data:
            continue

        # Aggregate all metrics (includes vram_max, vram_diff, vram_sum)
        agg_data = aggregate_ranks(ranks_data)

        key = (sl, mbs)
        results[key] = agg_data

    return results


def create_pp_ablation_tables(results: Dict):
    """Create tables for pp_ablation data."""
    if not results:
        return None, None, None, None, None

    # Get unique values
    splits = sorted(set(k[0] for k in results.keys()))
    chunksizes = sorted(set(k[1] for k in results.keys()))
    mbss = sorted(set(k[2] for k in results.keys()))

    # Create tables for each metric
    tables = {}
    for metric in ['thr', 'lat', 'vram_max', 'vram_diff', 'vram_sum']:
        data = {}
        for split in splits:
            for sl in chunksizes:
                key = (split, sl)
                data[key] = {'Split': split, 'Chunk_Size': sl}

                for mbs in mbss:
                    result_key = (split, sl, mbs)
                    if result_key in results:
                        r = results[result_key]
                        if metric == 'thr':
                            data[key][f"mbs={mbs}"] = f"{r['thr_mean']:.2f} ± {r['thr_std']:.2f}"
                        elif metric == 'lat':
                            data[key][f"mbs={mbs}"] = f"{r['lat_mean']:.4f} ± {r['lat_std']:.4f}"
                        elif metric == 'vram_max':
                            data[key][f"mbs={mbs}"] = f"{r['vram_max']:.2f}"
                        elif metric == 'vram_diff':
                            data[key][f"mbs={mbs}"] = f"{r['vram_diff']:.2f}"
                        elif metric == 'vram_sum':
                            data[key][f"mbs={mbs}"] = f"{r['vram_sum']:.2f}"
                    else:
                        data[key][f"mbs={mbs}"] = "FAILED"

        df = pd.DataFrame(list(data.values()))
        cols = ['Split', 'Chunk_Size'] + [f"mbs={m}" for m in mbss]
        tables[metric] = df[cols].fillna("FAILED")

    return tables['thr'], tables['lat'], tables['vram_max'], tables['vram_diff'], tables['vram_sum']


def create_toy_fsdp_tables(results: Dict):
    """Create tables for toy_fsdp_dtensor data."""
    if not results:
        return None, None, None, None, None

    # Get unique values
    chunksizes = sorted(set(k[0] for k in results.keys()))
    mbss = sorted(set(k[1] for k in results.keys()))

    # Create tables for each metric
    tables = {}
    for metric in ['thr', 'lat', 'vram_max', 'vram_diff', 'vram_sum']:
        data = {}
        for sl in chunksizes:
            data[sl] = {'Chunk_Size': sl}

            for mbs in mbss:
                result_key = (sl, mbs)
                if result_key in results:
                    r = results[result_key]
                    if metric == 'thr':
                        data[sl][f"mbs={mbs}"] = f"{r['thr_mean']:.2f} ± {r['thr_std']:.2f}"
                    elif metric == 'lat':
                        data[sl][f"mbs={mbs}"] = f"{r['lat_mean']:.4f} ± {r['lat_std']:.4f}"
                    elif metric == 'vram_max':
                        data[sl][f"mbs={mbs}"] = f"{r['vram_max']:.2f}"
                    elif metric == 'vram_diff':
                        data[sl][f"mbs={mbs}"] = f"{r['vram_diff']:.2f}"
                    elif metric == 'vram_sum':
                        data[sl][f"mbs={mbs}"] = f"{r['vram_sum']:.2f}"
                else:
                    data[sl][f"mbs={mbs}"] = "FAILED"

        df = pd.DataFrame(list(data.values()))
        cols = ['Chunk_Size'] + [f"mbs={m}" for m in mbss]
        tables[metric] = df[cols].fillna("FAILED")

    return tables['thr'], tables['lat'], tables['vram_max'], tables['vram_diff'], tables['vram_sum']


def main():
    """Main analyzer function."""
    print("Collecting FP experiment results...")

    # Collect pp_ablation data
    print("\n" + "="*80)
    print("PP ABLATION STUDY")
    print("="*80)
    pp_results = collect_pp_ablation()
    print(f"Found {len(pp_results)} experiments")

    if pp_results:
        df_thr, df_lat, df_vram_max, df_vram_diff, df_vram_sum = create_pp_ablation_tables(pp_results)

        print("\n" + "="*100)
        print("THROUGHPUT (tok/s) - Average across ranks")
        print("="*100)
        print(df_thr.to_string(index=False))

        print("\n" + "="*100)
        print("LATENCY (s) - Average across ranks")
        print("="*100)
        print(df_lat.to_string(index=False))

        print("\n" + "="*100)
        print("PEAK VRAM (MB) - Maximum across ranks")
        print("="*100)
        print(df_vram_max.to_string(index=False))

        print("\n" + "="*100)
        print("VRAM DIFFERENCE (MB) - Absolute difference between ranks")
        print("="*100)
        print(df_vram_diff.to_string(index=False))

        print("\n" + "="*100)
        print("VRAM SUM (MB) - Sum across ranks")
        print("="*100)
        print(df_vram_sum.to_string(index=False))

        # Save to CSV
        with open("fp_pp_ablation_results.csv", 'w', encoding='utf-8') as f:
            f.write("THROUGHPUT (tok/s) - Average across ranks\n")
            df_thr.to_csv(f, index=False)
            f.write("\n\n")

            f.write("LATENCY (s) - Average across ranks\n")
            df_lat.to_csv(f, index=False)
            f.write("\n\n")

            f.write("PEAK VRAM (MB) - Maximum across ranks\n")
            df_vram_max.to_csv(f, index=False)
            f.write("\n\n")

            f.write("VRAM DIFFERENCE (MB) - Absolute difference between ranks\n")
            df_vram_diff.to_csv(f, index=False)
            f.write("\n\n")

            f.write("VRAM SUM (MB) - Sum across ranks\n")
            df_vram_sum.to_csv(f, index=False)

        print("\nSaved: fp_pp_ablation_results.csv")

    # Collect toy_fsdp_dtensor data
    print("\n" + "="*80)
    print("TOY FSDP+DTENSOR STUDY (ac0 only)")
    print("="*80)
    toy_results = collect_toy_fsdp_dtensor()
    print(f"Found {len(toy_results)} experiments")

    if toy_results:
        df_thr, df_lat, df_vram_max, df_vram_diff, df_vram_sum = create_toy_fsdp_tables(toy_results)

        print("\n" + "="*100)
        print("THROUGHPUT (tok/s) - Average across ranks")
        print("="*100)
        print(df_thr.to_string(index=False))

        print("\n" + "="*100)
        print("LATENCY (s) - Average across ranks")
        print("="*100)
        print(df_lat.to_string(index=False))

        print("\n" + "="*100)
        print("PEAK VRAM (MB) - Maximum across ranks")
        print("="*100)
        print(df_vram_max.to_string(index=False))

        print("\n" + "="*100)
        print("VRAM DIFFERENCE (MB) - Absolute difference between ranks")
        print("="*100)
        print(df_vram_diff.to_string(index=False))

        print("\n" + "="*100)
        print("VRAM SUM (MB) - Sum across ranks")
        print("="*100)
        print(df_vram_sum.to_string(index=False))

        # Save to CSV
        with open("fp_toy_fsdp_dtensor_results.csv", 'w', encoding='utf-8') as f:
            f.write("THROUGHPUT (tok/s) - Average across ranks\n")
            df_thr.to_csv(f, index=False)
            f.write("\n\n")

            f.write("LATENCY (s) - Average across ranks\n")
            df_lat.to_csv(f, index=False)
            f.write("\n\n")

            f.write("PEAK VRAM (MB) - Maximum across ranks\n")
            df_vram_max.to_csv(f, index=False)
            f.write("\n\n")

            f.write("VRAM DIFFERENCE (MB) - Absolute difference between ranks\n")
            df_vram_diff.to_csv(f, index=False)
            f.write("\n\n")

            f.write("VRAM SUM (MB) - Sum across ranks\n")
            df_vram_sum.to_csv(f, index=False)

        print("\nSaved: fp_toy_fsdp_dtensor_results.csv")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
