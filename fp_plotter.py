#!/usr/bin/env python3
"""Plotter for full precision (FP) training experiment results."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Color scheme
COLORS = {
    'pp': '#d62728',           # Red
    'fsdp_dtensor': '#2ca02c'  # Green
}

# Line styles for different MBS values (not used for ablation plots)
LINE_STYLES = {
    1: '-',      # Solid
    2: '--',     # Dashed
    4: ':'       # Dotted
}

# Colors for different MBS values in ablation plots (light to dark red)
MBS_COLORS = {
    1: '#ff7f7f',  # Light red
    2: '#d62728',  # Medium red (same as PP)
    4: '#8b0000'   # Dark red
}

# Labels
STRATEGY_LABELS = {
    'pp': 'PP',
    'fsdp_dtensor': 'FSDP+DTensor'
}

MBS_LABELS = {
    1: 'mbs=1',
    2: 'mbs=2',
    4: 'mbs=4'
}


def parse_metric_value(value_str):
    """Parse 'mean ± std' or single value string into (mean, std) or (value, None)."""
    if value_str == "FAILED" or pd.isna(value_str):
        return None, None
    try:
        value_str = str(value_str).strip()
        if ' ± ' in value_str:
            parts = value_str.split(' ± ')
            return float(parts[0]), float(parts[1])
        else:
            return float(value_str), None
    except:
        return None, None


def load_pp_data():
    """Load PP ablation data from CSV."""
    csv_file = Path("fp_pp_ablation_results.csv")
    if not csv_file.exists():
        print(f"Error: {csv_file} not found! Run fp_analyzer.py first.")
        return None, None, None, None, None

    # Read CSV sections
    with open(csv_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find section boundaries
    sections = {}
    current_section = None
    section_start = 0

    for i, line in enumerate(lines):
        if 'THROUGHPUT' in line:
            current_section = 'throughput'
            section_start = i + 1
        elif 'LATENCY' in line:
            if current_section == 'throughput':
                sections['throughput'] = (section_start, i - 2)
            current_section = 'latency'
            section_start = i + 1
        elif 'PEAK VRAM' in line:
            if current_section == 'latency':
                sections['latency'] = (section_start, i - 2)
            current_section = 'vram_max'
            section_start = i + 1
        elif 'VRAM DIFFERENCE' in line:
            if current_section == 'vram_max':
                sections['vram_max'] = (section_start, i - 2)
            current_section = 'vram_diff'
            section_start = i + 1
        elif 'VRAM SUM' in line:
            if current_section == 'vram_diff':
                sections['vram_diff'] = (section_start, i - 2)
            current_section = 'vram_sum'
            section_start = i + 1

    sections['vram_sum'] = (section_start, len(lines))

    # Parse each section
    df_thr = pd.read_csv(csv_file, skiprows=sections['throughput'][0],
                         nrows=sections['throughput'][1] - sections['throughput'][0])
    df_lat = pd.read_csv(csv_file, skiprows=sections['latency'][0],
                         nrows=sections['latency'][1] - sections['latency'][0])
    df_vram_max = pd.read_csv(csv_file, skiprows=sections['vram_max'][0],
                              nrows=sections['vram_max'][1] - sections['vram_max'][0])
    df_vram_diff = pd.read_csv(csv_file, skiprows=sections['vram_diff'][0],
                               nrows=sections['vram_diff'][1] - sections['vram_diff'][0])
    df_vram_sum = pd.read_csv(csv_file, skiprows=sections['vram_sum'][0],
                              nrows=sections['vram_sum'][1] - sections['vram_sum'][0])

    # Filter out non-data rows and convert Split column to int for proper filtering
    df_thr = df_thr[pd.to_numeric(df_thr['Split'], errors='coerce').notna()]
    df_thr['Split'] = df_thr['Split'].astype(int)

    df_lat = df_lat[pd.to_numeric(df_lat['Split'], errors='coerce').notna()]
    df_lat['Split'] = df_lat['Split'].astype(int)

    df_vram_max = df_vram_max[pd.to_numeric(df_vram_max['Split'], errors='coerce').notna()]
    df_vram_max['Split'] = df_vram_max['Split'].astype(int)

    df_vram_diff = df_vram_diff[pd.to_numeric(df_vram_diff['Split'], errors='coerce').notna()]
    df_vram_diff['Split'] = df_vram_diff['Split'].astype(int)

    df_vram_sum = df_vram_sum[pd.to_numeric(df_vram_sum['Split'], errors='coerce').notna()]
    df_vram_sum['Split'] = df_vram_sum['Split'].astype(int)

    return df_thr, df_lat, df_vram_max, df_vram_diff, df_vram_sum


def load_fsdp_data():
    """Load FSDP+DTensor data from CSV."""
    csv_file = Path("fp_toy_fsdp_dtensor_results.csv")
    if not csv_file.exists():
        print(f"Error: {csv_file} not found! Run fp_analyzer.py first.")
        return None, None, None, None, None

    # Read CSV sections
    with open(csv_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find section boundaries
    sections = {}
    current_section = None
    section_start = 0

    for i, line in enumerate(lines):
        if 'THROUGHPUT' in line:
            current_section = 'throughput'
            section_start = i + 1
        elif 'LATENCY' in line:
            if current_section == 'throughput':
                sections['throughput'] = (section_start, i - 2)
            current_section = 'latency'
            section_start = i + 1
        elif 'PEAK VRAM' in line:
            if current_section == 'latency':
                sections['latency'] = (section_start, i - 2)
            current_section = 'vram_max'
            section_start = i + 1
        elif 'VRAM DIFFERENCE' in line:
            if current_section == 'vram_max':
                sections['vram_max'] = (section_start, i - 2)
            current_section = 'vram_diff'
            section_start = i + 1
        elif 'VRAM SUM' in line:
            if current_section == 'vram_diff':
                sections['vram_diff'] = (section_start, i - 2)
            current_section = 'vram_sum'
            section_start = i + 1

    sections['vram_sum'] = (section_start, len(lines))

    # Parse each section
    df_thr = pd.read_csv(csv_file, skiprows=sections['throughput'][0],
                         nrows=sections['throughput'][1] - sections['throughput'][0])
    df_lat = pd.read_csv(csv_file, skiprows=sections['latency'][0],
                         nrows=sections['latency'][1] - sections['latency'][0])
    df_vram_max = pd.read_csv(csv_file, skiprows=sections['vram_max'][0],
                              nrows=sections['vram_max'][1] - sections['vram_max'][0])
    df_vram_diff = pd.read_csv(csv_file, skiprows=sections['vram_diff'][0],
                               nrows=sections['vram_diff'][1] - sections['vram_diff'][0])
    df_vram_sum = pd.read_csv(csv_file, skiprows=sections['vram_sum'][0],
                              nrows=sections['vram_sum'][1] - sections['vram_sum'][0])

    # Filter out non-data rows (Chunk_Size should be numeric)
    df_thr = df_thr[pd.to_numeric(df_thr['Chunk_Size'], errors='coerce').notna()]
    df_lat = df_lat[pd.to_numeric(df_lat['Chunk_Size'], errors='coerce').notna()]
    df_vram_max = df_vram_max[pd.to_numeric(df_vram_max['Chunk_Size'], errors='coerce').notna()]
    df_vram_diff = df_vram_diff[pd.to_numeric(df_vram_diff['Chunk_Size'], errors='coerce').notna()]
    df_vram_sum = df_vram_sum[pd.to_numeric(df_vram_sum['Chunk_Size'], errors='coerce').notna()]

    return df_thr, df_lat, df_vram_max, df_vram_diff, df_vram_sum


def plot_comparison(df_pp, df_fsdp, metric_name, pp_split=16, pp_mbs=4):
    """Plot comparison of PP vs FSDP+DTensor for a metric vs chunk size."""
    if df_pp is None or df_fsdp is None:
        return

    plt.figure(figsize=(10, 6))

    # Extract PP data (split=16, mbs=4)
    pp_row = df_pp[(df_pp['Split'] == pp_split) & (df_pp['Chunk_Size'].notna())]
    chunk_sizes = []
    pp_means = []
    pp_stds = []

    for _, row in pp_row.iterrows():
        chunk = int(row['Chunk_Size'])
        col = f'mbs={pp_mbs}'
        if col in row.index:
            mean, std = parse_metric_value(row[col])
            if mean is not None:
                chunk_sizes.append(chunk)
                pp_means.append(mean)
                pp_stds.append(std if std is not None else 0)

    # Plot PP line
    if chunk_sizes:
        plt.errorbar(chunk_sizes, pp_means, yerr=pp_stds,
                    marker='o', linewidth=4, markersize=12,
                    color=COLORS['pp'], label=STRATEGY_LABELS['pp'],
                    capsize=10, capthick=4, elinewidth=4, alpha=0.9)

    # Extract FSDP data (mbs=4)
    fsdp_chunk_sizes = []
    fsdp_means = []
    fsdp_stds = []

    for _, row in df_fsdp.iterrows():
        chunk = int(row['Chunk_Size'])
        col = 'mbs=4'
        if col in row.index:
            mean, std = parse_metric_value(row[col])
            if mean is not None:
                fsdp_chunk_sizes.append(chunk)
                fsdp_means.append(mean)
                fsdp_stds.append(std if std is not None else 0)

    # Plot FSDP line
    if fsdp_chunk_sizes:
        plt.errorbar(fsdp_chunk_sizes, fsdp_means, yerr=fsdp_stds,
                    marker='s', linewidth=4, markersize=12,
                    color=COLORS['fsdp_dtensor'], label=STRATEGY_LABELS['fsdp_dtensor'],
                    capsize=10, capthick=4, elinewidth=4, alpha=0.9)

    # Labels and formatting
    if metric_name == 'throughput':
        ylabel = 'Throughput per Device (tok/s)'
        title = f'Throughput per Device vs Chunk Size'
    elif metric_name == 'latency':
        ylabel = 'Normalized Latency (s)'
        title = f'Latency vs Chunk Size'
    elif metric_name == 'vram_max':
        ylabel = 'Peak VRAM Usage (MB)'
        title = f'Peak VRAM Usage vs Chunk Size'
    elif metric_name == 'vram_diff':
        ylabel = 'VRAM Difference (MB)'
        title = f'VRAM Difference vs Chunk Size'
    elif metric_name == 'vram_sum':
        ylabel = 'Total VRAM (MB)'
        title = f'Total VRAM vs Chunk Size'
    else:
        ylabel = metric_name
        title = f'{metric_name} vs Chunk Size'

    plt.xlabel('Chunk Size', fontsize=24, fontweight='bold')
    plt.ylabel(ylabel, fontsize=21.5, fontweight='bold')

    all_chunks = sorted(set(chunk_sizes + fsdp_chunk_sizes))
    plt.xticks(all_chunks, [str(c) for c in all_chunks], fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=22)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = Path("plots") / f"fp_plot_{metric_name}_vs_chunk_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_single_strategy(df, metric_name, strategy, split=None, mbs=4):
    """Plot single strategy metric vs chunk size."""
    if df is None:
        return

    plt.figure(figsize=(10, 6))

    # Extract data
    if split is not None:
        # For PP data with split parameter
        df_filtered = df[(df['Split'] == split) & (df['Chunk_Size'].notna())]
    else:
        # For FSDP data without split parameter
        df_filtered = df[df['Chunk_Size'].notna()]

    chunk_sizes = []
    means = []
    stds = []

    for _, row in df_filtered.iterrows():
        chunk = int(row['Chunk_Size'])
        col = f'mbs={mbs}'
        if col in row.index:
            mean, std = parse_metric_value(row[col])
            if mean is not None:
                chunk_sizes.append(chunk)
                means.append(mean)
                stds.append(std if std is not None else 0)

    # Plot line
    if chunk_sizes:
        if stds and any(s > 0 for s in stds):
            plt.errorbar(chunk_sizes, means, yerr=stds,
                        marker='o', linewidth=4, markersize=12,
                        color=COLORS[strategy], label=STRATEGY_LABELS[strategy],
                        capsize=10, capthick=4, elinewidth=4, alpha=0.9)
        else:
            plt.plot(chunk_sizes, means,
                    marker='o', linewidth=4, markersize=12,
                    color=COLORS[strategy], label=STRATEGY_LABELS[strategy],
                    alpha=0.9)

    # Labels and formatting
    if metric_name == 'vram_diff':
        ylabel = 'VRAM Difference (MB)'
        title = f'VRAM Difference vs Chunk Size ({STRATEGY_LABELS[strategy]})'
    elif metric_name == 'vram_sum':
        ylabel = 'Total VRAM (MB)'
        title = f'Total VRAM vs Chunk Size ({STRATEGY_LABELS[strategy]})'
    else:
        ylabel = metric_name
        title = f'{metric_name} vs Chunk Size ({STRATEGY_LABELS[strategy]})'

    plt.xlabel('Chunk Size', fontsize=24, fontweight='bold')
    plt.ylabel(ylabel, fontsize=24, fontweight='bold')
    plt.xticks(chunk_sizes, [str(c) for c in chunk_sizes], fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=22)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = Path("plots") / f"fp_plot_{metric_name}_vs_chunk_{strategy}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_ablation(df, metric_name, chunk_size=4096, mbs_values=[1, 2, 4]):
    """Plot PP ablation study: metric vs split for different mbs values."""
    if df is None:
        return

    plt.figure(figsize=(10, 6))

    # For each mbs value, plot a line
    for mbs in mbs_values:
        splits = []
        means = []
        stds = []

        # Filter by chunk_size
        df_filtered = df[df['Chunk_Size'] == chunk_size]

        for _, row in df_filtered.iterrows():
            split = row['Split']
            col = f'mbs={mbs}'
            if col in row.index:
                mean, std = parse_metric_value(row[col])
                if mean is not None:
                    splits.append(split)
                    means.append(mean)
                    stds.append(std if std is not None else 0)

        # Plot line with error bars (solid line, different color shades)
        if splits:
            plt.errorbar(splits, means, yerr=stds,
                        marker='o', linewidth=4, markersize=12,
                        color=MBS_COLORS[mbs], linestyle='-',
                        label=MBS_LABELS[mbs],
                        capsize=10, capthick=4, elinewidth=4, alpha=0.9)

    # Labels and formatting
    if metric_name == 'throughput':
        ylabel = 'Throughput per Device (tok/s)'
        title = f'Throughput per Device vs Split (Chunk Size = {chunk_size})'
    elif metric_name == 'latency':
        ylabel = 'Normalized Latency (s)'
        title = f'Latency vs Split (Chunk Size = {chunk_size})'
    elif metric_name == 'vram_max':
        ylabel = 'Peak VRAM Usage (MB)'
        title = f'Peak VRAM Usage vs Split (Chunk Size = {chunk_size})'
    elif metric_name == 'vram_diff':
        ylabel = 'VRAM Difference (MB)'
        title = f'VRAM Difference vs Split (Chunk Size = {chunk_size})'
    elif metric_name == 'vram_sum':
        ylabel = 'Total VRAM (MB)'
        title = f'Total VRAM vs Split (Chunk Size = {chunk_size})'
    else:
        ylabel = metric_name
        title = f'{metric_name} vs Split (Chunk Size = {chunk_size})'

    plt.xlabel('Split', fontsize=24, fontweight='bold')
    plt.ylabel(ylabel, fontsize=24, fontweight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=22)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = Path("plots") / f"fp_plot_{metric_name}_vs_split_ablation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Generate all FP plots."""
    print("Generating FP plots...")

    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Load data
    print("\nLoading PP ablation data...")
    df_pp_thr, df_pp_lat, df_pp_vram_max, df_pp_vram_diff, df_pp_vram_sum = load_pp_data()

    print("Loading FSDP+DTensor data...")
    df_fsdp_thr, df_fsdp_lat, df_fsdp_vram_max, df_fsdp_vram_diff, df_fsdp_vram_sum = load_fsdp_data()

    if df_pp_thr is None or df_fsdp_thr is None:
        print("Error: Failed to load data!")
        return

    # Group 1: Comparison plots (5)
    print("\n" + "="*60)
    print("Generating comparison plots (PP vs FSDP+DTensor)...")
    print("="*60)

    plot_comparison(df_pp_thr, df_fsdp_thr, 'throughput', pp_split=16, pp_mbs=4)
    plot_comparison(df_pp_lat, df_fsdp_lat, 'latency', pp_split=16, pp_mbs=4)
    plot_comparison(df_pp_vram_max, df_fsdp_vram_max, 'vram_max', pp_split=16, pp_mbs=4)
    plot_comparison(df_pp_vram_diff, df_fsdp_vram_diff, 'vram_diff', pp_split=16, pp_mbs=4)
    plot_comparison(df_pp_vram_sum, df_fsdp_vram_sum, 'vram_sum', pp_split=16, pp_mbs=4)

    # Group 2: PP Ablation plots (5)
    print("\n" + "="*60)
    print("Generating PP ablation plots...")
    print("="*60)

    plot_ablation(df_pp_thr, 'throughput', chunk_size=4096, mbs_values=[1, 2, 4])
    plot_ablation(df_pp_lat, 'latency', chunk_size=4096, mbs_values=[1, 2, 4])
    plot_ablation(df_pp_vram_max, 'vram_max', chunk_size=4096, mbs_values=[1, 2, 4])
    plot_ablation(df_pp_vram_diff, 'vram_diff', chunk_size=4096, mbs_values=[1, 2, 4])
    plot_ablation(df_pp_vram_sum, 'vram_sum', chunk_size=4096, mbs_values=[1, 2, 4])

    print("\n" + "="*60)
    print("All FP plots generated successfully!")
    print("="*60)
    print("\nColor scheme:")
    print(f"  - PP: Red ({COLORS['pp']})")
    print(f"  - FSDP+DTensor: Green ({COLORS['fsdp_dtensor']})")
    print("\nAblation plot colors (light to dark red):")
    print(f"  - mbs=1: Light red ({MBS_COLORS[1]})")
    print(f"  - mbs=2: Medium red ({MBS_COLORS[2]})")
    print(f"  - mbs=4: Dark red ({MBS_COLORS[4]})")
    print("\nError bars represent standard deviation (where available)")


if __name__ == "__main__":
    main()
