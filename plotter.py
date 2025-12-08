#!/usr/bin/env python3
"""Plotter for hybrid LoRA parallel training experiment results."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Color scheme for strategies
COLORS = {
    'ddp': '#1f77b4',           # Blue
    'fsdp': '#ff7f0e',          # Orange
    'fsdp_dtensor': '#2ca02c'   # Green
}

# Line styles for different n values
LINE_STYLES = {
    1: '-',      # Solid
    2: '--',     # Dashed
    4: ':'       # Dotted
}

STRATEGY_LABELS = {
    'ddp': 'DDP',
    'fsdp': 'FSDP',
    'fsdp_dtensor': 'FSDP+DTensor'
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


def load_data_from_csv():
    """Load all three data sections from CSV."""
    csv_file = Path("experiment_results.csv")
    if not csv_file.exists():
        print("Error: experiment_results.csv not found! Run analyzer.py first.")
        return None, None, None

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
        elif 'VRAM' in line:
            if current_section == 'latency':
                sections['latency'] = (section_start, i - 2)
            current_section = 'vram'
            section_start = i + 1

    sections['vram'] = (section_start, len(lines))

    # Parse each section
    df_thr = pd.read_csv(csv_file, skiprows=sections['throughput'][0],
                         nrows=sections['throughput'][1] - sections['throughput'][0])
    df_lat = pd.read_csv(csv_file, skiprows=sections['latency'][0],
                         nrows=sections['latency'][1] - sections['latency'][0])
    df_vram = pd.read_csv(csv_file, skiprows=sections['vram'][0],
                          nrows=sections['vram'][1] - sections['vram'][0])

    return df_thr, df_lat, df_vram


def plot_metric_vs_n(df, metric_name, chunk_size=4096):
    """Plot metric vs n for fixed chunk_size with error bars."""
    if df is None:
        return

    df_subset = df[df['Chunk_Size'] == chunk_size]

    plt.figure(figsize=(8, 6))

    for strategy in ['ddp', 'fsdp', 'fsdp_dtensor']:
        row = df_subset[df_subset['Strategy'] == strategy]
        if row.empty:
            continue

        row = row.iloc[0]
        ns = [1, 2, 4]
        means, stds = [], []

        for n in ns:
            col = f'n={n}'
            if col in row.index:
                mean, std = parse_metric_value(row[col])
                means.append(mean)
                stds.append(std if std is not None else 0)
            else:
                means.append(None)
                stds.append(None)

        # Filter out None values
        valid_data = [(n, m, s) for n, m, s in zip(ns, means, stds) if m is not None]
        if not valid_data:
            continue

        ns_valid, means_valid, stds_valid = zip(*valid_data)
        plt.errorbar(ns_valid, means_valid, yerr=stds_valid,
                    marker='o', linewidth=2, markersize=8,
                    color=COLORS[strategy], label=STRATEGY_LABELS[strategy],
                    capsize=5, capthick=2, elinewidth=2, alpha=0.9)

    ylabel = 'Throughput (tok/s)' if metric_name == 'throughput' else 'Latency (s)'
    title = f'{metric_name.capitalize()} vs Number of GPUs (Chunk Size = {chunk_size})'
    if metric_name == 'throughput':
        title = f'Throughput per Device vs Number of GPUs (Chunk Size = {chunk_size})'
    plt.xlabel('Number of GPUs (n)', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks([1, 2, 4])
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = Path("plots") / f"plot_{metric_name}_vs_n.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_metric_vs_chunk(df, metric_name, n):
    """Plot metric vs chunk_size for specific n with error bars."""
    if df is None:
        return

    col = f'n={n}'
    if col not in df.columns:
        print(f"Warning: No data for n={n}")
        return

    plt.figure(figsize=(10, 6))
    chunk_sizes = [1024, 2048, 4096, 8192, 16384]

    for strategy in ['ddp', 'fsdp', 'fsdp_dtensor']:
        strategy_data = df[df['Strategy'] == strategy]

        means, stds = [], []
        valid_chunks = []

        for chunk in chunk_sizes:
            row = strategy_data[strategy_data['Chunk_Size'] == chunk]
            if not row.empty:
                mean, std = parse_metric_value(row[col].iloc[0])
                if mean is not None:
                    valid_chunks.append(chunk)
                    means.append(mean)
                    stds.append(std if std is not None else 0)

        # Skip line if no valid data
        if not valid_chunks:
            continue

        plt.errorbar(valid_chunks, means, yerr=stds,
                    marker='o', linewidth=4, markersize=12,
                    color=COLORS[strategy], label=STRATEGY_LABELS[strategy],
                    capsize=10, capthick=4, elinewidth=4, alpha=0.9)

    ylabel = 'Throughput (tok/s)' if metric_name == 'throughput' else 'Latency (s)'
    title = f'{metric_name.capitalize()} vs Chunk Size (n = {n} GPU{"s" if n > 1 else ""})'
    if metric_name == 'throughput':
        title = f'Throughput per Device vs Chunk Size (n = {n} GPU{"s" if n > 1 else ""})'
    plt.xlabel('Chunk Size', fontsize=24, fontweight='bold')
    plt.ylabel(ylabel, fontsize=24, fontweight='bold')
    plt.title(title, fontsize=28, fontweight='bold')
    plt.xscale('log', base=2)
    plt.xticks(chunk_sizes, [str(c) for c in chunk_sizes], fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=22)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = Path("plots") / f"plot_{metric_name}_vs_chunk_n{n}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_vram_vs_chunk(df_vram):
    """Plot VRAM vs chunk_size with 9 lines (3 strategies × 3 n values).
    No error bars since VRAM is just max value."""
    if df_vram is None:
        return

    plt.figure(figsize=(12, 7))
    chunk_sizes = [1024, 2048, 4096, 8192, 16384]

    for strategy in ['ddp', 'fsdp', 'fsdp_dtensor']:
        for n in [1, 2, 4]:
            col = f'n={n}'
            if col not in df_vram.columns:
                continue

            strategy_data = df_vram[df_vram['Strategy'] == strategy]
            valid_chunks = []
            values = []

            for chunk in chunk_sizes:
                row = strategy_data[strategy_data['Chunk_Size'] == chunk]
                if not row.empty:
                    value, _ = parse_metric_value(row[col].iloc[0])
                    if value is not None:
                        valid_chunks.append(chunk)
                        values.append(value)

            # Skip line if no valid data
            if not valid_chunks:
                continue

            label = f'{STRATEGY_LABELS[strategy]}, n={n}'
            plt.plot(valid_chunks, values,
                    marker='o', linewidth=2.5, markersize=7,
                    color=COLORS[strategy], linestyle=LINE_STYLES[n],
                    label=label, alpha=0.9)

    plt.xlabel('Chunk Size', fontsize=12, fontweight='bold')
    plt.ylabel('Peak VRAM Usage (MB)', fontsize=12, fontweight='bold')
    plt.title('Peak VRAM Usage vs Chunk Size', fontsize=14, fontweight='bold')
    plt.xscale('log', base=2)
    plt.xticks(chunk_sizes, [str(c) for c in chunk_sizes])
    plt.legend(fontsize=9, ncol=2, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = Path("plots") / "plot_vram_vs_chunk.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Generate all plots."""
    print("Generating plots...")

    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Load data
    df_thr, df_lat, df_vram = load_data_from_csv()

    if df_thr is None:
        return

    # Throughput plots
    print("\nGenerating throughput plots...")
    plot_metric_vs_n(df_thr, 'throughput', chunk_size=4096)
    for n in [1, 2, 4]:
        plot_metric_vs_chunk(df_thr, 'throughput', n)

    # Latency plots
    print("\nGenerating latency plots...")
    plot_metric_vs_n(df_lat, 'latency', chunk_size=4096)
    for n in [1, 2, 4]:
        plot_metric_vs_chunk(df_lat, 'latency', n)

    # VRAM plot
    print("\nGenerating VRAM plot...")
    plot_vram_vs_chunk(df_vram)

    print("\n" + "="*60)
    print("All plots generated successfully!")
    print("="*60)
    print("\nColor scheme:")
    print(f"  - DDP: Blue ({COLORS['ddp']})")
    print(f"  - FSDP: Orange ({COLORS['fsdp']})")
    print(f"  - FSDP+DTensor: Green ({COLORS['fsdp_dtensor']})")
    print("\nLine styles (for VRAM plot):")
    print(f"  - n=1: Solid line")
    print(f"  - n=2: Dashed line")
    print(f"  - n=4: Dotted line")
    print("\nError bars represent standard deviation (where available)")


if __name__ == "__main__":
    main()
