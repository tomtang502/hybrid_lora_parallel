import statistics
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PerformanceStats:
    """Data class to hold the calculated statistics."""
    min: float
    max: float
    mean: float
    std_dev: float
    count: int

class PerformanceMonitor:
    def __init__(self):
        self.latencies: List[float] = []
        self.throughputs: List[float] = []
        self.vram_history: List[float] = []
        
    def record_step(self, latency: float, throughput: float, vram: float):
        """
        Manually record metrics for a single step.
        
        Args:
            latency (float): Time taken for the step (seconds).
            throughput (float): Tokens per second (or items per second).
            vram (float): VRAM usage (MB/GB).
        """
        self.latencies.append(latency)
        self.throughputs.append(throughput)
        self.vram_history.append(vram)

    def reset(self):
        """Clears all recorded history."""
        self.latencies = []
        self.throughputs = []
        self.vram_history = []

    def _calculate_stats(self, data: List[float]) -> PerformanceStats:
        """Internal helper to calculate min, max, mean, std."""
        if not data:
            return PerformanceStats(0.0, 0.0, 0.0, 0.0, 0)
        
        return PerformanceStats(
            min=min(data),
            max=max(data),
            mean=statistics.mean(data),
            std_dev=statistics.stdev(data) if len(data) > 1 else 0.0,
            count=len(data)
        )

    def get_report(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing statistical summaries.
        """
        latency_stats = self._calculate_stats(self.latencies)
        tps_stats = self._calculate_stats(self.throughputs)
        vram_stats = self._calculate_stats(self.vram_history)

        return {
            "latency_seconds": {
                "min": f"{latency_stats.min:.4f}",
                "max": f"{latency_stats.max:.4f}",
                "mean": f"{latency_stats.mean:.4f}",
                "std": f"{latency_stats.std_dev:.4f}",
                "count": latency_stats.count
            },
            "throughput_tokens_per_sec": {
                "min": f"{tps_stats.min:.2f}",
                "max": f"{tps_stats.max:.2f}",
                "mean": f"{tps_stats.mean:.2f}",
                "std": f"{tps_stats.std_dev:.2f}"
            },
            "vram_usage_mb": {
                "max_peak": f"{vram_stats.max:.2f}",
                "current_mean": f"{vram_stats.mean:.2f}"
            }
        }
    
    def generate_report_string(self) -> str:
        """Generates the formatted report string."""
        report = self.get_report()
        lines = []
        
        lines.append("\n" + "="*50)
        lines.append("PERFORMANCE REPORT")
        lines.append("="*50)
        
        lines.append(f"{'Metric':<20} | {'Mean':<10} | {'Max':<10} | {'Min':<10} | {'StdDev':<10}")
        lines.append("-" * 70)
        
        l = report['latency_seconds']
        lines.append(f"{'Latency (s)':<20} | {l['mean']:<10} | {l['max']:<10} | {l['min']:<10} | {l['std']:<10}")
        
        t = report['throughput_tokens_per_sec']
        lines.append(f"{'Throughput (tok/s)':<20} | {t['mean']:<10} | {t['max']:<10} | {t['min']:<10} | {t['std']:<10}")
        
        v = report['vram_usage_mb']
        lines.append("-" * 70)
        lines.append(f"Peak VRAM Usage: {v['max_peak']} MB")
        lines.append(f"Total Samples:   {l['count']}")
        lines.append("="*50 + "\n")
        
        return "\n".join(lines)

    def print_summary(self):
        """Prints a readable table of the stats."""
        report = self.get_report()
        
        print("\n" + "="*50)
        print("PERFORMANCE REPORT")
        print("="*50)
        
        print(f"{'Metric':<20} | {'Mean':<10} | {'Max':<10} | {'Min':<10} | {'StdDev':<10}")
        print("-" * 70)
        
        l = report['latency_seconds']
        print(f"{'Latency (s)':<20} | {l['mean']:<10} | {l['max']:<10} | {l['min']:<10} | {l['std']:<10}")
        
        t = report['throughput_tokens_per_sec']
        print(f"{'Throughput (tok/s)':<20} | {t['mean']:<10} | {t['max']:<10} | {t['min']:<10} | {t['std']:<10}")
        
        v = report['vram_usage_mb']
        print("-" * 70)
        print(f"Peak VRAM Usage: {v['max_peak']} MB")
        print(f"Total Samples:   {l['count']}")
        print("="*50 + "\n")


# ==========================================
# Example Usage Simulation
# ==========================================
if __name__ == "__main__":
    import random
    
    monitor = PerformanceMonitor()
    print("Simulating manual recording...")

    # Simulate 10 runs
    for i in range(10):
        # User calculates these values externally
        sim_latency = random.uniform(0.1, 0.5)
        sim_tokens = random.randint(20, 100)
        sim_throughput = sim_tokens / sim_latency
        sim_vram = 4000 + random.uniform(-100, 100)
        
        # Pass them directly
        monitor.record_step(latency=sim_latency, throughput=sim_throughput, vram=sim_vram)
        print(f".", end="", flush=True)

    monitor.print_summary()