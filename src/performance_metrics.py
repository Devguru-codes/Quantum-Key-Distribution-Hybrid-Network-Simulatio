"""
Performance metrics calculation and profiling for QKD network simulation.
Tracks key rates, QBER, latency, throughput, and other performance indicators.
"""

import time
import numpy as np
import cProfile
import pstats
import io
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict
import threading

# Optional psutil import with fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not available - system monitoring disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system and simulation performance."""

    def __init__(self, config: Dict):
        self.config = config
        self.start_time = time.time()
        self.metrics_history = defaultdict(list)
        self.current_metrics = {}
        self.profiling_enabled = config.get('enable_profiling', True)
        self.profiler = cProfile.Profile() if self.profiling_enabled else None

    def start_profiling(self):
        """Start performance profiling."""
        if self.profiler:
            self.profiler.enable()
            logger.debug("Performance profiling started")

    def stop_profiling(self) -> Optional[str]:
        """Stop profiling and return statistics."""
        if self.profiler:
            self.profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
            ps.print_stats()
            profiling_stats = s.getvalue()
            logger.debug("Performance profiling stopped")
            return profiling_stats
        return None

    def record_metric(self, name: str, value: Any, timestamp: Optional[float] = None):
        """Record a performance metric."""
        if timestamp is None:
            timestamp = time.time()

        self.metrics_history[name].append((timestamp, value))
        self.current_metrics[name] = value

        # Keep history manageable (last 1000 entries per metric)
        if len(self.metrics_history[name]) > 1000:
            self.metrics_history[name] = self.metrics_history[name][-1000:]

    def get_metric_history(self, name: str, time_window: Optional[float] = None) -> List[Tuple[float, Any]]:
        """Get historical data for a metric within a time window."""
        if name not in self.metrics_history:
            return []

        history = self.metrics_history[name]

        if time_window is None:
            return history

        current_time = time.time()
        cutoff_time = current_time - time_window

        return [(t, v) for t, v in history if t >= cutoff_time]

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current values of all metrics."""
        return dict(self.current_metrics)

    def calculate_rate(self, name: str, time_window: float = 60.0) -> float:
        """Calculate rate of change for a metric over a time window."""
        history = self.get_metric_history(name, time_window)

        if len(history) < 2:
            return 0.0

        # Calculate rate as (final - initial) / time_span
        initial_time, initial_value = history[0]
        final_time, final_value = history[-1]

        time_span = final_time - initial_time

        if time_span == 0:
            return 0.0

        if isinstance(final_value, (int, float)) and isinstance(initial_value, (int, float)):
            return (final_value - initial_value) / time_span
        else:
            return 0.0

    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        stats = {'simulation_time': time.time() - self.start_time}

        if PSUTIL_AVAILABLE:
            try:
                stats.update({
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_used_mb': psutil.virtual_memory().used / (1024 * 1024),
                    'disk_usage_percent': psutil.disk_usage('/').percent,
                })
            except Exception as e:
                logger.warning(f"Failed to get system stats: {e}")
                stats.update({
                    'cpu_percent': 0.0,
                    'memory_percent': 0.0,
                    'memory_used_mb': 0.0,
                    'disk_usage_percent': 0.0,
                })
        else:
            # Provide default values when psutil is not available
            stats.update({
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_used_mb': 0.0,
                'disk_usage_percent': 0.0,
            })

        return stats


class QKDPerformanceMetrics:
    """QKD-specific performance metrics."""

    def __init__(self, config: Dict):
        self.config = config
        self.monitor = PerformanceMonitor(config)
        self.protocol_stats = defaultdict(dict)
        self.network_stats = {}

    def record_qkd_metrics(self, protocol: str, results: Dict):
        """Record QKD protocol performance metrics."""
        timestamp = time.time()

        # Key generation metrics
        if 'key_rate' in results:
            self.monitor.record_metric(f'{protocol}_key_rate', results['key_rate'], timestamp)

        if 'final_bits' in results:
            self.monitor.record_metric(f'{protocol}_key_length', results['final_bits'], timestamp)

        # Error metrics
        if 'qber' in results:
            self.monitor.record_metric(f'{protocol}_qber', results['qber'], timestamp)

        # Efficiency metrics
        if 'efficiency' in results:
            self.monitor.record_metric(f'{protocol}_efficiency', results['efficiency'], timestamp)

        # Raw bit metrics
        if 'raw_bits' in results:
            self.monitor.record_metric(f'{protocol}_raw_bits', results['raw_bits'], timestamp)

        if 'sifted_bits' in results:
            self.monitor.record_metric(f'{protocol}_sifted_bits', results['sifted_bits'], timestamp)

        # Store protocol results
        self.protocol_stats[protocol] = results

        logger.debug(f"Recorded QKD metrics for {protocol}: {results}")

    def record_network_metrics(self, routing_results: Dict):
        """Record network-level performance metrics."""
        timestamp = time.time()

        if 'total_delay' in routing_results:
            self.monitor.record_metric('network_latency', routing_results['total_delay'], timestamp)

        if 'throughput' in routing_results:
            self.monitor.record_metric('network_throughput', routing_results['throughput'], timestamp)

        if 'hops' in routing_results:
            self.monitor.record_metric('network_hops', routing_results['hops'], timestamp)

        if 'total_distance' in routing_results:
            self.monitor.record_metric('network_distance', routing_results['total_distance'], timestamp)

        if 'routing_time' in routing_results:
            self.monitor.record_metric('routing_overhead', routing_results['routing_time'], timestamp)

        # Success rate
        success = 1 if routing_results.get('success', False) else 0
        self.monitor.record_metric('routing_success_rate', success, timestamp)

    def record_communication_metrics(self, comm_results: Dict):
        """Record communication performance metrics."""
        timestamp = time.time()

        if 'total_messages_sent' in comm_results:
            self.monitor.record_metric('messages_sent_total', comm_results['total_messages_sent'], timestamp)

        if 'total_messages_received' in comm_results:
            self.monitor.record_metric('messages_received_total', comm_results['total_messages_received'], timestamp)

        if 'total_bytes_sent' in comm_results:
            self.monitor.record_metric('bytes_sent_total', comm_results['total_bytes_sent'], timestamp)

        if 'total_bytes_received' in comm_results:
            self.monitor.record_metric('bytes_received_total', comm_results['total_bytes_received'], timestamp)

    def calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate performance metrics."""
        metrics = {}

        # QKD Protocol Metrics
        for protocol in ['bb84', 'e91']:
            key_rate_history = self.monitor.get_metric_history(f'{protocol}_key_rate', 300)  # Last 5 minutes
            if key_rate_history:
                rates = [v for t, v in key_rate_history if isinstance(v, (int, float))]
                if rates:
                    metrics[f'{protocol}_avg_key_rate'] = np.mean(rates)
                    metrics[f'{protocol}_max_key_rate'] = np.max(rates)
                    metrics[f'{protocol}_key_rate_std'] = np.std(rates)

            qber_history = self.monitor.get_metric_history(f'{protocol}_qber', 300)
            if qber_history:
                qbers = [v for t, v in qber_history if isinstance(v, (int, float))]
                if qbers:
                    metrics[f'{protocol}_avg_qber'] = np.mean(qbers)
                    metrics[f'{protocol}_max_qber'] = np.max(qbers)

        # Network Metrics
        latency_history = self.monitor.get_metric_history('network_latency', 300)
        if latency_history:
            latencies = [v for t, v in latency_history if isinstance(v, (int, float))]
            if latencies:
                metrics['avg_network_latency'] = np.mean(latencies)
                metrics['min_network_latency'] = np.min(latencies)
                metrics['max_network_latency'] = np.max(latencies)
                metrics['latency_std'] = np.std(latencies)

        throughput_history = self.monitor.get_metric_history('network_throughput', 300)
        if throughput_history:
            throughputs = [v for t, v in throughput_history if isinstance(v, (int, float))]
            if throughputs:
                metrics['avg_network_throughput'] = np.mean(throughputs)
                metrics['max_network_throughput'] = np.max(throughputs)

        # Success Rate
        success_history = self.monitor.get_metric_history('routing_success_rate', 300)
        if success_history:
            successes = [v for t, v in success_history if isinstance(v, (int, float))]
            if successes:
                metrics['routing_success_rate'] = np.mean(successes)

        # System Metrics
        system_stats = self.monitor.get_system_stats()
        metrics.update({f'system_{k}': v for k, v in system_stats.items()})

        return metrics

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        current_metrics = self.monitor.get_current_metrics()
        aggregate_metrics = self.calculate_aggregate_metrics()

        # Calculate rates
        rates = {}
        for metric_name in ['bb84_key_rate', 'e91_key_rate', 'network_throughput']:
            rate = self.monitor.calculate_rate(metric_name, 60)  # per second
            rates[f'{metric_name}_rate'] = rate

        report = {
            'timestamp': time.time(),
            'current_metrics': current_metrics,
            'aggregate_metrics': aggregate_metrics,
            'rates': rates,
            'protocol_stats': dict(self.protocol_stats),
            'network_stats': self.network_stats,
            'system_stats': self.monitor.get_system_stats()
        }

        return report

    def export_metrics_csv(self, filename: str):
        """Export metrics history to CSV file."""
        import csv

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['timestamp', 'metric', 'value'])

            # Write data
            for metric_name, history in self.monitor.metrics_history.items():
                for timestamp, value in history:
                    writer.writerow([timestamp, metric_name, value])

        logger.info(f"Metrics exported to {filename}")


class BenchmarkComparator:
    """Compare performance against benchmarks and classical systems."""

    def __init__(self, config: Dict):
        self.config = config
        self.classical_baselines = {
            'rsa_key_exchange_time': 0.1,  # seconds
            'aes_encryption_throughput': 1000000,  # bytes/second
            'classical_error_rate': 0.0  # Perfect classical channel
        }

    def compare_qkd_vs_classical(self, qkd_metrics: Dict) -> Dict[str, Any]:
        """Compare QKD performance against classical cryptography."""
        comparisons = {}

        # Key exchange time comparison
        qkd_key_rate = qkd_metrics.get('bb84_avg_key_rate', 0)
        if qkd_key_rate > 0:
            qkd_key_time = 256 / qkd_key_rate  # Time to generate 256-bit key
            classical_key_time = self.classical_baselines['rsa_key_exchange_time']

            comparisons['key_exchange_time_ratio'] = qkd_key_time / classical_key_time
            comparisons['qkd_advantage_key_exchange'] = classical_key_time > qkd_key_time

        # Throughput comparison
        qkd_throughput = qkd_metrics.get('avg_network_throughput', 0)
        classical_throughput = self.classical_baselines['aes_encryption_throughput']

        comparisons['throughput_ratio'] = qkd_throughput / classical_throughput if classical_throughput > 0 else 0
        comparisons['qkd_advantage_throughput'] = qkd_throughput > classical_throughput

        # Security comparison
        qkd_qber = qkd_metrics.get('bb84_avg_qber', 0.05)
        classical_error = self.classical_baselines['classical_error_rate']

        comparisons['security_advantage'] = qkd_qber < 0.11  # QBER threshold for security
        comparisons['error_rate_comparison'] = qkd_qber - classical_error

        return comparisons

    def generate_comparison_report(self, qkd_metrics: Dict) -> str:
        """Generate human-readable comparison report."""
        comparisons = self.compare_qkd_vs_classical(qkd_metrics)

        report = []
        report.append("=" * 60)
        report.append("QKD vs CLASSICAL CRYPTOGRAPHY COMPARISON")
        report.append("=" * 60)

        report.append("\nKey Exchange Performance:")
        if 'key_exchange_time_ratio' in comparisons:
            ratio = comparisons['key_exchange_time_ratio']
            if ratio < 1:
                report.append(f"✓ QKD is {1/ratio:.2f}x faster for key exchange")
            else:
                report.append(f"⚠ QKD is {ratio:.2f}x slower for key exchange")
        report.append("\nThroughput Comparison:")
        if 'throughput_ratio' in comparisons:
            ratio = comparisons['throughput_ratio']
            if ratio > 1:
                report.append(f"✓ QKD throughput is {ratio:.1f}x higher")
            else:
                report.append(f"⚠ QKD throughput is {1/ratio if ratio > 0 else 0:.1f}x lower")
        report.append("\nSecurity Advantages:")
        if comparisons.get('security_advantage', False):
            report.append("✓ QKD provides information-theoretic security")
            report.append("✓ Resistance to quantum computing attacks")
        else:
            report.append("⚠ QBER too high - security may be compromised")

        report.append("=" * 60)
        return "\n".join(report)


def profile_function(func):
    """Decorator to profile a function's performance."""
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        profiler.disable()

        execution_time = end_time - start_time

        # Get profiling stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions

        logger.info(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        logger.debug(f"Profiling stats for {func.__name__}:\n{s.getvalue()}")

        return result

    return wrapper


class RealTimeMonitor:
    """Real-time performance monitoring with threading."""

    def __init__(self, metrics_collector: QKDPerformanceMetrics, update_interval: float = 1.0):
        self.metrics = metrics_collector
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start real-time monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Real-time monitoring started")

    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Real-time monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Record system stats
                system_stats = self.metrics.monitor.get_system_stats()
                for key, value in system_stats.items():
                    self.metrics.monitor.record_metric(f'system_{key}', value)

                # Log current performance
                current_metrics = self.metrics.monitor.get_current_metrics()
                if current_metrics:
                    logger.debug(f"Current metrics: {current_metrics}")

                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)


# Example usage
def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    config = {'enable_profiling': True}

    # Create performance monitor
    perf_metrics = QKDPerformanceMetrics(config)

    # Start profiling
    perf_metrics.monitor.start_profiling()

    # Simulate some QKD operations
    bb84_results = {
        'key_rate': 1500,
        'qber': 0.03,
        'final_bits': 256,
        'efficiency': 0.8
    }

    perf_metrics.record_qkd_metrics('bb84', bb84_results)

    # Simulate network operations
    network_results = {
        'total_delay': 0.001,
        'throughput': 100000,
        'hops': 2,
        'total_distance': 20.0,
        'success': True
    }

    perf_metrics.record_network_metrics(network_results)

    # Stop profiling
    profiling_stats = perf_metrics.monitor.stop_profiling()

    # Generate report
    report = perf_metrics.get_performance_report()

    print("Performance Report:")
    print(f"Average BB84 Key Rate: {report['aggregate_metrics'].get('bb84_avg_key_rate', 'N/A')}")
    print(f"Average Network Latency: {report['aggregate_metrics'].get('avg_network_latency', 'N/A'):.6f} seconds")

    # Compare with classical
    comparator = BenchmarkComparator(config)
    comparison_report = comparator.generate_comparison_report(report['aggregate_metrics'])
    print("\n" + comparison_report)

    return report


if __name__ == "__main__":
    demonstrate_performance_monitoring()
