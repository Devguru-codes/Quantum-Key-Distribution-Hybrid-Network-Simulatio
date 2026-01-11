"""
Visualization and plotting functions for QKD network simulation results.
Creates graphs for performance metrics, network topology, and comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class QKDVisualizer:
    """Visualization tools for QKD network simulations."""

    def __init__(self, output_dir: str = "results/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_qkd_metrics_over_time(self, metrics_history: Dict[str, List[Tuple[float, Any]]],
                                  save_path: Optional[str] = None):
        """Plot QKD metrics over time."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('QKD Protocol Metrics Over Time', fontsize=16, fontweight='bold')

            plot_count = 0

            # Key Rate
            if 'bb84_key_rate' in metrics_history and metrics_history['bb84_key_rate']:
                times, values = zip(*metrics_history['bb84_key_rate'])
                # Filter out non-numeric values
                valid_data = [(t, v) for t, v in zip(times, values) if isinstance(v, (int, float)) and not np.isnan(v)]
                if valid_data:
                    times, values = zip(*valid_data)
                    axes[0, 0].plot(times, values, 'b-', linewidth=2, marker='o', markersize=3)
                    axes[0, 0].set_title('BB84 Key Rate')
                    axes[0, 0].set_xlabel('Time (s)')
                    axes[0, 0].set_ylabel('Key Rate (bits/s)')
                    axes[0, 0].grid(True, alpha=0.3)
                    plot_count += 1

            # QBER
            if 'bb84_qber' in metrics_history and metrics_history['bb84_qber']:
                times, values = zip(*metrics_history['bb84_qber'])
                valid_data = [(t, v) for t, v in zip(times, values) if isinstance(v, (int, float)) and not np.isnan(v)]
                if valid_data:
                    times, values = zip(*valid_data)
                    axes[0, 1].plot(times, values, 'r-', linewidth=2, marker='s', markersize=3)
                    axes[0, 1].set_title('Quantum Bit Error Rate (QBER)')
                    axes[0, 1].set_xlabel('Time (s)')
                    axes[0, 1].set_ylabel('QBER')
                    axes[0, 1].grid(True, alpha=0.3)
                    plot_count += 1

            # Efficiency
            if 'bb84_efficiency' in metrics_history and metrics_history['bb84_efficiency']:
                times, values = zip(*metrics_history['bb84_efficiency'])
                valid_data = [(t, v) for t, v in zip(times, values) if isinstance(v, (int, float)) and not np.isnan(v)]
                if valid_data:
                    times, values = zip(*valid_data)
                    axes[1, 0].plot(times, values, 'g-', linewidth=2, marker='^', markersize=3)
                    axes[1, 0].set_title('Protocol Efficiency')
                    axes[1, 0].set_xlabel('Time (s)')
                    axes[1, 0].set_ylabel('Efficiency')
                    axes[1, 0].grid(True, alpha=0.3)
                    plot_count += 1

            # Key Length
            if 'bb84_key_length' in metrics_history and metrics_history['bb84_key_length']:
                times, values = zip(*metrics_history['bb84_key_length'])
                valid_data = [(t, v) for t, v in zip(times, values) if isinstance(v, (int, float)) and not np.isnan(v)]
                if valid_data:
                    times, values = zip(*valid_data)
                    axes[1, 1].plot(times, values, 'purple', linewidth=2, marker='d', markersize=3)
                    axes[1, 1].set_title('Generated Key Length')
                    axes[1, 1].set_xlabel('Time (s)')
                    axes[1, 1].set_ylabel('Key Length (bits)')
                    axes[1, 1].grid(True, alpha=0.3)
                    plot_count += 1

            # If no plots were created, show a message
            if plot_count == 0:
                fig.text(0.5, 0.5, 'No QKD metrics data available for plotting',
                        ha='center', va='center', fontsize=14)
                logger.warning("No valid QKD metrics data found for plotting")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"QKD metrics plot saved to {save_path}")

            plt.show()
            logger.info(f"QKD metrics plot displayed with {plot_count} subplots")

        except Exception as e:
            logger.error(f"Error creating QKD metrics plot: {e}")
            # Create a simple fallback plot
            try:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, f'QKD Metrics Plot Error:\n{str(e)}',
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
                plt.title('QKD Metrics Plot - Error')
                plt.axis('off')
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.show()
            except Exception as e2:
                logger.error(f"Fallback plot also failed: {e2}")

    def plot_network_performance(self, network_metrics: Dict[str, Any],
                                save_path: Optional[str] = None):
        """Plot network performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Network Performance Metrics', fontsize=16, fontweight='bold')

        # Latency distribution
        if 'latency_history' in network_metrics:
            latencies = network_metrics['latency_history']
            axes[0, 0].hist(latencies, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Message Latency Distribution')
            axes[0, 0].set_xlabel('Latency (seconds)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)

        # Throughput over time
        if 'throughput_history' in network_metrics:
            times, throughputs = zip(*network_metrics['throughput_history'])
            axes[0, 1].plot(times, throughputs, 'green', linewidth=2, marker='o', markersize=3)
            axes[0, 1].set_title('Network Throughput Over Time')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Throughput (bytes/s)')
            axes[0, 1].grid(True, alpha=0.3)

        # Success rate
        if 'success_rate_history' in network_metrics:
            times, success_rates = zip(*network_metrics['success_rate_history'])
            axes[1, 0].plot(times, success_rates, 'orange', linewidth=2, marker='s', markersize=3)
            axes[1, 0].set_title('Message Success Rate')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].grid(True, alpha=0.3)

        # Hops distribution
        if 'hops_history' in network_metrics:
            hops = network_metrics['hops_history']
            unique_hops, counts = np.unique(hops, return_counts=True)
            axes[1, 1].bar(unique_hops, counts, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 1].set_title('Path Length Distribution')
            axes[1, 1].set_xlabel('Number of Hops')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Network performance plot saved to {save_path}")

        plt.show()

    def plot_qkd_vs_classical_comparison(self, qkd_metrics: Dict[str, Any],
                                       classical_metrics: Dict[str, Any],
                                       save_path: Optional[str] = None):
        """Plot comparison between QKD and classical networks."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('QKD vs Classical Network Comparison', fontsize=16, fontweight='bold')

        # Setup time comparison
        labels = ['QKD', 'Classical']
        setup_times = [
            qkd_metrics.get('setup_time', 0),
            classical_metrics.get('setup_time', 0)
        ]
        axes[0, 0].bar(labels, setup_times, color=['blue', 'red'], alpha=0.7)
        axes[0, 0].set_title('Network Setup Time')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].grid(True, alpha=0.3)

        # Throughput comparison
        throughputs = [
            qkd_metrics.get('throughput_bytes_per_sec', 0),
            classical_metrics.get('throughput_bytes_per_sec', 0)
        ]
        axes[0, 1].bar(labels, throughputs, color=['blue', 'red'], alpha=0.7)
        axes[0, 1].set_title('Network Throughput')
        axes[0, 1].set_ylabel('Throughput (bytes/second)')
        axes[0, 1].grid(True, alpha=0.3)

        # Latency comparison
        latencies = [
            qkd_metrics.get('avg_network_latency', 0),
            classical_metrics.get('avg_total_latency', 0)
        ]
        axes[1, 0].bar(labels, latencies, color=['blue', 'red'], alpha=0.7)
        axes[1, 0].set_title('Average Message Latency')
        axes[1, 0].set_ylabel('Latency (seconds)')
        axes[1, 0].grid(True, alpha=0.3)

        # Security metrics
        security_labels = ['QBER', 'Error Rate']
        security_values_qkd = [
            qkd_metrics.get('bb84_avg_qber', 0),
            0  # QKD has no classical error rate
        ]
        security_values_classical = [
            0,  # Classical has no QBER
            classical_metrics.get('expansion_ratio', 0)
        ]

        x = np.arange(len(security_labels))
        width = 0.35

        axes[1, 1].bar(x - width/2, security_values_qkd, width, label='QKD', color='blue', alpha=0.7)
        axes[1, 1].bar(x + width/2, security_values_classical, width, label='Classical', color='red', alpha=0.7)
        axes[1, 1].set_title('Security Metrics')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(security_labels)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")

        plt.show()

    def plot_attack_analysis(self, attack_results: Dict[str, Any],
                           save_path: Optional[str] = None):
        """Plot attack analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Attack Analysis Results', fontsize=16, fontweight='bold')

        # QBER with and without attack
        qber_values = [
            attack_results.get('baseline_qber', 0.05),
            attack_results.get('attacked_qber', 0.15)
        ]
        labels = ['No Attack', 'Under Attack']

        axes[0, 0].bar(labels, qber_values, color=['green', 'red'], alpha=0.7)
        axes[0, 0].set_title('QBER Comparison')
        axes[0, 0].set_ylabel('QBER')
        axes[0, 0].grid(True, alpha=0.3)

        # Key rate impact
        key_rates = [
            attack_results.get('baseline_key_rate', 1000),
            attack_results.get('attacked_key_rate', 500)
        ]
        axes[0, 1].bar(labels, key_rates, color=['green', 'red'], alpha=0.7)
        axes[0, 1].set_title('Key Rate Impact')
        axes[0, 1].set_ylabel('Key Rate (bits/s)')
        axes[0, 1].grid(True, alpha=0.3)

        # Detection confidence
        if 'detection_results' in attack_results:
            detection = attack_results['detection_results']
            detection_labels = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
            detection_values = [
                detection.get('true_positive', 0),
                detection.get('false_positive', 0),
                detection.get('true_negative', 0),
                detection.get('false_negative', 0)
            ]

            axes[1, 0].bar(detection_labels, detection_values, color='orange', alpha=0.7)
            axes[1, 0].set_title('Attack Detection Performance')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

        # Information leakage
        if 'eve_information' in attack_results:
            eve_info = attack_results['eve_information']
            axes[1, 1].bar(['Eve Information', 'Remaining Security'],
                          [eve_info, 1-eve_info], color=['red', 'green'], alpha=0.7)
            axes[1, 1].set_title('Information Leakage Analysis')
            axes[1, 1].set_ylabel('Fraction')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attack analysis plot saved to {save_path}")

        plt.show()

    def create_performance_dashboard(self, all_metrics: Dict[str, Any],
                                   save_path: Optional[str] = None):
        """Create comprehensive performance dashboard."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle('QKD Network Performance Dashboard', fontsize=16, fontweight='bold')

        # QKD Protocol Metrics
        protocols = ['bb84', 'e91']
        key_rates = [all_metrics.get(f'{p}_avg_key_rate', 0) for p in protocols]
        qbers = [all_metrics.get(f'{p}_avg_qber', 0) for p in protocols]

        x = np.arange(len(protocols))
        width = 0.35

        axes[0, 0].bar(x - width/2, key_rates, width, label='Key Rate', color='blue', alpha=0.7)
        axes[0, 0].bar(x + width/2, qbers, width, label='QBER', color='red', alpha=0.7)
        axes[0, 0].set_title('QKD Protocol Performance')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(['BB84', 'E91'])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Network Metrics
        network_metrics = ['avg_network_latency', 'avg_network_throughput', 'routing_success_rate']
        network_values = [all_metrics.get(m, 0) for m in network_metrics]
        network_labels = ['Latency', 'Throughput', 'Success Rate']

        axes[0, 1].bar(network_labels, network_values, color='green', alpha=0.7)
        axes[0, 1].set_title('Network Performance')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # System Resources
        system_metrics = ['system_cpu_percent', 'system_memory_percent', 'system_simulation_time']
        system_values = [all_metrics.get(m, 0) for m in system_metrics]
        system_labels = ['CPU %', 'Memory %', 'Sim Time']

        axes[0, 2].bar(system_labels, system_values, color='purple', alpha=0.7)
        axes[0, 2].set_title('System Resources')
        axes[0, 2].grid(True, alpha=0.3)

        # Time series plots
        if 'metrics_history' in all_metrics:
            history = all_metrics['metrics_history']

            # Key rate over time
            if 'bb84_key_rate' in history:
                times, values = zip(*history['bb84_key_rate'])
                axes[1, 0].plot(times, values, 'b-', linewidth=2)
                axes[1, 0].set_title('Key Rate Over Time')
                axes[1, 0].set_xlabel('Time (s)')
                axes[1, 0].set_ylabel('Key Rate')
                axes[1, 0].grid(True, alpha=0.3)

            # QBER over time
            if 'bb84_qber' in history:
                times, values = zip(*history['bb84_qber'])
                axes[1, 1].plot(times, values, 'r-', linewidth=2)
                axes[1, 1].set_title('QBER Over Time')
                axes[1, 1].set_xlabel('Time (s)')
                axes[1, 1].set_ylabel('QBER')
                axes[1, 1].grid(True, alpha=0.3)

            # Throughput over time
            if 'network_throughput' in history:
                times, values = zip(*history['network_throughput'])
                axes[1, 2].plot(times, values, 'g-', linewidth=2)
                axes[1, 2].set_title('Throughput Over Time')
                axes[1, 2].set_xlabel('Time (s)')
                axes[1, 2].set_ylabel('Throughput')
                axes[1, 2].grid(True, alpha=0.3)

        # Performance summary
        summary_text = ".2f"".2f"".2f"".2f"f"""
Performance Summary:
• Avg Key Rate: {all_metrics.get('bb84_avg_key_rate', 0):.1f} bits/s
• Avg QBER: {all_metrics.get('bb84_avg_qber', 0):.4f}
• Network Latency: {all_metrics.get('avg_network_latency', 0):.6f} s
• Success Rate: {all_metrics.get('routing_success_rate', 0):.2%}
• CPU Usage: {all_metrics.get('system_cpu_percent', 0):.1f}%
"""

        axes[2, 0].text(0.1, 0.5, summary_text, transform=axes[2, 0].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        axes[2, 0].set_title('Performance Summary')
        axes[2, 0].set_xlim(0, 1)
        axes[2, 0].set_ylim(0, 1)
        axes[2, 0].axis('off')

        # Leave other subplots empty for now
        axes[2, 1].axis('off')
        axes[2, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance dashboard saved to {save_path}")

        plt.show()

    def export_all_plots(self, all_metrics: Dict[str, Any], base_filename: str = "qkd_results"):
        """Export all plots to files."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Create plots
        self.plot_qkd_metrics_over_time(
            all_metrics.get('metrics_history', {}),
            f"{self.output_dir}/{base_filename}_qkd_metrics_{timestamp}.png"
        )

        # Prepare network metrics data for plotting
        network_metrics = all_metrics.get('metrics_history', {})
        prepared_network_metrics = {}

        # Convert individual metric histories to expected format
        if 'network_latency' in network_metrics:
            # Extract just the values, not timestamps
            prepared_network_metrics['latency_history'] = [v for t, v in network_metrics['network_latency'] if isinstance(v, (int, float))]

        if 'network_throughput' in network_metrics:
            # Keep as (time, value) tuples
            prepared_network_metrics['throughput_history'] = [(t, v) for t, v in network_metrics['network_throughput'] if isinstance(v, (int, float))]

        if 'routing_success_rate' in network_metrics:
            # Keep as (time, value) tuples
            prepared_network_metrics['success_rate_history'] = [(t, v) for t, v in network_metrics['routing_success_rate'] if isinstance(v, (int, float))]

        if 'network_hops' in network_metrics:
            # Extract just the values, not timestamps
            prepared_network_metrics['hops_history'] = [v for t, v in network_metrics['network_hops'] if isinstance(v, (int, float))]

        self.plot_network_performance(
            prepared_network_metrics,
            f"{self.output_dir}/{base_filename}_network_perf_{timestamp}.png"
        )

        self.create_performance_dashboard(
            all_metrics,
            f"{self.output_dir}/{base_filename}_dashboard_{timestamp}.png"
        )

        logger.info(f"All plots exported to {self.output_dir}")


def create_sample_plots():
    """Create sample plots for demonstration."""
    visualizer = QKDVisualizer()

    # Sample QKD metrics history
    sample_history = {
        'bb84_key_rate': [(i, 1000 + np.random.normal(0, 100)) for i in range(100)],
        'bb84_qber': [(i, 0.05 + np.random.normal(0, 0.01)) for i in range(100)],
        'bb84_efficiency': [(i, 0.8 + np.random.normal(0, 0.05)) for i in range(100)],
        'bb84_key_length': [(i, 256 + np.random.normal(0, 10)) for i in range(100)]
    }

    # Sample network metrics
    sample_network = {
        'latency_history': np.random.exponential(0.001, 1000),
        'throughput_history': [(i, 50000 + np.random.normal(0, 5000)) for i in range(100)],
        'success_rate_history': [(i, 0.95 + np.random.normal(0, 0.02)) for i in range(100)],
        'hops_history': np.random.randint(1, 5, 1000)
    }

    # Create plots
    visualizer.plot_qkd_metrics_over_time(sample_history)
    visualizer.plot_network_performance(sample_network)

    print("Sample plots created successfully!")


if __name__ == "__main__":
    create_sample_plots()
