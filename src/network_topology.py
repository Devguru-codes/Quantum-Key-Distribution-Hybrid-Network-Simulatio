"""
Network topology implementation for QKD hybrid communication networks.
Includes node definitions, quantum/classical channels, and NetworkX visualization.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from qunetsim.components import Host, Network
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumNetworkNode:
    """Represents a node in the quantum network."""

    def __init__(self, node_id: str, node_type: str, position: Tuple[float, float] = None):
        self.node_id = node_id
        self.node_type = node_type  # 'source', 'destination', 'intermediate', 'repeater'
        self.position = position or (0, 0)
        self.host = None
        self.keys = {}  # Keys shared with other nodes
        self.routing_table = {}

    def __repr__(self):
        return f"Node({self.node_id}, {self.node_type})"


class QuantumChannel:
    """Represents a quantum communication channel."""

    def __init__(self, from_node: str, to_node: str, distance: float, loss_rate: float = 0.1):
        self.from_node = from_node
        self.to_node = to_node
        self.distance = distance
        self.loss_rate_db_per_km = loss_rate
        self.total_loss_db = distance * loss_rate
        self.transmission_efficiency = 10 ** (-self.total_loss_db / 10)

    def get_channel_quality(self) -> float:
        """Get channel transmission efficiency."""
        return self.transmission_efficiency

    def simulate_quantum_noise(self, qubits: List) -> List:
        """Simulate quantum channel noise and loss."""
        transmitted_qubits = []

        for qubit in qubits:
            # Simulate photon loss
            if np.random.random() > self.transmission_efficiency:
                continue  # Photon lost

            # Simulate depolarization noise
            if np.random.random() < 0.01:  # 1% depolarization probability
                # Apply random Pauli errors
                error_type = np.random.randint(0, 4)
                if error_type == 1:
                    qubit.X()
                elif error_type == 2:
                    qubit.Z()
                elif error_type == 3:
                    qubit.Y()

            transmitted_qubits.append(qubit)

        return transmitted_qubits


class HybridNetworkTopology:
    """Hybrid classical-quantum network topology."""

    def __init__(self, config: Dict):
        self.config = config
        self.nodes = {}
        self.channels = {}
        self.graph = nx.Graph()
        self.quantum_network = Network.get_instance()

        self._build_topology()

    def _build_topology(self):
        """Build the network topology from configuration."""
        # Create nodes
        for node_config in self.config['network_topology']['nodes']:
            node = QuantumNetworkNode(
                node_config['name'],
                node_config['type']
            )
            self.nodes[node.node_id] = node
            self.graph.add_node(node.node_id, type=node.node_type)

            # Create QuNetSim host
            host = Host(node.node_id)
            host.start()
            node.host = host
            self.quantum_network.add_host(host)

        # Create channels
        for channel_config in self.config['network_topology']['channels']:
            from_node = channel_config['from']
            to_node = channel_config['to']
            distance = channel_config['distance']

            channel = QuantumChannel(
                from_node,
                to_node,
                distance,
                self.config['quantum_channel']['loss_rate_db_per_km']
            )

            channel_key = f"{from_node}-{to_node}"
            self.channels[channel_key] = channel
            self.graph.add_edge(from_node, to_node, weight=distance, channel=channel)

        logger.info(f"Created network with {len(self.nodes)} nodes and {len(self.channels)} channels")

    def get_shortest_path(self, source: str, destination: str) -> List[str]:
        """Find shortest path between nodes."""
        try:
            path = nx.shortest_path(self.graph, source, destination, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return []

    def get_path_channels(self, path: List[str]) -> List[QuantumChannel]:
        """Get channels along a path."""
        channels = []
        for i in range(len(path) - 1):
            channel_key = f"{path[i]}-{path[i+1]}"
            if channel_key in self.channels:
                channels.append(self.channels[channel_key])
            else:
                # Try reverse direction
                channel_key = f"{path[i+1]}-{path[i]}"
                if channel_key in self.channels:
                    channels.append(self.channels[channel_key])

        return channels

    def simulate_key_distribution(self, source: str, destination: str, key_length: int) -> Dict:
        """Simulate key distribution along a path."""
        path = self.get_shortest_path(source, destination)
        if not path:
            return {'success': False, 'error': 'No path found'}

        channels = self.get_path_channels(path)
        total_distance = sum(channel.distance for channel in channels)

        # Simulate key distribution efficiency
        total_efficiency = np.prod([channel.get_channel_quality() for channel in channels])

        # Estimate final key length after losses
        estimated_final_key = int(key_length * total_efficiency)

        # Simulate timing
        speed_of_light = 2e8  # m/s in fiber (2/3 speed in vacuum)
        propagation_time = total_distance * 1000 / speed_of_light  # seconds

        result = {
            'success': True,
            'path': path,
            'total_distance': total_distance,
            'channels': len(channels),
            'total_efficiency': total_efficiency,
            'estimated_final_key_length': estimated_final_key,
            'propagation_time': propagation_time,
            'key_rate': estimated_final_key / propagation_time if propagation_time > 0 else 0
        }

        return result

    def visualize_topology(self, save_path: str = None, show_security: bool = False):
        """Visualize network topology using NetworkX and matplotlib."""
        plt.figure(figsize=(12, 8))

        # Position nodes
        pos = nx.spring_layout(self.graph, seed=42)

        # Draw nodes
        node_colors = []
        node_sizes = []

        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data['type']
            if node_type == 'source':
                node_colors.append('lightgreen')
                node_sizes.append(800)
            elif node_type == 'destination':
                node_colors.append('lightblue')
                node_sizes.append(800)
            elif node_type == 'repeater':
                node_colors.append('orange')
                node_sizes.append(600)
            else:  # intermediate
                node_colors.append('lightgray')
                node_sizes.append(500)

        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors,
                              node_size=node_sizes, alpha=0.8)

        # Draw edges
        edges = self.graph.edges()
        edge_colors = []
        edge_widths = []

        for u, v in edges:
            channel_key = f"{u}-{v}"
            if channel_key not in self.channels:
                channel_key = f"{v}-{u}"

            if channel_key in self.channels:
                channel = self.channels[channel_key]
                efficiency = channel.get_channel_quality()

                # Color based on channel quality
                if efficiency > 0.8:
                    edge_colors.append('green')
                elif efficiency > 0.5:
                    edge_colors.append('orange')
                else:
                    edge_colors.append('red')

                edge_widths.append(2 + efficiency * 3)
            else:
                edge_colors.append('gray')
                edge_widths.append(1)

        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors,
                              width=edge_widths, alpha=0.6)

        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_weight='bold')

        # Add edge labels (distances)
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            edge_labels[(u, v)] = f"{data['weight']}km"

        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=8)

        plt.title("QKD Hybrid Network Topology", fontsize=16, fontweight='bold')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Topology visualization saved to {save_path}")

        plt.tight_layout()
        plt.show()

    def get_topology_stats(self) -> Dict:
        """Get comprehensive topology statistics."""
        stats = {
            'num_nodes': len(self.nodes),
            'num_channels': len(self.channels),
            'node_types': {},
            'total_distance': 0,
            'average_channel_efficiency': 0,
            'network_diameter': 0
        }

        # Node type distribution
        for node in self.nodes.values():
            node_type = node.node_type
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1

        # Channel statistics
        efficiencies = []
        for channel in self.channels.values():
            stats['total_distance'] += channel.distance
            efficiencies.append(channel.get_channel_quality())

        if efficiencies:
            stats['average_channel_efficiency'] = np.mean(efficiencies)

        # Network diameter
        try:
            stats['network_diameter'] = nx.diameter(self.graph, weight='weight')
        except nx.NetworkXError:
            stats['network_diameter'] = float('inf')

        return stats

    def find_repeater_nodes(self) -> List[str]:
        """Find all repeater nodes in the network."""
        return [node_id for node_id, node in self.nodes.items()
                if node.node_type == 'repeater']

    def get_node_connectivity(self, node_id: str) -> Dict:
        """Get connectivity information for a specific node."""
        if node_id not in self.nodes:
            return {}

        neighbors = list(self.graph.neighbors(node_id))
        channels = []

        for neighbor in neighbors:
            channel_key = f"{node_id}-{neighbor}"
            if channel_key in self.channels:
                channels.append(self.channels[channel_key])
            else:
                channel_key = f"{neighbor}-{node_id}"
                if channel_key in self.channels:
                    channels.append(self.channels[channel_key])

        return {
            'node_id': node_id,
            'neighbors': neighbors,
            'degree': len(neighbors),
            'channels': channels,
            'average_channel_quality': np.mean([c.get_channel_quality() for c in channels]) if channels else 0
        }


def create_standard_topology(config: Dict) -> HybridNetworkTopology:
    """Create a standard 4-node topology with repeater."""
    return HybridNetworkTopology(config)


def create_mesh_topology(num_nodes: int, config: Dict) -> HybridNetworkTopology:
    """Create a mesh topology (for scalability testing)."""
    # Modify config for mesh topology
    mesh_config = config.copy()
    mesh_config['network_topology'] = {
        'nodes': [{'name': f'Node{i}', 'type': 'intermediate'} for i in range(num_nodes)],
        'channels': []
    }

    # Set source and destination
    mesh_config['network_topology']['nodes'][0]['type'] = 'source'
    mesh_config['network_topology']['nodes'][-1]['type'] = 'destination'

    # Add channels (simplified mesh)
    channels = []
    for i in range(num_nodes):
        for j in range(i+1, min(i+3, num_nodes)):  # Connect to next 2 nodes
            channels.append({
                'from': f'Node{i}',
                'to': f'Node{j}',
                'distance': 10.0
            })

    mesh_config['network_topology']['channels'] = channels

    return HybridNetworkTopology(mesh_config)
