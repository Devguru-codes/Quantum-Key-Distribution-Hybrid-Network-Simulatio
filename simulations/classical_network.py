"""
Classical network simulation for comparison with QKD networks.
Implements RSA/AES-based secure communication without quantum security.
"""

import time
import random
import numpy as np
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from typing import Dict, List, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassicalCrypto:
    """Classical cryptographic primitives for comparison."""

    def __init__(self, config: Dict):
        self.config = config
        self.backend = default_backend()
        self.key_pairs = {}  # {node_id: (private_key, public_key)}

    def generate_rsa_keypair(self, node_id: str, key_size: int = 2048):
        """Generate RSA key pair for a node."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=self.backend
        )
        public_key = private_key.public_key()

        self.key_pairs[node_id] = (private_key, public_key)
        logger.info(f"Generated RSA key pair for {node_id}")

        return private_key, public_key

    def rsa_encrypt(self, message: bytes, public_key) -> bytes:
        """Encrypt message with RSA public key."""
        ciphertext = public_key.encrypt(
            message,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return ciphertext

    def rsa_decrypt(self, ciphertext: bytes, private_key) -> bytes:
        """Decrypt message with RSA private key."""
        plaintext = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return plaintext

    def aes_encrypt(self, message: bytes, key: bytes) -> Dict[str, bytes]:
        """Encrypt message with AES."""
        iv = random.randbytes(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()

        # Pad message
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(message) + padder.finalize()

        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        return {'ciphertext': ciphertext, 'iv': iv}

    def aes_decrypt(self, encrypted_data: Dict[str, bytes], key: bytes) -> bytes:
        """Decrypt message with AES."""
        cipher = Cipher(algorithms.AES(key), modes.CBC(encrypted_data['iv']), backend=self.backend)
        decryptor = cipher.decryptor()

        padded_plaintext = decryptor.update(encrypted_data['ciphertext']) + decryptor.finalize()

        # Unpad
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

        return plaintext


class ClassicalNetworkNode:
    """Classical network node with RSA/AES capabilities."""

    def __init__(self, node_id: str, crypto: ClassicalCrypto):
        self.node_id = node_id
        self.crypto = crypto
        self.public_keys = {}  # {peer_id: public_key}
        self.session_keys = {}  # {peer_id: session_key}
        self.messages_sent = 0
        self.messages_received = 0

    def setup_keys(self):
        """Set up RSA key pair."""
        self.crypto.generate_rsa_keypair(self.node_id)

    def get_public_key(self):
        """Get this node's public key."""
        _, public_key = self.crypto.key_pairs[self.node_id]
        return public_key

    def exchange_public_keys(self, peer_id: str, peer_public_key):
        """Exchange public keys with peer."""
        self.public_keys[peer_id] = peer_public_key

    def establish_session_key(self, peer_id: str) -> bytes:
        """Establish AES session key with peer."""
        # Generate random session key
        session_key = random.randbytes(32)  # 256-bit AES key

        # Encrypt session key with peer's public key
        encrypted_key = self.crypto.rsa_encrypt(session_key, self.public_keys[peer_id])

        # Store session key
        self.session_keys[peer_id] = session_key

        return encrypted_key

    def receive_session_key(self, peer_id: str, encrypted_key: bytes):
        """Receive and decrypt session key from peer."""
        private_key, _ = self.crypto.key_pairs[self.node_id]
        session_key = self.crypto.rsa_decrypt(encrypted_key, private_key)
        self.session_keys[peer_id] = session_key

    def send_message(self, peer_id: str, message: bytes) -> Dict[str, Any]:
        """Send encrypted message to peer."""
        if peer_id not in self.session_keys:
            raise ValueError(f"No session key established with {peer_id}")

        start_time = time.time()

        # Encrypt message with session key
        encrypted_data = self.crypto.aes_encrypt(message, self.session_keys[peer_id])

        encryption_time = time.time() - start_time

        self.messages_sent += 1

        result = {
            'encrypted_data': encrypted_data,
            'encryption_time': encryption_time,
            'message_size': len(message),
            'ciphertext_size': len(encrypted_data['ciphertext'])
        }

        return result

    def receive_message(self, peer_id: str, encrypted_data: Dict[str, bytes]) -> Dict[str, Any]:
        """Receive and decrypt message from peer."""
        if peer_id not in self.session_keys:
            raise ValueError(f"No session key established with {peer_id}")

        start_time = time.time()

        # Decrypt message
        plaintext = self.crypto.aes_decrypt(encrypted_data, self.session_keys[peer_id])

        decryption_time = time.time() - start_time

        self.messages_received += 1

        result = {
            'plaintext': plaintext,
            'decryption_time': decryption_time,
            'ciphertext_size': len(encrypted_data['ciphertext']),
            'plaintext_size': len(plaintext)
        }

        return result


class ClassicalNetworkSimulation:
    """Classical network simulation with RSA/AES security."""

    def __init__(self, config: Dict):
        self.config = config
        self.crypto = ClassicalCrypto(config)
        self.nodes = {}
        self.channels = {}  # {channel_id: properties}
        self.message_log = []
        self.setup_time = 0
        self.communication_time = 0

    def setup_network(self, node_ids: List[str]):
        """Set up classical network with given nodes."""
        start_time = time.time()

        for node_id in node_ids:
            node = ClassicalNetworkNode(node_id, self.crypto)
            node.setup_keys()
            self.nodes[node_id] = node

        # Exchange public keys (simulating PKI)
        for node_id, node in self.nodes.items():
            for peer_id, peer_node in self.nodes.items():
                if node_id != peer_id:
                    node.exchange_public_keys(peer_id, peer_node.get_public_key())

        self.setup_time = time.time() - start_time
        logger.info(f"Classical network setup completed in {self.setup_time:.4f} seconds")

    def establish_secure_channels(self):
        """Establish AES session keys between all node pairs."""
        start_time = time.time()

        for sender_id, sender in self.nodes.items():
            for receiver_id, receiver in self.nodes.items():
                if sender_id != receiver_id:
                    # Sender initiates key exchange
                    encrypted_key = sender.establish_session_key(receiver_id)

                    # Receiver accepts key
                    receiver.receive_session_key(sender_id, encrypted_key)

        key_exchange_time = time.time() - start_time
        logger.info(f"Key exchange completed in {key_exchange_time:.4f} seconds")

        return key_exchange_time

    def simulate_communication(self, num_messages: int = 100, message_size: int = 1024):
        """Simulate secure communication between nodes."""
        start_time = time.time()

        node_ids = list(self.nodes.keys())

        for _ in range(num_messages):
            # Random sender and receiver
            sender_id = random.choice(node_ids)
            receiver_id = random.choice([nid for nid in node_ids if nid != sender_id])

            # Generate random message
            message = random.randbytes(message_size)

            # Send message
            send_result = self.nodes[sender_id].send_message(receiver_id, message)

            # Simulate network delay (classical channel)
            network_delay = random.uniform(0.0001, 0.001)  # 0.1-1ms
            time.sleep(network_delay)

            # Receive message
            receive_result = self.nodes[receiver_id].receive_message(sender_id, send_result['encrypted_data'])

            # Log communication
            self.message_log.append({
                'sender': sender_id,
                'receiver': receiver_id,
                'message_size': message_size,
                'ciphertext_size': send_result['ciphertext_size'],
                'encryption_time': send_result['encryption_time'],
                'decryption_time': receive_result['decryption_time'],
                'network_delay': network_delay,
                'total_time': send_result['encryption_time'] + receive_result['decryption_time'] + network_delay
            })

        self.communication_time = time.time() - start_time

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self.message_log:
            return {}

        # Calculate metrics
        total_messages = len(self.message_log)
        total_encryption_time = sum(msg['encryption_time'] for msg in self.message_log)
        total_decryption_time = sum(msg['decryption_time'] for msg in self.message_log)
        total_network_time = sum(msg['network_delay'] for msg in self.message_log)
        total_time = sum(msg['total_time'] for msg in self.message_log)

        avg_message_size = np.mean([msg['message_size'] for msg in self.message_log])
        avg_ciphertext_size = np.mean([msg['ciphertext_size'] for msg in self.message_log])

        # Throughput calculation
        total_data_sent = sum(msg['message_size'] for msg in self.message_log)
        throughput = total_data_sent / self.communication_time if self.communication_time > 0 else 0

        # Latency metrics
        latencies = [msg['total_time'] for msg in self.message_log]

        metrics = {
            'setup_time': self.setup_time,
            'key_exchange_time': self.establish_secure_channels(),
            'communication_time': self.communication_time,
            'total_messages': total_messages,
            'avg_encryption_time': total_encryption_time / total_messages,
            'avg_decryption_time': total_decryption_time / total_messages,
            'avg_network_delay': total_network_time / total_messages,
            'avg_total_latency': total_time / total_messages,
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'latency_std': np.std(latencies),
            'throughput_bytes_per_sec': throughput,
            'throughput_messages_per_sec': total_messages / self.communication_time if self.communication_time > 0 else 0,
            'avg_message_size': avg_message_size,
            'avg_ciphertext_size': avg_ciphertext_size,
            'expansion_ratio': avg_ciphertext_size / avg_message_size if avg_message_size > 0 else 0,
            'rsa_key_size': self.config.get('rsa_key_size', 2048),
            'aes_key_size': 256
        }

        return metrics

    def compare_with_qkd(self, qkd_metrics: Dict) -> Dict[str, Any]:
        """Compare classical performance with QKD metrics."""
        classical_metrics = self.get_performance_metrics()

        comparisons = {}

        # Setup time comparison
        if 'setup_time' in qkd_metrics:
            comparisons['setup_time_ratio'] = classical_metrics['setup_time'] / qkd_metrics['setup_time']

        # Throughput comparison
        if 'throughput' in qkd_metrics:
            comparisons['throughput_ratio'] = classical_metrics['throughput_bytes_per_sec'] / qkd_metrics['throughput']

        # Latency comparison
        if 'avg_network_latency' in qkd_metrics:
            comparisons['latency_ratio'] = classical_metrics['avg_total_latency'] / qkd_metrics['avg_network_latency']

        return comparisons

    def run_simulation(self, num_nodes: int = 4, num_messages: int = 100) -> Dict[str, Any]:
        """Run complete classical network simulation."""
        logger.info("Starting classical network simulation...")

        # Setup network
        node_ids = [f'Node{i}' for i in range(num_nodes)]
        self.setup_network(node_ids)

        # Establish secure channels
        key_exchange_time = self.establish_secure_channels()

        # Simulate communication
        self.simulate_communication(num_messages)

        # Get metrics
        metrics = self.get_performance_metrics()
        metrics['num_nodes'] = num_nodes
        metrics['num_messages'] = num_messages

        logger.info("Classical network simulation completed")
        logger.info(f"Results: {num_messages} messages, throughput: {metrics['throughput_messages_per_sec']:.1f} msg/s")

        return metrics


def demonstrate_classical_network():
    """Demonstrate classical network simulation."""
    config = {
        'rsa_key_size': 2048,
        'aes_mode': 'CBC'
    }

    simulation = ClassicalNetworkSimulation(config)
    results = simulation.run_simulation(num_nodes=4, num_messages=50)

    print("\nClassical Network Simulation Results:")
    print(f"Setup Time: {results['setup_time']:.4f} seconds")
    print(f"Key Exchange Time: {results['key_exchange_time']:.4f} seconds")
    print(f"Communication Time: {results['communication_time']:.4f} seconds")
    print(f"Throughput: {results['throughput_messages_per_sec']:.1f} messages/second")
    print(f"Average Latency: {results['avg_total_latency']:.6f} seconds")
    print(f"Expansion Ratio: {results['expansion_ratio']:.2f}")

    return results


if __name__ == "__main__":
    demonstrate_classical_network()
