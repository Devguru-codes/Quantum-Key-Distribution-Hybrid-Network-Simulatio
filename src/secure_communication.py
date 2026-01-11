"""
Secure communication layer using QKD-generated keys for encryption.
Implements AES encryption/decryption with keys from QKD protocols.
"""

import os
import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from typing import Dict, List, Tuple, Optional, Union
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumKeyManager:
    """Manages quantum-generated keys for secure communication."""

    def __init__(self, config: Dict):
        self.config = config
        self.key_store = {}  # {peer_id: key_data}
        self.key_counter = {}  # {peer_id: usage_count}
        self.key_expiry = {}  # {peer_id: expiry_time}

    def store_key(self, peer_id: str, key_bits: List[int], protocol: str = "BB84"):
        """Store a quantum-generated key."""
        # Convert bit list to bytes
        key_bytes = self._bits_to_bytes(key_bits)

        # Hash for additional security (optional)
        key_hash = hashlib.sha256(key_bytes).digest()

        key_data = {
            'key': key_bytes,
            'hash': key_hash,
            'length': len(key_bits),
            'protocol': protocol,
            'timestamp': time.time(),
            'used_count': 0
        }

        self.key_store[peer_id] = key_data
        self.key_counter[peer_id] = 0

        # Set expiry (24 hours by default)
        expiry_hours = self.config.get('key_expiry_hours', 24)
        self.key_expiry[peer_id] = time.time() + (expiry_hours * 3600)

        logger.info(f"Stored {len(key_bits)}-bit key for peer {peer_id} using {protocol}")

    def get_key(self, peer_id: str) -> Optional[bytes]:
        """Retrieve a key for a peer."""
        if peer_id not in self.key_store:
            return None

        # Check expiry
        if time.time() > self.key_expiry.get(peer_id, 0):
            logger.warning(f"Key for peer {peer_id} has expired")
            self.delete_key(peer_id)
            return None

        key_data = self.key_store[peer_id]
        self.key_counter[peer_id] += 1
        key_data['used_count'] += 1

        return key_data['key']

    def delete_key(self, peer_id: str):
        """Delete a key for a peer."""
        if peer_id in self.key_store:
            del self.key_store[peer_id]
            del self.key_counter[peer_id]
            del self.key_expiry[peer_id]
            logger.info(f"Deleted key for peer {peer_id}")

    def get_key_status(self, peer_id: str) -> Dict:
        """Get status of key for a peer."""
        if peer_id not in self.key_store:
            return {'available': False}

        key_data = self.key_store[peer_id]
        current_time = time.time()

        return {
            'available': True,
            'length': key_data['length'],
            'protocol': key_data['protocol'],
            'used_count': key_data['used_count'],
            'age_seconds': current_time - key_data['timestamp'],
            'expires_in_seconds': self.key_expiry[peer_id] - current_time,
            'expired': current_time > self.key_expiry[peer_id]
        }

    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Convert list of bits to bytes."""
        # Pad to multiple of 8
        while len(bits) % 8 != 0:
            bits.append(0)

        byte_list = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte |= bits[i + j] << (7 - j)
            byte_list.append(byte)

        return bytes(byte_list)

    def rotate_key(self, peer_id: str, new_key_bits: List[int], protocol: str = "BB84"):
        """Rotate to a new key."""
        self.delete_key(peer_id)
        self.store_key(peer_id, new_key_bits, protocol)


class AESEncryptor:
    """AES encryption/decryption using quantum keys."""

    def __init__(self, key_manager: QuantumKeyManager, config: Dict):
        self.key_manager = key_manager
        self.config = config
        self.backend = default_backend()

    def encrypt_message(self, message: Union[str, bytes], peer_id: str) -> Optional[Dict]:
        """Encrypt a message using quantum key for peer."""
        key = self.key_manager.get_key(peer_id)
        if not key:
            logger.error(f"No key available for peer {peer_id}")
            return None

        # Convert message to bytes
        if isinstance(message, str):
            message_bytes = message.encode('utf-8')
        else:
            message_bytes = message

        # Generate random IV
        iv = os.urandom(16)

        # Create cipher
        cipher = Cipher(
            algorithms.AES(key[:32]),  # Use first 32 bytes (256-bit)
            modes.CBC(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()

        # Pad message
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(message_bytes) + padder.finalize()

        # Encrypt
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        encrypted_data = {
            'ciphertext': ciphertext,
            'iv': iv,
            'peer_id': peer_id,
            'timestamp': time.time(),
            'algorithm': 'AES-256-CBC'
        }

        logger.info(f"Encrypted message for peer {peer_id} using quantum key")
        return encrypted_data

    def decrypt_message(self, encrypted_data: Dict, peer_id: str) -> Optional[bytes]:
        """Decrypt a message using quantum key for peer."""
        key = self.key_manager.get_key(peer_id)
        if not key:
            logger.error(f"No key available for peer {peer_id}")
            return None

        try:
            # Extract components
            ciphertext = encrypted_data['ciphertext']
            iv = encrypted_data['iv']

            # Create cipher
            cipher = Cipher(
                algorithms.AES(key[:32]),  # Use first 32 bytes (256-bit)
                modes.CBC(iv),
                backend=self.backend
            )
            decryptor = cipher.decryptor()

            # Decrypt
            padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

            # Unpad
            unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
            plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

            logger.info(f"Decrypted message from peer {peer_id}")
            return plaintext

        except Exception as e:
            logger.error(f"Decryption failed for peer {peer_id}: {e}")
            return None

    def encrypt_file(self, file_path: str, peer_id: str, output_path: str = None) -> Optional[str]:
        """Encrypt a file using quantum key."""
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()

            encrypted_data = self.encrypt_message(file_data, peer_id)
            if not encrypted_data:
                return None

            if not output_path:
                output_path = file_path + '.encrypted'

            # Save encrypted data
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(encrypted_data, f)

            logger.info(f"Encrypted file {file_path} to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"File encryption failed: {e}")
            return None

    def decrypt_file(self, encrypted_file_path: str, peer_id: str, output_path: str = None) -> Optional[str]:
        """Decrypt a file using quantum key."""
        try:
            import pickle
            with open(encrypted_file_path, 'rb') as f:
                encrypted_data = pickle.load(f)

            decrypted_data = self.decrypt_message(encrypted_data, peer_id)
            if not decrypted_data:
                return None

            if not output_path:
                output_path = encrypted_file_path.replace('.encrypted', '.decrypted')

            with open(output_path, 'wb') as f:
                f.write(decrypted_data)

            logger.info(f"Decrypted file {encrypted_file_path} to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"File decryption failed: {e}")
            return None


class SecureCommunicationChannel:
    """End-to-end secure communication channel using QKD + AES."""

    def __init__(self, config: Dict):
        self.config = config
        self.key_manager = QuantumKeyManager(config)
        self.encryptor = AESEncryptor(self.key_manager, config)
        self.message_log = []

    def establish_secure_channel(self, peer_id: str, qkd_key_bits: List[int],
                                protocol: str = "BB84") -> bool:
        """Establish secure channel with quantum key."""
        try:
            self.key_manager.store_key(peer_id, qkd_key_bits, protocol)
            logger.info(f"Secure channel established with {peer_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to establish secure channel with {peer_id}: {e}")
            return False

    def send_secure_message(self, message: Union[str, bytes], peer_id: str) -> Optional[Dict]:
        """Send an encrypted message."""
        encrypted_data = self.encryptor.encrypt_message(message, peer_id)
        if encrypted_data:
            self.message_log.append({
                'type': 'sent',
                'peer_id': peer_id,
                'timestamp': time.time(),
                'size': len(encrypted_data['ciphertext'])
            })

        return encrypted_data

    def receive_secure_message(self, encrypted_data: Dict, peer_id: str) -> Optional[bytes]:
        """Receive and decrypt a message."""
        decrypted_message = self.encryptor.decrypt_message(encrypted_data, peer_id)
        if decrypted_message:
            self.message_log.append({
                'type': 'received',
                'peer_id': peer_id,
                'timestamp': time.time(),
                'size': len(decrypted_message)
            })

        return decrypted_message

    def get_channel_status(self, peer_id: str) -> Dict:
        """Get status of secure channel with peer."""
        key_status = self.key_manager.get_key_status(peer_id)

        # Count messages
        sent_count = len([m for m in self.message_log if m['type'] == 'sent' and m['peer_id'] == peer_id])
        received_count = len([m for m in self.message_log if m['type'] == 'received' and m['peer_id'] == peer_id])

        return {
            'peer_id': peer_id,
            'key_status': key_status,
            'messages_sent': sent_count,
            'messages_received': received_count,
            'channel_active': key_status.get('available', False) and not key_status.get('expired', True)
        }

    def rotate_keys(self, peer_id: str, new_qkd_key_bits: List[int], protocol: str = "BB84"):
        """Rotate to new quantum keys."""
        self.key_manager.rotate_key(peer_id, new_qkd_key_bits, protocol)
        logger.info(f"Keys rotated for peer {peer_id}")

    def get_communication_stats(self) -> Dict:
        """Get communication statistics."""
        total_sent = len([m for m in self.message_log if m['type'] == 'sent'])
        total_received = len([m for m in self.message_log if m['type'] == 'received'])

        total_sent_bytes = sum(m['size'] for m in self.message_log if m['type'] == 'sent')
        total_received_bytes = sum(m['size'] for m in self.message_log if m['type'] == 'received')

        return {
            'total_messages_sent': total_sent,
            'total_messages_received': total_received,
            'total_bytes_sent': total_sent_bytes,
            'total_bytes_received': total_received_bytes,
            'unique_peers': len(set(m['peer_id'] for m in self.message_log))
        }


def demonstrate_secure_communication():
    """Demonstrate secure communication with QKD keys."""
    # Example usage
    config = {
        'key_expiry_hours': 24,
        'algorithm': 'AES',
        'mode': 'CBC'
    }

    channel = SecureCommunicationChannel(config)

    # Simulate QKD key generation (normally from BB84/E91)
    alice_key = [1, 0, 1, 1, 0, 1, 0, 0] * 32  # 256-bit key
    bob_key = alice_key.copy()  # In reality, keys would be identical after QKD

    # Establish secure channels
    channel.establish_secure_channel('Alice', alice_key, 'BB84')
    channel.establish_secure_channel('Bob', bob_key, 'BB84')

    # Send secure messages
    message = "This is a secret message protected by quantum keys!"
    encrypted = channel.send_secure_message(message, 'Bob')

    if encrypted:
        # Simulate transmission
        decrypted = channel.receive_secure_message(encrypted, 'Bob')
        if decrypted:
            print(f"Original: {message}")
            print(f"Decrypted: {decrypted.decode('utf-8')}")
            print("✓ Secure communication successful!")

    # Show channel status
    print("\nChannel Status:")
    for peer in ['Alice', 'Bob']:
        status = channel.get_channel_status(peer)
        print(f"{peer}: Key available = {status['key_status']['available']}")


if __name__ == "__main__":
    demonstrate_secure_communication()
