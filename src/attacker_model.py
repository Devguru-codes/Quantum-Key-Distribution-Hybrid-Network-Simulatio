"""
Attacker model for QKD network simulation.
Includes Eve eavesdropper with various attack strategies and detection mechanisms.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EveAttacker:
    """Eve eavesdropper implementation with various attack strategies."""

    def __init__(self, config: Dict):
        self.config = config
        self.interception_rate = config.get('interception_rate', 0.1)
        self.detection_threshold = config.get('detection_threshold', 0.11)
        self.false_positive_rate = config.get('false_positive_rate', 0.01)
        self.attack_detected = False
        self.attack_history = []

    def intercept_qubit(self, qubit) -> Tuple[bool, Optional[int]]:
        """
        Intercept and measure a qubit.
        Returns: (intercepted, measured_bit)
        """
        if random.random() < self.interception_rate:
            # Intercept the qubit
            basis = random.randint(0, 1)  # Random measurement basis

            if basis == 1:  # Diagonal basis
                qubit.H()

            measured_bit = qubit.measure()

            # Resend qubit in same state (intercept-resend attack)
            # In practice, this introduces noise
            if random.random() < self.config.get('quantum_error_rate', 0.05):
                # Simulate imperfect resend
                if random.random() < 0.5:
                    qubit.X()  # Flip bit
                if random.random() < 0.5:
                    qubit.Z()  # Flip phase

            return True, measured_bit
        else:
            return False, None

    def perform_attack(self, qubits: List) -> Dict[str, any]:
        """Perform eavesdropping attack on a batch of qubits."""
        intercepted_count = 0
        intercepted_bits = []

        for qubit in qubits:
            intercepted, bit = self.intercept_qubit(qubit)
            if intercepted:
                intercepted_count += 1
                if bit is not None:
                    intercepted_bits.append(bit)

        attack_info = {
            'intercepted_qubits': intercepted_count,
            'total_qubits': len(qubits),
            'interception_rate': intercepted_count / len(qubits) if qubits else 0,
            'intercepted_bits': intercepted_bits
        }

        self.attack_history.append(attack_info)
        logger.info(f"Eve intercepted {intercepted_count}/{len(qubits)} qubits")

        return attack_info

    def calculate_induced_qber(self, attack_info: Dict) -> float:
        """Calculate QBER induced by Eve's attack."""
        interception_rate = attack_info['interception_rate']

        # PNS attack QBER formula: QBER = (1/2) * (1 - sqrt(1 - 2*η))
        # where η is interception rate
        if interception_rate > 0:
            induced_qber = 0.5 * (1 - np.sqrt(1 - 2 * interception_rate))
        else:
            induced_qber = 0.0

        return induced_qber

    def detect_attack(self, measured_qber: float) -> Dict[str, any]:
        """Detect potential eavesdropping based on QBER."""
        expected_qber = self.config.get('quantum_error_rate', 0.05)
        threshold = self.detection_threshold

        # Calculate deviation
        deviation = measured_qber - expected_qber

        # Enhanced detection logic
        detected = False
        confidence = 0.0

        if measured_qber > threshold:
            detected = True
            # Calculate confidence based on how far above threshold
            if threshold > expected_qber:
                confidence = min(1.0, (measured_qber - expected_qber) / (threshold - expected_qber))
            else:
                confidence = 0.8  # High confidence if significantly above expected
        elif measured_qber > expected_qber * 1.5:  # 50% above expected
            detected = True
            confidence = 0.6  # Medium confidence
        else:
            # False positive consideration (very low probability)
            if random.random() < self.false_positive_rate:
                detected = True
                confidence = random.random() * 0.2  # Very low confidence false positive

        detection_result = {
            'attack_detected': detected,
            'confidence': confidence,
            'measured_qber': measured_qber,
            'expected_qber': expected_qber,
            'deviation': deviation,
            'threshold': threshold
        }

        if detected:
            self.attack_detected = True
            logger.warning(f"🚨 ATTACK DETECTED! QBER: {measured_qber:.4f} (expected: {expected_qber:.4f}, threshold: {threshold:.4f})")
            logger.warning(f"   Confidence: {confidence:.2f}, Deviation: {deviation:.4f}")

        return detection_result

    def get_attack_statistics(self) -> Dict[str, any]:
        """Get comprehensive attack statistics."""
        if not self.attack_history:
            return {}

        total_intercepted = sum(attack['intercepted_qubits'] for attack in self.attack_history)
        total_qubits = sum(attack['total_qubits'] for attack in self.attack_history)
        avg_interception_rate = total_intercepted / total_qubits if total_qubits > 0 else 0

        # Calculate information leakage more realistically
        # Eve's information depends on interception rate and measurement accuracy
        def binary_entropy(p):
            if p == 0 or p == 1:
                return 0
            if p < 0 or p > 1:
                return 0
            return -p * np.log2(p) - (1-p) * np.log2(1-p)

        induced_qber = self.calculate_induced_qber({'interception_rate': avg_interception_rate})

        # More realistic Eve information calculation
        # Eve gets partial information based on interception and measurement errors
        measurement_accuracy = 0.85  # Eve's measurement accuracy (85% realistic)
        eve_information = avg_interception_rate * measurement_accuracy * (1 - binary_entropy(induced_qber))

        # Cap at realistic maximum (Eve can't get more than ~50% information in QKD)
        eve_information = min(eve_information, 0.5)

        stats = {
            'total_attacks': len(self.attack_history),
            'total_intercepted_qubits': total_intercepted,
            'total_qubits_processed': total_qubits,
            'average_interception_rate': avg_interception_rate,
            'induced_qber': induced_qber,
            'eve_information_leakage': eve_information,
            'attack_detected': self.attack_detected,
            'detection_rate': 1.0 if self.attack_detected else 0.0
        }

        return stats

    def reset(self):
        """Reset attacker state."""
        self.attack_history = []
        self.attack_detected = False


class AttackDetector:
    """Advanced attack detection mechanisms."""

    def __init__(self, config: Dict):
        self.config = config
        self.qber_history = []
        self.key_rate_history = []

    def monitor_qber(self, qber: float):
        """Monitor QBER over time for anomaly detection."""
        self.qber_history.append(qber)

        # Keep only recent history
        if len(self.qber_history) > 100:
            self.qber_history.pop(0)

    def monitor_key_rate(self, key_rate: float):
        """Monitor key rate for sudden drops."""
        self.key_rate_history.append(key_rate)

        if len(self.key_rate_history) > 100:
            self.key_rate_history.pop(0)

    def detect_anomalies(self) -> Dict[str, any]:
        """Detect anomalies in QBER and key rate."""
        anomalies = {}

        if len(self.qber_history) >= 10:
            # Statistical anomaly detection
            mean_qber = np.mean(self.qber_history)
            std_qber = np.std(self.qber_history)

            current_qber = self.qber_history[-1]
            z_score = (current_qber - mean_qber) / std_qber if std_qber > 0 else 0

            anomalies['qber_anomaly'] = {
                'detected': abs(z_score) > 3.0,  # 3-sigma rule
                'z_score': z_score,
                'current_qber': current_qber,
                'mean_qber': mean_qber
            }

        if len(self.key_rate_history) >= 10:
            # Detect sudden key rate drops
            recent_rates = self.key_rate_history[-10:]
            mean_rate = np.mean(recent_rates)
            current_rate = self.key_rate_history[-1]

            rate_drop = (mean_rate - current_rate) / mean_rate if mean_rate > 0 else 0

            anomalies['key_rate_drop'] = {
                'detected': rate_drop > 0.5,  # 50% drop
                'rate_drop_percent': rate_drop * 100,
                'current_rate': current_rate,
                'mean_rate': mean_rate
            }

        return anomalies

    def get_security_assessment(self) -> Dict[str, any]:
        """Provide overall security assessment."""
        anomalies = self.detect_anomalies()

        risk_level = "LOW"
        risk_score = 0.0

        if anomalies.get('qber_anomaly', {}).get('detected', False):
            risk_score += 0.4
        if anomalies.get('key_rate_drop', {}).get('detected', False):
            risk_score += 0.3
        if self.config.get('eve_present', False):
            risk_score += 0.3

        if risk_score > 0.6:
            risk_level = "HIGH"
        elif risk_score > 0.3:
            risk_level = "MEDIUM"

        assessment = {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'anomalies_detected': len([a for a in anomalies.values() if a.get('detected', False)]),
            'anomalies': anomalies,
            'recommendations': self._get_recommendations(risk_level)
        }

        return assessment

    def _get_recommendations(self, risk_level: str) -> List[str]:
        """Get security recommendations based on risk level."""
        recommendations = []

        if risk_level == "HIGH":
            recommendations.extend([
                "Immediate protocol abort recommended",
                "Increase error correction redundancy",
                "Verify quantum channel integrity",
                "Consider switching to alternative key distribution"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "Monitor QBER closely",
                "Increase privacy amplification",
                "Verify key rate stability",
                "Consider additional authentication"
            ])
        else:
            recommendations.extend([
                "Continue normal operation",
                "Regular security audits recommended",
                "Monitor for gradual QBER increase"
            ])

        return recommendations
