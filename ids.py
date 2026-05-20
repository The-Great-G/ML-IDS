import os
import time
import queue
import psutil
import asyncio
import logging
import tempfile
import subprocess
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from threading import Thread, Event, RLock
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import joblib


# ======================= LOGGING CONFIG =======================

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s",
    handlers=[
        logging.FileHandler("ids.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ======================= TRAINING MODULE =======================


class IntrusionDetectionSystem:
    """
    Complete Intrusion Detection System with training and prediction capabilities
    """

    def __init__(
        self,
        model_path: str = "ids_model.pkl",
        scaler_path: str = "scaler.pkl",
        encoder_path: str = "label_encoder.pkl",
    ):
        self.features = [
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Fwd Packet Length Mean",
            "Bwd Packet Length Mean",
            "Fwd Packet Length Std",
            "Bwd Packet Length Std",
            "Flow Packets/s",
            "Flow Bytes/s",
        ]
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        logger.info("IntrusionDetectionSystem initialized")

    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and basic-clean the dataset.
        """
        try:
            logger.info(f"Loading dataset from {data_path}")
            data = pd.read_csv(data_path)

            # Normalize column names (strip spaces)
            data.columns = data.columns.str.strip()

            # Validate required columns
            required_cols = self.features + ["Label"]
            missing = [c for c in required_cols if c not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Remove duplicates and nulls
            before = len(data)
            data = data.drop_duplicates()
            after_dups = len(data)
            data = data.dropna()
            after_na = len(data)
            logger.info(
                f"Dataset loaded: {before} rows, "
                f"{before - after_dups} duplicates removed, "
                f"{after_dups - after_na} rows with NaN removed. "
                f"Final: {after_na} rows."
            )

            X = data[self.features]
            y = data["Label"]

            logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")
            return X, y

        except Exception as e:
            logger.exception(f"Error loading data: {e}")
            raise

    def preprocess_and_split(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Proper preprocessing with train/test split and no data leakage.
        """
        try:
            logger.info("Preprocessing data (split, scale, encode)")

            # Replace inf with NaN then drop rows with any NaN
            X = X.replace([np.inf, -np.inf], np.nan)
            mask = ~X.isna().any(axis=1)
            X = X[mask]
            y = y[mask]
            logger.info(f"After inf/NaN handling: {X.shape[0]} samples")

            # Split first to avoid leakage
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y,
            )
            logger.info(
                f"Split complete: X_train={X_train.shape}, X_test={X_test.shape}"
            )

            # Scale on train only
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Encode labels on train only
            self.label_encoder = LabelEncoder()
            y_train_enc = self.label_encoder.fit_transform(y_train)
            y_test_enc = self.label_encoder.transform(y_test)

            # Persist scaler and encoder
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump(self.label_encoder, self.encoder_path)
            logger.info(f"Preprocessing done. Classes: {self.label_encoder.classes_}")

            return X_train_scaled, X_test_scaled, y_train_enc, y_test_enc

        except Exception as e:
            logger.exception(f"Error preprocessing data: {e}")
            raise

    def train_model(self, X: np.ndarray, y: np.ndarray, model_type: str = "rf") -> None:
        """
        Train the machine learning model.
        """
        try:
            classes, counts = np.unique(y, return_counts=True)
            logger.info(f"Training labels: classes={classes}, counts={counts}")
            if len(classes) < 2:
                raise ValueError(
                    f"Need at least 2 classes, got {len(classes)}: {classes} "
                    f"with counts {counts}"
                )

            mt = model_type.lower()
            if mt == "rf":
                logger.info("Training Random Forest model")
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                )
            elif mt == "svm":
                # Use a scalable linear SVM implementation
                logger.info("Training LinearSVC model (linear SVM for large datasets)")

                # Optional: sample if dataset is huge, to avoid very long training
                max_svm_samples = 50000
                if X.shape[0] > max_svm_samples:
                    logger.warning(
                        f"Dataset has {X.shape[0]} samples; sampling "
                        f"{max_svm_samples} for SVM training to keep fit time reasonable."
                )
                    rng = np.random.RandomState(42)
                    idx = rng.choice(X.shape[0], max_svm_samples, replace=False)
                    X = X[idx]
                    y = y[idx]

                self.model = LinearSVC(
                    class_weight="balanced",
                    random_state=42,
                    max_iter=5000,
                    dual="auto",
            )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            start = time.time()
            self.model.fit(X, y)
            logger.info(f"Model fit completed in {time.time() - start:.2f} seconds")

            joblib.dump(self.model, self.model_path)
            logger.info("Model training completed and saved")

        except Exception as e:
            logger.exception(f"Error training model: {e}")
            raise

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the model using standard metrics.
        """
        try:
            y_pred = self.model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            f1 = f1_score(y_test, y_pred, average="weighted")

            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision (weighted): {precision:.4f}")
            logger.info(f"Recall (weighted): {recall:.4f}")
            logger.info(f"F1-Score (weighted): {f1:.4f}")
            logger.info(
                "Classification Report:\n"
                + classification_report(
                    y_test,
                    y_pred,
                    target_names=self.label_encoder.classes_,
                    zero_division=0,
                )
            )

            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            }

        except Exception as e:
            logger.exception(f"Error evaluating model: {e}")
            raise

    def load_components(self) -> None:
        """
        Load pre-trained model, scaler, and label encoder.
        """
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.label_encoder = joblib.load(self.encoder_path)
            logger.info("Model components loaded successfully")
        except Exception as e:
            logger.exception(f"Error loading components: {e}")
            raise


# ======================= REAL-TIME DETECTION MODULE =======================


class RealtimeIDSOptimized:
    """
    Optimized real-time IDS with threaded architecture,
    packet buffering, and confidence-based alerting
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        encoder_path: str,
        buffer_size: int = 2000,
        external_predictions: Optional[list] = None,
        external_alerts: Optional[list] = None,
    ):
        self.ids = IntrusionDetectionSystem(model_path, scaler_path, encoder_path)
        self.model = None
        self.scaler = None
        self.label_encoder = None

        # Thread-safe components
        self.packet_queue = queue.Queue(maxsize=buffer_size)
        self.prediction_queue = queue.Queue(maxsize=buffer_size)

        self.stop_event = Event()
        self.lock = RLock()

        # Thread references
        self.capture_thread = None
        self.detection_threads = []
        self.alert_thread = None

        # Metrics
        self.packets_captured = 0
        self.packets_processed = 0
        self.alerts_generated = 0
        self.detection_times = []
        self.start_time = datetime.utcnow()
        self.process = psutil.Process(os.getpid())

        # Alert aggregation
        self.recent_alerts = defaultdict(
            lambda: {"count": 0, "first_seen": None, "last_seen": None}
        )
        self.external_predictions = external_predictions
        self.external_alerts = external_alerts
        self.alert_aggregation_window = 5

        logger.info("RealtimeIDSOptimized initialized")

    def load_components(self) -> None:
        """
        Load pre-trained components for real-time use.
        """
        try:
            self.model = joblib.load(self.ids.model_path)
            self.scaler = joblib.load(self.ids.scaler_path)
            self.label_encoder = joblib.load(self.ids.encoder_path)
            logger.info("Real-time IDS components loaded successfully")
        except Exception as e:
            logger.exception(f"Error loading components: {e}")
            raise

    def optimized_capture_traffic(
        self, interface: str = "eth0", snaplen: int = 96, duration: int = 1
    ) -> List[str]:
        """
        Optimized packet capture using dumpcap/tshark with temp files.
        """
        import platform

        try:
            system = platform.system()
            logger.info(f"[{system}] Capturing packets from {interface}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
                tmp_file = tmp.name

            # Capture command
            if system == "Windows":
                cmd = (
                    f'tshark -i "{interface}" '
                    f"-a duration:{duration} "
                    f'-w "{tmp_file}" '
                    f"-q"
                )
            else:
                cmd = (
                    f"dumpcap -i {interface} "
                    f"-s {snaplen} "
                    f"-a duration:{duration} "
                    f"-w {tmp_file} "
                    f"-q"
                )

            logger.info(f"Capture command: {cmd}")
            result = subprocess.run(
                cmd,
                shell=True,
                timeout=duration + 5,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            if result.returncode not in (0, 1):
                logger.warning(
                    f"Capture returned code {result.returncode}: {result.stderr}"
                )

            # Parse captured packets with a consistent separator (comma)
            parse_cmd = (
                f'tshark -r "{tmp_file}" '
                f"-T fields "
                f"-e frame.time_delta "
                f"-e ip.proto "
                f"-e frame.len "
                f"-e tcp.srcport "
                f"-E separator=, "
                f"-E quote=n "
                f"-E header=n"
            )

            logger.info(f"Parse command: {parse_cmd}")
            parse_result = subprocess.run(
                parse_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if parse_result.returncode not in (0, 1):
                logger.warning(
                    f"Parse returned code {parse_result.returncode}: "
                    f"{parse_result.stderr}"
                )

            output = [
                line for line in parse_result.stdout.splitlines() if line.strip()
            ]

            logger.info(f"Captured {len(output)} packets from {interface}")

            # Update metrics
            with self.lock:
                self.packets_captured += len(output)

            return output

        except subprocess.TimeoutExpired:
            logger.warning(f"Packet capture timeout on {interface}")
            return []
        except FileNotFoundError as e:
            logger.error(f"Required tool not found: {e}")
            logger.error("Ensure Wireshark (tshark/dumpcap) is installed and in PATH")
            return []
        except Exception as e:
            logger.exception(f"Capture error: {e}")
            return []
        finally:
            try:
                if "tmp_file" in locals() and os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except Exception:
                pass

    def extract_features_stateful(self, packets: List[str]) -> List[Tuple]:
        """
        Extract flow-level features from packets (simplified placeholder).
        """
        flow_states: Dict[str, Dict] = {}
        features_list: List[Tuple[List[float], str]] = []

        for packet_line in packets:
            try:
                fields = packet_line.split(",")
                if len(fields) < 4:
                    continue

                timestamp = float(fields[0]) if fields[0] else 0.0
                protocol = int(fields[1]) if fields[1] else 6
                pkt_len = int(fields[2]) if fields[2] else 0
                src_port = int(fields[3]) if fields[3] else 0

                # NOTE: For real bidirectional flows, you would include IPs and dst port.
                flow_id = f"{protocol}:{src_port}"

                if flow_id not in flow_states:
                    flow_states[flow_id] = {
                        "start_time": timestamp,
                        "fwd_packets": 0,
                        "bwd_packets": 0,
                        "fwd_bytes": 0,
                        "bwd_bytes": 0,
                        "fwd_lengths": [],
                        "bwd_lengths": [],
                        "last_time": timestamp,
                    }

                flow = flow_states[flow_id]
                flow["last_time"] = timestamp

                # Placeholder: treat all observed packets as "forward"
                # (replace with true direction logic using src/dst IP/port)
                flow["fwd_packets"] += 1
                flow["fwd_bytes"] += max(pkt_len, 0)
                if pkt_len > 0:
                    flow["fwd_lengths"].append(pkt_len)

                features = self._compute_flow_features(flow)
                features_list.append((features, flow_id))

            except (ValueError, IndexError) as e:
                logger.debug(f"Packet parse error: {e}")
                continue

        return features_list

    def _compute_flow_features(self, flow: Dict) -> List[float]:
        """
        Compute statistical features from flow state.
        """
        flow_duration = max(flow["last_time"] - flow["start_time"], 0.001)
        fwd_packets = flow["fwd_packets"]
        bwd_packets = flow["bwd_packets"]

        fwd_len_mean = float(np.mean(flow["fwd_lengths"])) if flow["fwd_lengths"] else 0.0
        bwd_len_mean = float(np.mean(flow["bwd_lengths"])) if flow["bwd_lengths"] else 0.0
        fwd_len_std = (
            float(np.std(flow["fwd_lengths"]))
            if len(flow["fwd_lengths"]) > 1
            else 0.0
        )
        bwd_len_std = (
            float(np.std(flow["bwd_lengths"]))
            if len(flow["bwd_lengths"]) > 1
            else 0.0
        )

        packet_rate = (fwd_packets + bwd_packets) / flow_duration
        byte_rate = (flow["fwd_bytes"] + flow["bwd_bytes"]) / flow_duration

        return [
            flow_duration,
            fwd_packets,
            bwd_packets,
            fwd_len_mean,
            bwd_len_mean,
            fwd_len_std,
            bwd_len_std,
            packet_rate,
            byte_rate,
        ]

    def predict_with_confidence(
        self, features: List[float], confidence_threshold: float = 0.85
    ) -> Dict:
        """
        Make predictions with confidence scoring.
        """
        try:
            features = list(features)
            if len(features) != len(self.ids.features):
                raise ValueError(
                    f"Expected {len(self.ids.features)} features, got {len(features)}"
                )

            feature_df = pd.DataFrame([features], columns=self.ids.features)
            features_scaled = self.scaler.transform(feature_df)

            prediction = self.model.predict(features_scaled)[0]

            # Handle models that do not support predict_proba
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features_scaled)[0]
            elif hasattr(self.model, "decision_function"):
                # For LinearSVC and similar models without predict_proba
                scores = self.model.decision_function(features_scaled)

                # scores shape: (n_classes,) or scalar for binary
                if np.ndim(scores) == 1:
                    scores = scores[0]
                    if np.isscalar(scores):
                        # Binary case: create 2-class scores
                        scores = np.array([scores, -scores])
                else:
                    scores = scores[0]

                # Softmax over decision scores to get pseudo-probabilities
                scores = np.asarray(scores, dtype=float)
                scores = scores - np.max(scores)
                exp_scores = np.exp(scores)
                sum_exp = exp_scores.sum()
                if sum_exp == 0:
                    probabilities = np.ones_like(exp_scores) / len(exp_scores)
                else:
                    probabilities = exp_scores / sum_exp
            else:
                # Fallback: hard prediction only
                probabilities = np.zeros(len(self.label_encoder.classes_))
                class_index = int(prediction)
                if 0 <= class_index < len(probabilities):
                    probabilities[class_index] = 1.0
            probabilities = np.asarray(probabilities, dtype=float)
            sorted_indices = np.argsort(probabilities)[::-1]
            top_confidence = float(probabilities[sorted_indices[0]])
            second_confidence = (
                float(probabilities[sorted_indices[1]])
                if len(sorted_indices) > 1
                else 0.0
            )

            label = self.label_encoder.inverse_transform([prediction])[0]

            if top_confidence >= confidence_threshold:
                action = "ALERT" if label != "BENIGN" else "PASS"
                certainty = "HIGH"
            elif top_confidence >= 0.60:
                action = "INVESTIGATE"
                certainty = "MEDIUM"
            else:
                action = "PASS"
                certainty = "LOW"

            return {
                "label": label,
                "confidence": top_confidence,
                "second_confidence": second_confidence,
                "certainty": certainty,
                "action": action,
                "probabilities": dict(
                    zip(self.label_encoder.classes_, probabilities)
                ),
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "label": "UNKNOWN",
                "confidence": 0.0,
                "second_confidence": 0.0,
                "certainty": "ERROR",
                "action": "ERROR",
                "probabilities": {},
            }

    def capture_worker(self, interface: str = "eth0") -> None:
        """
        Continuously capture packets.
        """
        consecutive_errors = 0
        max_errors = 10

        while not self.stop_event.is_set():
            try:
                traffic = self.optimized_capture_traffic(
                    interface=interface, duration=1
                )

                if traffic:
                    for packet in traffic:
                        try:
                            self.packet_queue.put(packet, timeout=1)
                        except queue.Full:
                            logger.warning("Packet queue full - dropping packets")

                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    f"Capture error ({consecutive_errors}/{max_errors}): {e}"
                )

                if consecutive_errors >= max_errors:
                    logger.critical("Max capture errors - pausing capture briefly")
                    consecutive_errors = 0

                time.sleep(1)

    def detection_worker(self, worker_id: int = 0) -> None:
        """
        Process packets and generate predictions.
        """
        while not self.stop_event.is_set():
            try:
                packet_batch = []
                while len(packet_batch) < 10:
                    try:
                        packet = self.packet_queue.get(timeout=0.5)
                        packet_batch.append(packet)
                    except queue.Empty:
                        break

                if not packet_batch:
                    continue

                feature_tuples = self.extract_features_stateful(packet_batch)
                logger.debug(
                    f"Worker {worker_id}: {len(feature_tuples)} flows "
                    f"from {len(packet_batch)} packets"
                )

                for features, flow_id in feature_tuples:
                    start_time = time.time()
                    try:
                        prediction = self.predict_with_confidence(features)
                        detection_time = time.time() - start_time

                        with self.lock:
                            self.packets_processed += 1
                            self.detection_times.append(detection_time)

                        if prediction.get("action") == "ERROR":
                            continue

                        result = {
                            "label": prediction["label"],
                            "confidence": prediction["confidence"],
                            "action": prediction["action"],
                            "certainty": prediction["certainty"],
                            "flow_id": flow_id,
                            "timestamp": datetime.utcnow(),
                            "detection_time": detection_time,
                        }

                        try:
                            self.prediction_queue.put(result, timeout=1)
                        except queue.Full:
                            logger.warning(
                                "Prediction queue full - dropping prediction"
                            )

                        if self.external_predictions is not None:
                            with self.lock:
                                self.external_predictions.append(result)

                    except Exception as e:
                        logger.error(f"Detection error: {e}")

            except Exception as e:
                logger.error(f"Detection worker error: {e}")

    def alert_worker(self, confidence_threshold: float = 0.85) -> None:
        """
        Process predictions and generate aggregated alerts.
        """
        last_cleanup = time.time()

        while not self.stop_event.is_set():
            try:
                result = self.prediction_queue.get(timeout=2)

                if result["action"] == "PASS":
                    continue

                alert_key = (result["label"], result["flow_id"])
                current_time = time.time()

                alert_info = self.recent_alerts[alert_key]
                alert_info["count"] += 1
                alert_info["last_seen"] = current_time

                if alert_info["first_seen"] is None:
                    alert_info["first_seen"] = current_time

                if alert_info["count"] == 1:
                    severity = self._classify_severity(
                        result["label"], result["confidence"]
                    )
                    logger.warning(
                        f"ALERT: {result['label']} "
                        f"(Confidence: {result['confidence']:.2%}, "
                        f"Severity: {severity})"
                    )

                    with self.lock:
                        self.alerts_generated += 1
                        if self.external_alerts is not None:
                            self.external_alerts.append(
                                {
                                    "label": result["label"],
                                    "confidence": result["confidence"],
                                    "severity": severity,
                                    "flow_id": result["flow_id"],
                                    "certainty": result["certainty"],
                                    "timestamp": result["timestamp"],
                                }
                            )

                if current_time - last_cleanup > self.alert_aggregation_window:
                    expired_keys = [
                        k
                        for k, v in self.recent_alerts.items()
                        if current_time - v["last_seen"]
                        > self.alert_aggregation_window
                    ]

                    for k in expired_keys:
                        if self.recent_alerts[k]["count"] > 1:
                            logger.info(
                                f"Alert Summary: {k[0]} occurred "
                                f"{self.recent_alerts[k]['count']} times"
                            )
                        del self.recent_alerts[k]

                    last_cleanup = current_time

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Alert worker error: {e}")

    def _classify_severity(self, attack_type: str, confidence: float) -> str:
        """
        Classify alert severity.
        """
        if confidence < 0.7:
            return "LOW"
        elif "DDoS" in attack_type or "Brute Force" in attack_type:
            return "CRITICAL" if confidence > 0.95 else "HIGH"
        elif "Infiltration" in attack_type or "Backdoor" in attack_type:
            return "CRITICAL"
        else:
            return "MEDIUM"

    def start_realtime_detection(
        self, num_workers: int = 2, interface: str = "eth0"
    ) -> None:
        """
        Start capture, detection, and alert threads.
        """
        self.stop_event.clear()
        logger.info("Starting real-time detection system...")

        self.capture_thread = Thread(
            target=self.capture_worker,
            kwargs={"interface": interface},
            daemon=True,
            name="PacketCapture",
        )
        self.capture_thread.start()
        logger.info("Packet capture thread started")

        self.detection_threads = []
        for i in range(num_workers):
            t = Thread(
                target=self.detection_worker,
                kwargs={"worker_id": i},
                daemon=True,
                name=f"DetectionWorker-{i}",
            )
            t.start()
            self.detection_threads.append(t)
        logger.info(f"Started {num_workers} detection worker threads")

        self.alert_thread = Thread(
            target=self.alert_worker, daemon=True, name="AlertProcessor"
        )
        self.alert_thread.start()
        logger.info("Alert processing thread started")

    def stop_realtime_detection(self, timeout: int = 10) -> None:
        """
        Gracefully stop all detection threads.
        """
        logger.info("Stopping real-time detection...")
        self.stop_event.set()

        threads_to_join = [self.capture_thread] + self.detection_threads + [
            self.alert_thread
        ]

        for t in threads_to_join:
            if t and t.is_alive():
                t.join(timeout=timeout)

        logger.info("Real-time detection stopped")
        self.print_final_statistics()

    def get_metrics(self) -> Dict:
        """
        Get current performance metrics.
        """
        with self.lock:
            elapsed = (datetime.utcnow() - self.start_time).total_seconds()
            return {
                "packets_captured": self.packets_captured,
                "packets_processed": self.packets_processed,
                "alerts_generated": self.alerts_generated,
                "packets_per_second": self.packets_captured / max(elapsed, 1),
                "avg_detection_time_ms": np.mean(self.detection_times[-100:])
                * 1000
                if self.detection_times
                else 0.0,
                "uptime_seconds": elapsed,
                "cpu_percent": self.process.cpu_percent(interval=0.1),
                "memory_mb": self.process.memory_info().rss / 1024 / 1024,
            }

    def print_final_statistics(self) -> None:
        """
        Print final run statistics.
        """
        metrics = self.get_metrics()
        logger.info("=" * 70)
        logger.info("FINAL IDS STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Packets Captured: {metrics['packets_captured']}")
        logger.info(f"Packets Processed: {metrics['packets_processed']}")
        logger.info(f"Alerts Generated: {metrics['alerts_generated']}")
        logger.info(f"Throughput: {metrics['packets_per_second']:.2f} packets/sec")
        logger.info(
            f"Avg Detection Time: {metrics['avg_detection_time_ms']:.2f} ms"
        )
        logger.info(f"Uptime: {metrics['uptime_seconds']:.2f} seconds")
        logger.info(f"CPU Usage: {metrics['cpu_percent']:.1f}%")
        logger.info(f"Memory Usage: {metrics['memory_mb']:.2f} MB")
        logger.info("=" * 70)


# ======================= MAIN FUNCTIONS =======================


def train_mode(data_path: str, model_type: str = "rf"):
    """
    Training mode: load data, preprocess, train, evaluate.
    """
    logger.info("Starting IDS Training Mode")
    logger.info("=" * 70)

    ids = IntrusionDetectionSystem()

    try:
        X, y = ids.load_data(data_path)
        X_train, X_test, y_train, y_test = ids.preprocess_and_split(X, y)

        ids.train_model(X_train, y_train, model_type=model_type)
        metrics = ids.evaluate_model(X_test, y_test)

        logger.info("=" * 70)
        logger.info("Training completed successfully!")
        logger.info(f"Training metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def realtime_mode(num_workers: int = 2, interface: str = "eth0"):
    """
    Real-time detection mode: continuous detection loop.
    """
    logger.info("Starting IDS Real-Time Detection Mode")
    logger.info("=" * 70)

    ids = RealtimeIDSOptimized(
        model_path="ids_model.pkl",
        scaler_path="scaler.pkl",
        encoder_path="label_encoder.pkl",
        buffer_size=2000,
    )

    try:
        ids.load_components()
        logger.info("ML-IDS components loaded successfully")

        ids.start_realtime_detection(num_workers=num_workers, interface=interface)

        # Keep main thread alive until interrupted
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Interrupt received - shutting down gracefully")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        ids.stop_realtime_detection()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ML-Based Intrusion Detection System"
    )
    parser.add_argument(
        "mode",
        choices=["train", "realtime"],
        help="Mode: train (model training) or realtime (live detection)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="MERGED_ALL_DATASETS.csv",
        help="Path to training dataset (for train mode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["rf", "svm"],
        default="rf",
        help="Model type: rf (Random Forest) or svm (SVM)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of detection workers (for realtime mode)",
    )
    parser.add_argument(
        "--interface",
        type=str,
        default="eth0",
        help="Network interface to capture from (for realtime mode)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        logger.info(f"Training with {args.model.upper()} model on {args.data}")
        train_mode(args.data, model_type=args.model)

    elif args.mode == "realtime":
        logger.info(
            f"Starting real-time detection with {args.workers} workers "
            f"on {args.interface}"
        )
        realtime_mode(num_workers=args.workers, interface=args.interface)


if __name__ == "__main__":
    main()