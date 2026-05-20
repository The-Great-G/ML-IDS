import os
import time
from datetime import datetime
from typing import Dict, List, Optional


import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


try:
    # Optional: if installed, use for smoother auto-refresh
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None  # Fallback to manual refresh pattern


from ids import IntrusionDetectionSystem, RealtimeIDSOptimized



# =========================
# Session State Management
# =========================


def init_session_state():
    if "ids_trainer" not in st.session_state:
        st.session_state["ids_trainer"] = None


    if "ids_realtime" not in st.session_state:
        st.session_state["ids_realtime"] = None


    if "realtime_running" not in st.session_state:
        st.session_state["realtime_running"] = False


    if "alerts" not in st.session_state:
        st.session_state["alerts"] = []


    if "predictions" not in st.session_state:
        st.session_state["predictions"] = []


    if "metrics_history" not in st.session_state:
        # List of dicts: {"timestamp": ..., "packets_per_second": ..., "alerts_generated": ..., "cpu_percent": ...}
        st.session_state["metrics_history"] = []


    if "selected_interface" not in st.session_state:
        st.session_state["selected_interface"] = "eth0"


    if "num_workers" not in st.session_state:
        st.session_state["num_workers"] = 2



# =========================
# Helper Functions
# =========================


def get_realtime_metrics() -> Optional[Dict]:
    ids_rt: Optional[RealtimeIDSOptimized] = st.session_state.get("ids_realtime")
    if ids_rt is None:
        return None
    try:
        return ids_rt.get_metrics()
    except Exception as e:
        st.error(f"Error fetching real-time metrics: {e}")
        return None



def update_metrics_history(metrics: Dict):
    # Maintain simple history for charts
    ts = datetime.utcnow()
    entry = {
        "timestamp": ts,
        "packets_per_second": metrics.get("packets_per_second", 0),
        "alerts_generated": metrics.get("alerts_generated", 0),
        "cpu_percent": metrics.get("cpu_percent", 0),
    }
    st.session_state["metrics_history"].append(entry)
    # Keep last N points to avoid unbounded growth
    if len(st.session_state["metrics_history"]) > 500:
        st.session_state["metrics_history"] = st.session_state["metrics_history"][-500:]



def render_metrics_cards(metrics: Dict):
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)


    col1.metric("Packets Captured", f"{metrics.get('packets_captured', 0):,}")
    col2.metric("Packets Processed", f"{metrics.get('packets_processed', 0):,}")
    col3.metric("Alerts Generated", f"{metrics.get('alerts_generated', 0):,}")
    col4.metric(
        "Throughput (pkts/s)",
        f"{metrics.get('packets_per_second', 0):.2f}",
    )


    col5.metric(
        "Avg Detection Time (ms)",
        f"{metrics.get('avg_detection_time_ms', 0):.2f}",
    )
    col6.metric(
        "Uptime (s)",
        f"{metrics.get('uptime_seconds', 0):.1f}",
    )
    col7.metric(
        "CPU Usage (%)",
        f"{metrics.get('cpu_percent', 0):.1f}",
    )
    col8.metric(
        "Memory Usage (MB)",
        f"{metrics.get('memory_mb', 0):.2f}",
    )



def render_metrics_charts():
    history = st.session_state.get("metrics_history", [])
    if not history:
        st.info("No metrics history yet. Start real-time detection to see charts.")
        return


    df = pd.DataFrame(history)
    df = df.set_index("timestamp")


    st.markdown("#### Performance Trends (Packets/sec & CPU%)")


    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["packets_per_second"],
            mode="lines+markers",
            name="Packets/sec",
            line=dict(color="blue", width=2),
        ),
        secondary_y=False,
    )


    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["cpu_percent"],
            mode="lines",
            name="CPU %",
            line=dict(color="red", width=2, dash="dot"),
        ),
        secondary_y=True,
    )


    fig.update_layout(
        title="Real-time Performance",
        xaxis_title="Time",
        yaxis_title="Packets/sec",
        yaxis2_title="CPU %",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


    st.plotly_chart(fig, use_container_width=True)



def ensure_model_files_exist(
    model_path: str = "ids_model.pkl",
    scaler_path: str = "scaler.pkl",
    encoder_path: str = "label_encoder.pkl",
) -> bool:
    missing = []
    for path in [model_path, scaler_path, encoder_path]:
        if not os.path.exists(path):
            missing.append(path)


    if missing:
        st.warning(
            "Model components not found. Train a model first on the Training page.\n\n"
            f"Missing: {', '.join(missing)}"
        )
        return False
    return True



def start_realtime_detection(interface: str, num_workers: int):
    if st.session_state["realtime_running"]:
        st.info("Real-time detection is already running.")
        return


    if not ensure_model_files_exist():
        return


    if st.session_state["ids_realtime"] is None:
        st.session_state["alerts"] = []
        st.session_state["predictions"] = []


        st.session_state["ids_realtime"] = RealtimeIDSOptimized(
            model_path="ids_model.pkl",
            scaler_path="scaler.pkl",
            encoder_path="label_encoder.pkl",
            buffer_size=2000,
            external_predictions=st.session_state["predictions"],
            external_alerts=st.session_state["alerts"],
        )


    ids_rt: RealtimeIDSOptimized = st.session_state["ids_realtime"]
    try:
        ids_rt.load_components()
    except Exception as e:
        st.error(f"Failed to load model components for real-time IDS: {e}")
        return


    try:
        ids_rt.start_realtime_detection(num_workers=num_workers, interface=interface)
        st.session_state["realtime_running"] = True
        st.session_state["selected_interface"] = interface
        st.session_state["num_workers"] = num_workers
        st.success(
            f"Started real-time detection on interface '{interface}' "
            f"with {num_workers} worker(s)."
        )
    except Exception as e:
        st.error(f"Failed to start real-time detection: {e}")



def stop_realtime_detection():
    if not st.session_state["realtime_running"]:
        st.info("Real-time detection is not running.")
        return


    ids_rt: Optional[RealtimeIDSOptimized] = st.session_state.get("ids_realtime")
    if ids_rt is None:
        st.session_state["realtime_running"] = False
        st.warning("Real-time IDS instance not found, resetting state.")
        return


    try:
        ids_rt.stop_realtime_detection()
        st.session_state["realtime_running"] = False
        st.success("Real-time detection stopped.")
    except Exception as e:
        st.error(f"Error stopping real-time detection: {e}")



def build_alert_reason(alert: Dict) -> str:
    label = alert.get("label", "UNKNOWN")
    confidence = alert.get("confidence", 0.0)
    certainty = alert.get("certainty", "UNKNOWN")
    severity = alert.get("severity", "UNKNOWN")


    reason = (
        f"Detected '{label}' with confidence {confidence:.2%} "
        f"({certainty} certainty), classified as {severity} severity. "
        "The detection engine raised this alert because the predicted attack "
        "confidence exceeded the internal alerting threshold and the label "
        "is not considered BENIGN."
    )
    return reason



def build_alert_location(alert: Dict) -> str:
    flow_id = alert.get("flow_id", "unknown")
    interface = st.session_state.get("selected_interface", "unknown")


    protocol = "unknown"
    src_port = "unknown"


    if isinstance(flow_id, str) and ":" in flow_id:
        proto_id, port_str = flow_id.split(":", 1)
        src_port = port_str or "unknown"
        if proto_id == "6":
            protocol = "TCP"
        elif proto_id == "17":
            protocol = "UDP"
        else:
            protocol = proto_id or "unknown"


    location = (
        f"Check traffic for protocol {protocol} on source port {src_port} "
        f"associated with flow ID {flow_id} on interface {interface}."
    )
    return location



def alerts_to_dataframe(alerts: List[Dict]) -> pd.DataFrame:
    if not alerts:
        return pd.DataFrame()


    rows = []
    for a in alerts:
        ts = a.get("timestamp")
        if isinstance(ts, datetime):
            ts_str = ts.isoformat()
        else:
            ts_str = str(ts)


        row = {
            "Timestamp": ts_str,
            "Label": a.get("label", "UNKNOWN"),
            "Confidence": a.get("confidence", 0.0),
            "Certainty": a.get("certainty", "UNKNOWN"),
            "Severity": a.get("severity", "UNKNOWN"),
            "Flow ID": a.get("flow_id", "unknown"),
        }
        row["Reason"] = build_alert_reason(a)
        row["Where to Investigate"] = build_alert_location(a)
        rows.append(row)


    df = pd.DataFrame(rows)
    return df.sort_values("Timestamp", ascending=False)



# =========================
# Page Implementations
# =========================


def show_dashboard():
    st.title("Intrusion Detection System Dashboard")


    if st.session_state.get("realtime_running") and st_autorefresh is not None:
        st_autorefresh(interval=2000, key="dashboard_autorefresh")


    metrics = get_realtime_metrics()
    if metrics is None:
        st.info("Start real-time detection to see live metrics.")
        return


    render_metrics_cards(metrics)
    update_metrics_history(metrics)
    render_metrics_charts()



def show_training_page():
    st.title("Model Training")


    st.markdown(
        "Train the IDS model on a labeled network traffic dataset. "
        "After training, the model, scaler, and encoder will be saved and "
        "used by the real-time detection engine."
    )


    with st.expander("Dataset Source", expanded=True):
        data_path = st.text_input(
            "Training CSV file path (on server)",
            value="MERGED_ALL_DATASETS.csv",
            help="Path to CSV file accessible from the machine running this app.",
        )
        uploaded_file = st.file_uploader(
            "Or upload a CSV file", type=["csv"], accept_multiple_files=False
        )


    model_type = st.selectbox(
        "Model Type",
        options=["rf", "svm"],
        index=0,
        help="rf = Random Forest, svm = Support Vector Machine.",
    )


    start_train = st.button("Start Training", type="primary")


    if start_train:
        if not data_path and not uploaded_file:
            st.warning("Please provide a dataset path or upload a CSV file.")
            return

        # Decide which path to use
        if uploaded_file is not None:
            tmp_path = "uploaded_training_dataset.csv"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            data_path_to_use = tmp_path
        else:
            data_path_to_use = data_path

        # Use the backend IDS class
        ids_trainer = IntrusionDetectionSystem(
            model_path="ids_model.pkl",
            scaler_path="scaler.pkl",
            encoder_path="label_encoder.pkl",
        )
        st.session_state["ids_trainer"] = ids_trainer

        with st.status("Training model...", expanded=True) as status:
            try:
                st.write(f"Loading data from: `{data_path_to_use}`")
                X, y = ids_trainer.load_data(data_path_to_use)

                st.write("Preprocessing data (split, scale, encode)...")
                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                ) = ids_trainer.preprocess_and_split(X, y)

                st.write(f"Training {model_type.upper()} model...")
                ids_trainer.train_model(X_train, y_train, model_type=model_type)

                st.write("Evaluating model on test set...")
                metrics = ids_trainer.evaluate_model(X_test, y_test)

                status.update(label="Training completed!", state="complete")

                st.success("Model training finished successfully.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                c2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                c3.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                c4.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")

                st.info(
                    "Model, scaler, and label encoder have been saved and are "
                    "ready for use in real-time detection."
                )

            except Exception as e:
                status.update(label="Training failed.", state="error")
                st.error(f"Training failed: {e}")



def show_realtime_page():
    st.title("Real-Time Monitoring")


    st.markdown(
        "Control real-time network traffic monitoring and view live alerts and metrics."
    )


    with st.sidebar:
        st.markdown("### Real-Time Controls")
        interface = st.text_input(
            "Network Interface",
            value=st.session_state.get("selected_interface", "eth0"),
            help="Example: eth0, wlan0, en0, etc.",
        )
        num_workers = st.number_input(
            "Number of Detection Workers",
            min_value=1,
            max_value=16,
            value=st.session_state.get("num_workers", 2),
            step=1,
        )
        start_button = st.button("Start Detection", type="primary")
        stop_button = st.button("Stop Detection")


    if start_button:
        start_realtime_detection(interface=interface, num_workers=num_workers)


    if stop_button:
        stop_realtime_detection()


    if st.session_state.get("realtime_running") and st_autorefresh is not None:
        st_autorefresh(interval=2000, key="realtime_autorefresh")


    st.markdown("### Live Metrics")
    metrics = get_realtime_metrics()
    if metrics is not None:
        render_metrics_cards(metrics)
        update_metrics_history(metrics)
        render_metrics_charts()
    else:
        st.info("No metrics available. Start detection to see live metrics.")


    st.markdown("### Recent Alerts")
    alerts = st.session_state.get("alerts", [])
    if not alerts:
        st.info("No alerts generated yet.")
    else:
        df_alerts = alerts_to_dataframe(alerts)
        st.dataframe(df_alerts, use_container_width=True)



def show_logs_page():
    st.title("Logs & Historical Alerts")


    st.markdown(
        "Review IDS logs and historical alerts. All important events and alerts "
        "are logged for auditing and investigation."
    )


    log_path = "ids.log"
    col1, col2 = st.columns([3, 1])


    with col1:
        st.markdown("### Log Output (Tail)")


        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                tail_lines = lines[-300:] if len(lines) > 300 else lines
                log_text = "".join(tail_lines)
                st.text_area(
                    "Recent Log Entries",
                    value=log_text,
                    height=400,
                    disabled=True,
                )
            except Exception as e:
                st.error(f"Error reading log file: {e}")
        else:
            st.info("No log file found yet (ids.log).")


    with col2:
        st.markdown("### Download Logs")
        if os.path.exists(log_path):
            try:
                with open(log_path, "rb") as f:
                    log_bytes = f.read()
                st.download_button(
                    "Download Full Log",
                    data=log_bytes,
                    file_name="ids.log",
                    mime="text/plain",
                )
            except Exception as e:
                st.error(f"Error preparing log download: {e}")
        else:
            st.info("Logs will be available once the IDS has run.")


    st.markdown("### Historical Alerts")


    alerts = st.session_state.get("alerts", [])
    if not alerts:
        st.info("No alerts recorded yet.")
        return


    df_alerts = alerts_to_dataframe(alerts)


    with st.expander("Filter Alerts", expanded=False):
        unique_severity = sorted(df_alerts["Severity"].unique())
        unique_labels = sorted(df_alerts["Label"].unique())
        sel_severity = st.multiselect(
            "Severity",
            options=unique_severity,
            default=unique_severity,
        )
        sel_labels = st.multiselect(
            "Attack Label",
            options=unique_labels,
            default=unique_labels,
        )


    mask = df_alerts["Severity"].isin(sel_severity) & df_alerts["Label"].isin(
        sel_labels
    )
    st.dataframe(df_alerts[mask], use_container_width=True)



def show_predict_sample_page():
    st.title("Predict Single Network Flow")


    st.markdown(
        "Use the trained real-time model to predict the label and action for a single flow."
    )


    if not ensure_model_files_exist():
        return


    if st.session_state["ids_realtime"] is None:
        st.session_state["alerts"] = []
        st.session_state["predictions"] = []
        st.session_state["ids_realtime"] = RealtimeIDSOptimized(
            model_path="ids_model.pkl",
            scaler_path="scaler.pkl",
            encoder_path="label_encoder.pkl",
            buffer_size=2000,
            external_predictions=st.session_state["predictions"],
            external_alerts=st.session_state["alerts"],
        )


    ids_realtime: RealtimeIDSOptimized = st.session_state["ids_realtime"]


    try:
        ids_realtime.load_components()
    except Exception as e:
        st.error(f"Could not load model components; please retrain first. Error: {e}")
        return


    st.markdown("Enter sample features from one flow below:")


    user_features = {}
    for feature in ids_realtime.ids.features:
        user_features[feature] = st.number_input(feature, value=0.0, format="%f")


    confidence_threshold = st.slider(
        "Alert confidence threshold", min_value=0.0, max_value=1.0, value=0.85
    )


    if st.button("Run prediction"):
        try:
            feature_vector = [user_features[f] for f in ids_realtime.ids.features]
            prediction = ids_realtime.predict_with_confidence(
                feature_vector, confidence_threshold=confidence_threshold
            )


            st.write("### Prediction result")
            st.write(f"Label: {prediction['label']}")
            st.write(f"Action: {prediction['action']}")
            st.write(f"Certainty: {prediction['certainty']}")
            st.write(f"Confidence: {prediction['confidence']:.4f}")
            st.write("Probability distribution:")
            st.json(prediction["probabilities"])
        except Exception as e:
            st.error(f"Prediction failed: {e}")



# =========================
# Main App
# =========================


def main():
    st.set_page_config(
        page_title="ML-Based Intrusion Detection System",
        layout="wide",
        initial_sidebar_state="expanded",
    )


    init_session_state()


    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=[
            "Dashboard",
            "Training",
            "Real-Time Monitoring",
            "Logs & Alerts",
            "Predict Sample",
        ],
    )


    if page == "Dashboard":
        show_dashboard()
    elif page == "Training":
        show_training_page()
    elif page == "Real-Time Monitoring":
        show_realtime_page()
    elif page == "Logs & Alerts":
        show_logs_page()
    elif page == "Predict Sample":
        show_predict_sample_page()



if __name__ == "__main__":
    main()