import streamlit as st
import requests
import pandas as pd

# FastAPI backend URL
API_URL = "http://localhost:5000"

st.set_page_config(page_title="AIOps Dashboard", layout="wide")

# ---------------- Helper functions ---------------- #
def get_data(endpoint):
    try:
        r = requests.get(f"{API_URL}{endpoint}")
        return r.json() if r.status_code == 200 else {}
    except Exception as e:
        return {"error": str(e)}

# ---------------- TOP NAVIGATION ---------------- #
tabs = st.tabs([
    "Overview", 
    "System Monitoring", 
    "ML Model Monitoring", 
    "Anomalies", 
    "Events & Alerts", 
    "Users & Teams", 
    "Settings"
])

# --- OVERVIEW TAB --- #
with tabs[0]:
    st.title("AIOps Dashboard")

    # Call FastAPI for overview data
    overview = get_data("/overview")

    # Show metrics as cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Logs Processed", f"{overview.get('total_logs', 0):,}")
    col2.metric("Anomalies Detected", f"{overview.get('anomalies', 0):,}")
    col3.metric("Models Deployed", f"{overview.get('models_deployed', 0):,}")
    col4.metric("Active Users", f"{overview.get('active_users', 0):,}")

    st.markdown("---")

    # --- Quick Analyzer ---
    st.subheader("Quick Analyzer")
    log_entry = st.text_area(
        "Log Entry",
        "[2025-09-22 14:35:17] ERROR [org.jboss.ejb3] – Transaction timeout for UserSessionBean"
    )

    col_cpu, col_mem = st.columns(2)
    cpu_usage = col_cpu.text_input("CPU Usage (%)")
    mem_usage = col_mem.text_input("Memory Usage (%)")

    if st.button("Analyze"):
        if log_entry.strip():
            try:
                response = requests.post(f"{API_URL}/analyze", json={"log": log_entry})
                if response.status_code == 200:
                    result = response.json()
                    st.success("Analysis Result:")

                    # ✅ Highlight anomaly result
                    if result.get("is_anomaly"):
                        st.error("⚠️ Anomaly Detected")
                    else:
                        st.success("✅ Log is Normal")

                    # Show full JSON result
                    st.json(result)

                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Failed to connect API: {e}")
        else:
            st.warning("Please enter a log entry before analyzing.")

    st.markdown("### Event Logs")
    st.info("Event logs will appear here...")

# --- SYSTEM MONITORING TAB --- #
with tabs[1]:
    st.title("System Monitoring")

    import plotly.express as px
    import time
    from streamlit_autorefresh import st_autorefresh

    # Auto-refresh every 1 minute (60000 ms)
    st_autorefresh(interval=60000, key="system_monitor_refresh")

    # Initialize session state for history
    if "stats_history" not in st.session_state:
        st.session_state.stats_history = {
            "time": [],
            "cpu": [],
            "memory": [],
            "disk": [],
            "net_sent": [],
            "net_recv": []
        }

    # Fetch system stats from FastAPI
    system_stats = get_data("/system-stats")

    if "error" not in system_stats:
        # Save history with current timestamp
        st.session_state.stats_history["time"].append(time.strftime("%H:%M:%S"))
        st.session_state.stats_history["cpu"].append(system_stats["cpu"])
        st.session_state.stats_history["memory"].append(system_stats["memory"])
        st.session_state.stats_history["disk"].append(system_stats["disk"])
        st.session_state.stats_history["net_sent"].append(system_stats["network_sent"])
        st.session_state.stats_history["net_recv"].append(system_stats["network_recv"])

        # Keep only last 30 points
        max_points = 30
        for key in st.session_state.stats_history:
            st.session_state.stats_history[key] = st.session_state.stats_history[key][-max_points:]

        # --- Show Current Metrics ---
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("CPU Usage (%)", f"{system_stats['cpu']:.2f}")
        col2.metric("Memory Usage (%)", f"{system_stats['memory']:.2f}")
        col3.metric("Disk Usage (%)", f"{system_stats['disk']:.2f}")
        col4.metric("Network Sent (bytes)", f"{system_stats['network_sent']:,}")
        col5.metric("Network Received (bytes)", f"{system_stats['network_recv']:,}")

    # --- Charts Section ---
    if len(st.session_state.stats_history["time"]) > 0:
        df_stats = pd.DataFrame(st.session_state.stats_history)

        # Chart 1: CPU, Memory, Disk
        df_usage = df_stats.melt(
            id_vars=["time"], 
            value_vars=["cpu", "memory", "disk"],
            var_name="Metric", 
            value_name="Value"
        )
        fig_usage = px.line(
            df_usage, x="time", y="Value", color="Metric",
            title="CPU, Memory, Disk Usage (%)",
            markers=True
        )
        st.plotly_chart(fig_usage, use_container_width=True)

        # Chart 2: Network
        df_net = df_stats.melt(
            id_vars=["time"], 
            value_vars=["net_sent", "net_recv"],
            var_name="Metric", 
            value_name="Value"
        )
        fig_net = px.line(
            df_net, x="time", y="Value", color="Metric",
            title="Network I/O (bytes)",
            markers=True
        )
        st.plotly_chart(fig_net, use_container_width=True)


# --- ML MODEL MONITORING TAB --- #
with tabs[2]:
    st.title("ML Model Monitoring")

    # Get model info from FastAPI
    if st.button("Get Model Info"):
        response = requests.get(f"{API_URL}/model-info")
        if response.status_code == 200:
            st.json(response.json())
        else:
            st.error(f"Error fetching model info: {response.status_code} - {response.text}")

    st.markdown("### Retrain Model with Logs")

    uploaded_file = st.file_uploader("Upload logs file (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Ensure 'log' column exists
        if "log" not in df.columns:
            possible_cols = [c for c in df.columns if "log" in c.lower() or "message" in c.lower()]
            if possible_cols:
                df.rename(columns={possible_cols[0]: "log"}, inplace=True)
            else:
                st.error("CSV must contain a 'log' column (or similar like 'message').")
                st.stop()

        # Prepare logs list for FastAPI
        if "label" in df.columns:
            logs = [{"log": row["log"], "label": int(row["label"])} for _, row in df.iterrows()]
        else:
            logs = [{"log": l} for l in df["log"].tolist()]

        st.write("Preview of uploaded logs:")
        st.dataframe(df.head())

        if st.button("Retrain"):
            try:
                response = requests.post(f"{API_URL}/retrain", json=logs)
                if response.status_code == 200:
                    st.success("Retrain Result:")
                    st.json(response.json())
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Retrain failed: {e}")



# --- ANOMALIES TAB --- #
with tabs[3]:
    st.title("Detected Anomalies")
    history = get_data("/anomaly-history")
    if history and isinstance(history, list) and len(history) > 0:
        df = pd.DataFrame(history)
        st.dataframe(df)
    else:
        st.info("No anomalies detected yet.")

# --- EVENTS & ALERTS TAB --- #
with tabs[4]:
    st.title("Events & Alerts")
    st.write("Event/alert history display here...")

# --- USERS & TEAMS TAB --- #
with tabs[5]:
    st.title("Users & Teams Management")
    st.write("User roles, team assignments...")

# --- SETTINGS TAB --- #
with tabs[6]:
    st.title("Settings")
    st.write("Configuration options...")
