import streamlit as st
import requests
import pandas as pd

# FastAPI backend URL
API_URL = "http://localhost:5000"   # change if running in Docker or remote server

st.set_page_config(page_title="AIOps Dashboard", layout="wide")

# --- TOP NAVIGATION BAR ---
tabs = st.tabs([
    "Overview", 
    "System Monitoring", 
    "ML Model Monitoring", 
    "Anomalies", 
    "Events & Alerts", 
    "Users & Teams", 
    "Settings"
])

# --- OVERVIEW TAB ---
with tabs[0]:
    st.title("AIOps Dashboard")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Logs Processed", "1,233")
    col2.metric("Anomalies Detected", "5")
    col3.metric("Models Deployed", "1")
    col4.metric("Active Users", "5")

    # Quick Analyzer
    st.subheader("Quick Analyzer")
    log_entry = st.text_area("Log Entry", "[2025-09-22 14:35:17] ERROR [org.jboss.ejb3] – Transaction timeout for UserSessionBean")
    cpu, mem = st.columns(2)
    cpu_usage = cpu.text_input("CPU Usage (%)")
    mem_usage = mem.text_input("Memory Usage (%)")

    if st.button("Analyze"):
        if log_entry.strip():
            try:
                response = requests.post(f"{API_URL}/analyze", json={"log": log_entry})
                if response.status_code == 200:
                    result = response.json()
                    st.json(result)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Failed to connect API: {e}")
        else:
            st.warning("Please enter a log entry before analyzing.")

    st.subheader("Event Logs")
    st.write("Event logs will appear here...")

# --- SYSTEM MONITORING TAB ---
with tabs[1]:
    st.title("System Monitoring")
    st.write("Graphs for CPU, Memory, Disk, Network usage can go here...")

# --- ML MODEL MONITORING TAB ---
with tabs[2]:
    st.title("ML Model Monitoring")
    if st.button("Get Model Info"):
        try:
            response = requests.get(f"{API_URL}/model-info")
            st.json(response.json())
        except Exception as e:
            st.error(f"Error: {e}")

    st.subheader("Retrain Model")
    uploaded_file = st.file_uploader("Upload logs file (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Auto-detect log column
        if "log" not in df.columns:
            possible_cols = [c for c in df.columns if "log" in c.lower() or "message" in c.lower()]
            if possible_cols:
                df.rename(columns={possible_cols[0]: "log"}, inplace=True)
            else:
                st.error("CSV must contain a 'log' column (or something similar like 'message').")
                st.stop()

        logs = [{"log": l} for l in df["log"].tolist()]
        if st.button("Retrain"):
            try:
                response = requests.post(f"{API_URL}/retrain", json=logs)
                st.json(response.json())
            except Exception as e:
                st.error(f"Retrain failed: {e}")

# --- ANOMALIES TAB ---
with tabs[3]:
    st.title("Detected Anomalies")
    st.write("Anomalies dashboard (tables/graphs) can be shown here...")

# --- EVENTS & ALERTS TAB ---
with tabs[4]:
    st.title("Events & Alerts")
    st.write("Event/alert history display here...")

# --- USERS & TEAMS TAB ---
with tabs[5]:
    st.title("Users & Teams Management")
    st.write("User roles, team assignments...")

# --- SETTINGS TAB ---
with tabs[6]:
    st.title("Settings")
    st.write("Configuration options...")
