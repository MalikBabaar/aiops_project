import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ---------------- CONFIG ---------------- #
API_URL = "http://localhost:5000"  # FastAPI backend URL

st.set_page_config(page_title="AIOps Dashboard", layout="wide")


# ---------------- Helper function ---------------- #
def get_data(endpoint):
    try:
        r = requests.get(f"{API_URL}{endpoint}")
        return r.json() if r.status_code == 200 else {}
    except Exception as e:
        return {"error": str(e)}


# ---------------- SESSION STATE ---------------- #
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "email" not in st.session_state:
    st.session_state.email = ""


# ---------------- LOGIN PAGE ---------------- #
if not st.session_state.authenticated:
    st.title("🔐 Login to AIOps Dashboard")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        response = requests.post(f"{API_URL}/login", json={"email": email, "password": password})
        if response.status_code == 200:
            st.session_state.authenticated = True
            st.session_state.email = email
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid email or password")

# ---------------- MAIN DASHBOARD ---------------- #
else:
    # --- Sidebar ---
    st.sidebar.title("LogIn")
    st.sidebar.write(f"Welcome, {st.session_state.email}")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    # --- Dashboard Tabs ---
    tabs = st.tabs([
        "Overview",
        "System Monitoring",
        "ML Model Monitoring",
        "Anomalies",
        "Events & Alerts",
        "Event Grouping",
        "Users & Teams"
    ])

    # --- OVERVIEW TAB --- #
    with tabs[0]:
        st.title("AIOps Dashboard")
        overview = get_data("/overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Logs Processed", f"{overview.get('total_logs', 0):,}")
        col2.metric("Anomalies Detected", f"{overview.get('anomalies', 0):,}")
        col3.metric("Models Deployed", f"{overview.get('models_deployed', 0):,}")
        col4.metric("Active Users", f"{overview.get('active_users', 0):,}")

        st.markdown("---")

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
                        if result.get("is_anomaly"):
                            st.error("⚠️ Anomaly Detected")
                        else:
                            st.success("✅ Log is Normal")
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

        st_autorefresh(interval=60000, key="system_monitor_refresh")

        if "stats_history" not in st.session_state:
            st.session_state.stats_history = {
                "time": [],
                "cpu": [],
                "memory": [],
                "disk": [],
                "net_sent": [],
                "net_recv": []
            }

        system_stats = get_data("/system-stats")

        if "error" not in system_stats:
            st.session_state.stats_history["time"].append(time.strftime("%H:%M:%S"))
            st.session_state.stats_history["cpu"].append(system_stats["cpu"])
            st.session_state.stats_history["memory"].append(system_stats["memory"])
            st.session_state.stats_history["disk"].append(system_stats["disk"])
            st.session_state.stats_history["net_sent"].append(system_stats["network_sent"])
            st.session_state.stats_history["net_recv"].append(system_stats["network_recv"])

            max_points = 30
            for key in st.session_state.stats_history:
                st.session_state.stats_history[key] = st.session_state.stats_history[key][-max_points:]

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("CPU Usage (%)", f"{system_stats['cpu']:.2f}")
            col2.metric("Memory Usage (%)", f"{system_stats['memory']:.2f}")
            col3.metric("Disk Usage (%)", f"{system_stats['disk']:.2f}")
            col4.metric("Network Sent (bytes)", f"{system_stats['network_sent']:,}")
            col5.metric("Network Received (bytes)", f"{system_stats['network_recv']:,}")

        if len(st.session_state.stats_history["time"]) > 0:
            df_stats = pd.DataFrame(st.session_state.stats_history)

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
            st.plotly_chart(fig_usage, config={"responsive": True}, key="usage_chart")

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
            st.plotly_chart(fig_net, config={"responsive": True}, key="network_chart")

    # --- ML MODEL MONITORING TAB --- #
    with tabs[2]:
        st.title("ML Model Monitoring")
        if st.button("Get Model Info"):
            response = requests.get(f"{API_URL}/model-info")
            if response.status_code == 200:
                st.json(response.json())
            else:
                st.error(f"Error fetching model info: {response.status_code} - {response.text}")

        st.markdown("### Retrain Model with Logs")

        uploaded_file = st.file_uploader("Upload logs file (CSV, TXT, LOG)", type=["csv", "txt", "log"])
        if uploaded_file is not None:
            df = None  # ✅ make sure df always defined

            if uploaded_file.name.endswith(".csv"):
                try:
                    df = pd.read_csv(uploaded_file)
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
                    st.stop()
            else:
                try:
                    lines = uploaded_file.read().decode("utf-8").splitlines()
                    df = pd.DataFrame(lines, columns=["log"])
                except Exception as e:
                    st.error(f"Error reading text file: {e}")
                    st.stop()

            # ✅ only continue if df was created successfully
            if df is not None:
                if "log" not in df.columns:
                    possible_cols = [c for c in df.columns if "log" in c.lower() or "message" in c.lower()]
                    if possible_cols:
                        df.rename(columns={possible_cols[0]: "log"}, inplace=True)
                    else:
                        st.error("File must contain a 'log' column (or similar like 'message').")
                        st.stop()

            if "label" in df.columns:
                logs = [{"log": row["log"], "label": int(row["label"])} for _, row in df.iterrows()]
            else:
                logs = [{"log": l} for l in df["log"].tolist()]

            st.write("Preview of uploaded logs:")
            st.dataframe(df.head(), width="stretch")

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
            st.dataframe(df, width="stretch")
        else:
            st.info("No anomalies detected yet.")

    # --- EVENTS & ALERTS TAB --- #
    with tabs[4]:
        st.title("Alert Management")

        events = get_data("/events")
        if isinstance(events, list) and len(events) > 0:
            df = pd.DataFrame(events)
            # --- Extract only the actual log message ---
            def extract_message(full_log):
                # Example pattern: "[timestamp] [level] message"
                parts = full_log.split("] ")
                if len(parts) >= 3:
                    return "] ".join(parts[2:]).strip()
                elif len(parts) >= 2:
                    return parts[1].strip()
                else:
                    return full_log.strip()

            df["message"] = df["message"].apply(extract_message)
            severity_icons = {
                "critical": "🔴 Critical",
                "warning": "🟠 Warning",
                "info": "🟢 Info"
            }
            df["severity"] = df["severity"].apply(lambda x: severity_icons.get(str(x).lower(), x))
            st.subheader("Alert History")
            st.dataframe(df, width="stretch")

            st.markdown("---")
            st.subheader("Acknowledge Alert")

            active_alerts = df[df["status"] == "active"]["timestamp"].tolist()
            if active_alerts:
                selected_event = st.selectbox("Select Active Alert", active_alerts)
                if st.button("Acknowledge Selected Alert"):
                    response = requests.post(f"{API_URL}/events/ack/{selected_event}")
                    if response.status_code == 200:
                        st.success(response.json().get("message", "Alert acknowledged"))
                        st.rerun()
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
            else:
                st.info("No active alerts to acknowledge.")
        else:
            st.info("No events or alerts available.")

    # --- GROUP EVENTS FROM LOGS TAB --- #
        with tabs[5]:
            st.title("Group Events from Logs")
            st.markdown("Upload logs (CSV or TXT) to automatically group similar events")

            uploaded_file = st.file_uploader("Upload logs file (CSV or TXT)", type=["csv", "txt"])

            if uploaded_file is not None:
                try:
                    # Read the file (CSV or TXT)
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        lines = uploaded_file.read().decode("utf-8").splitlines()
                        df = pd.DataFrame({"message": lines})

                    # Add missing columns
                    if "timestamp" not in df.columns:
                        df["timestamp"] = datetime.now().isoformat()
                    if "level" not in df.columns:
                        df["level"] = "INFO"

                    st.subheader("Preview of Uploaded Logs")
                    st.dataframe(df.head(), width="stretch")

                    logs = df.to_dict(orient="records")

                    # Only run this when button is clicked
                    if st.button("🚀 Group Events"):
                        with st.spinner("Grouping events..."):
                            try:
                                response = requests.post(f"{API_URL}/group-events", json=logs)
                                if response.status_code == 200:
                                    grouped = response.json()
                                    if grouped:
                                        st.success(f"✅ Found {len(grouped)} Event Groups")
                                        grouped_df = pd.DataFrame(grouped)
                                        st.dataframe(grouped_df, width="stretch")
                                        st.bar_chart(grouped_df.set_index("event_type")["count"])
                                    else:
                                        st.info("No event groups found.")
                                else:
                                    st.error(f"Error {response.status_code}: {response.text}")
                            except Exception as e:
                                st.error(f"❌ API request failed: {e}")

                except Exception as e:
                    st.error(f"❌ Failed to process log file: {e}")

            else:
                st.info("📤 Please upload a log file to start grouping.")

    # --- USERS & TEAMS TAB --- #
    with tabs[6]:
        st.title("Users & Teams Management")
        st.write("Manage user roles, team assignments, and permissions.")

        backend_url = API_URL

        # --- Add New User Section --- #
        st.subheader("➕ Add New User")
        with st.form("add_user_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email Address")
            role = st.selectbox("Role", ["admin", "analyst", "viewer"])
            team = st.text_input("Team Name")
            submit_user = st.form_submit_button("Add User")

        if submit_user:
            if not (name and email and team):
                st.warning("Please fill in all required fields.")
            else:
                payload = {"name": name, "email": email, "role": role, "team": team}
                res = requests.post(f"{backend_url}/users", json=payload)
                if res.status_code == 200:
                    st.success(f"✅ User '{name}' added successfully.")
                else:
                    st.error(f"❌ Failed to add user: {res.text}")

        st.divider()

        # --- View All Users --- #
        st.subheader("👥 Current Users")
        res = requests.get(f"{backend_url}/users")
        if res.status_code == 200:
            users = res.json()
            if users:
                df_users = pd.DataFrame(users)
                teams = sorted(df_users["team"].unique())
                selected_team = st.selectbox("Filter by Team", ["All"] + teams)
                if selected_team != "All":
                    df_users = df_users[df_users["team"] == selected_team]

                st.dataframe(df_users, width="stretch")

                st.subheader("🗑️ Delete a User")
                delete_email = st.selectbox("Select User to Delete", df_users["email"].tolist())
                if st.button("Delete User"):
                    del_res = requests.delete(f"{backend_url}/users/{delete_email}")
                    if del_res.status_code == 200:
                        st.success(f"User {delete_email} deleted successfully.")
                    else:
                        st.error(f"Error deleting user: {del_res.text}")
            else:
                st.info("No users found yet.")
        else:
            st.error("⚠️ Failed to load users from backend.")
