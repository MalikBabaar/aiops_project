import streamlit as st
import requests

st.title("JBoss AIOps Dashboard")

# Log input
log_input = st.text_area("Enter log to analyze:")
if st.button("Analyze"):
    response = requests.post("http://backend:5000/analyze", json={"log": log_input})
    st.json(response.json())

# Retrain trigger
if st.button("Trigger Retrain"):
    response = requests.post("http://backend:5000/retrain")
    st.json(response.json())
