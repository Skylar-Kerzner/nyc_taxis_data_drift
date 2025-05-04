import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Model Performance Over Time")

try:
    df = pd.read_csv("data/metrics.csv")
    st.line_chart(df.set_index("date")["MAE"], use_container_width=True)
    st.line_chart(df.set_index("date")["R2"], use_container_width=True)
except FileNotFoundError:
    st.write("No performance data available yet.")