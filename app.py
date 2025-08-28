import streamlit as st
import pandas as pd
from main import run_pipeline

st.title("📂 Data Ingestion & Quality Filtering Pipeline")

if st.button("Run Pipeline on Sample Data"):
    run_pipeline()
    df = pd.read_csv("data/results.csv")
    st.success("Pipeline Completed ✅")
    st.dataframe(df)
    st.download_button("Download Results CSV", df.to_csv(index=False), "results.csv")
