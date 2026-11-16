
"""EvalScope interactive benchmark analysis dashboard."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from evalscope.data_loader import (
    load_experiment,
    summarize_experiment,
)


st.set_page_config(
    page_title="EvalScope | LLM Evaluation",
    page_icon="📊",
    layout="wide",
)

st.title("📊 EvalScope")
st.caption("Statistical Reliability of LLM Benchmarks")

st.sidebar.header("Dataset")

uploaded = st.sidebar.file_uploader(
    "Upload AgentSecBench experiment JSON",
    type=["json"],
)

default_path = Path("data/agentsecbench_experiment.json")


def load_uploaded_file(uploaded_file):
    """Validate uploaded JSON using the existing importer."""
    content = uploaded_file.getvalue()

    # Limit uploads to 20 MB.
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("Uploaded file exceeds 20 MB.")

    # Check JSON syntax before passing to the importer.
    json.loads(content.decode("utf-8"))

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "experiment.json"
        path.write_bytes(content)
        return load_experiment(path)


try:
    if uploaded is not None:
        df = load_uploaded_file(uploaded)
        dataset_name = uploaded.name
    elif default_path.exists():
        df = load_experiment(default_path)
        dataset_name = default_path.name
    else:
        st.info(
            "Upload an AgentSecBench experiment JSON "
            "to begin analyzing results."
        )
        st.stop()

except (ValueError, OSError, UnicodeError) as exc:
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

st.sidebar.success("Dataset loaded")
st.sidebar.caption(dataset_name)

summary = summarize_experiment(df)

st.subheader("Experiment Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Trials", summary["total_trials"])
col2.metric("Unique Attacks", summary["unique_attacks"])
col3.metric("Source Seeds", summary["source_attacks"])
col4.metric("Completed Trials", summary["completed_trials"])

st.divider()

st.subheader("Trial Completion")

completion = (
    df.groupby("mode")
    .agg(
        total=("attack_id", "size"),
        completed=("completed", "sum"),
    )
    .reset_index()
)

completion["incomplete"] = (
    completion["total"] - completion["completed"]
)

st.bar_chart(
    completion.set_index("mode")[["completed", "incomplete"]]
)

st.subheader("Explore Benchmark Records")

mode_options = sorted(df["mode"].unique().tolist())
category_options = sorted(df["category"].unique().tolist())

selected_modes = st.multiselect(
    "Defense configurations",
    options=mode_options,
    default=mode_options,
)

selected_categories = st.multiselect(
    "Attack categories",
    options=category_options,
    default=category_options,
)

filtered = df[
    df["mode"].isin(selected_modes)
    & df["category"].isin(selected_categories)
].copy()

st.caption(f"Showing {len(filtered)} of {len(df)} trials")

display_columns = [
    "attack_id",
    "source_attack_id",
    "category",
    "mode",
    "attack_success",
    "status",
    "completed",
    "duration_seconds",
]

st.dataframe(
    filtered[display_columns],
    use_container_width=True,
    hide_index=True,
)

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download filtered dataset (CSV)",
    data=csv_data,
    file_name="evalscope_filtered_trials.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "EvalScope | Experimental results are not guarantees "
    "of real-world model or defense performance."
)
