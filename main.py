from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data-4-cleaned.csv"
MODEL_PATH = ROOT / "knn_model.joblib"
SCALER_PATH = ROOT / "scaler.joblib"

MODEL_FEATURES = [
    "total_keystrokes",
    "inter_key_std_ms",
    "backspace_ratio",
    "paste_char_ratio",
    "time_on_problem_min",
    "time_to_first_key_ms",
]

DISPLAY_NAMES = {
    "total_keystrokes": "Total keystrokes",
    "inter_key_std_ms": "Inter-key std (ms)",
    "backspace_ratio": "Backspace ratio",
    "paste_char_ratio": "Paste character ratio",
    "time_on_problem_min": "Time on problem (min)",
    "time_to_first_key_ms": "Time to first key (ms)",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    required = [*MODEL_FEATURES, "ai_score"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    # Repair numeric fields before any filtering, charting, or prediction.
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data[column].replace([np.inf, -np.inf], np.nan)
        data[column] = data[column].fillna(data[column].median())
    data = data.dropna(subset=["ai_score"]).copy()
    data["backspace_ratio"] = data["backspace_ratio"].clip(0, 1)
    data["paste_char_ratio"] = data["paste_char_ratio"].clip(0, 1)
    data["ai_score"] = data["ai_score"].clip(0, 100)

    # Convert any remaining text columns to numeric category codes.
    for column in data.select_dtypes(include=["object", "string", "category"]):
        data[column] = data[column].astype("category").cat.codes
    return data


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1180px; padding-top: 2rem; }
        [data-testid="stAppViewContainer"] { background: #ffffff; color: #172033; }
        [data-testid="stSidebar"] {
            background: #e8edf3;
            border-right: 1px solid #b8c2cf;
        }
        [data-testid="stSidebar"] * { color: #172033; }
        [data-testid="stSidebar"] h2 {
            color: #101828;
            font-size: 1.05rem;
            margin-bottom: .4rem;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label,
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-testid="stSlider"] {
            color: #172033;
        }
        [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input {
            background: #ffffff;
            color: #172033;
            border-color: #8996a6;
        }
        label, [data-testid="stCaptionContainer"] {
            color: #344054 !important;
        }
        h1, h2, h3, p { color: #172033; }
        div[data-testid="stMetric"] {
            border: 1px solid #aeb9c7; border-radius: 10px; padding: 12px;
            background: #f5f7fa;
        }
        .score-card {
            border: 2px solid #1769d1; border-radius: 10px; background: #dbeafe;
            color: #102a56; padding: 13px 16px; font-size: 1.15rem; font-weight: 700;
        }
        [data-testid="stButton"] button {
            background: #1769d1;
            color: #ffffff;
            border: 1px solid #0f4fa8;
        }
        [data-testid="stButton"] button:hover {
            background: #0f4fa8;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar() -> str:
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio("Navigation", ["Explore", "Predict"], label_visibility="collapsed")
        if page == "Explore":
            st.markdown("## Filters")
            st.session_state["min_score"] = st.slider(
                "Min score", min_value=0, max_value=100, value=0
            )
            filter_feature = st.selectbox(
                "Filter rows by feature", MODEL_FEATURES,
                format_func=lambda value: DISPLAY_NAMES[value],
            )
            values = load_data()[filter_feature]
            st.session_state["filter_feature"] = filter_feature
            st.session_state["filter_range"] = st.slider(
                "Feature range",
                min_value=float(values.min()),
                max_value=float(values.max()),
                value=(float(values.min()), float(values.max())),
            )
        else:
            st.markdown("## About")
            st.caption("Score = mean of k nearest submissions.")
    return page


def explore_page(df: pd.DataFrame) -> None:
    st.title("Page 1 · Explore")
    left, right = st.columns([1.35, 1])
    with left:
        query = st.selectbox(
            "Query",
            [
                "score by paste band",
                "score by time on problem",
                "typing pattern analysis",
                "feature correlation with AI score",
                "filtered rows",
            ],
            format_func=lambda value: f"Query: {value}",
        )
    with right:
        st.write("")
        st.caption("Explore historical AI suspicion scores")

    filtered = df.copy()
    min_score = st.session_state.get("min_score", 0)
    filtered = filtered[filtered["ai_score"] >= min_score]
    filter_feature = st.session_state.get("filter_feature", MODEL_FEATURES[0])
    low, high = st.session_state.get(
        "filter_range", (float(df[filter_feature].min()), float(df[filter_feature].max()))
    )
    filtered = filtered[filtered[filter_feature].between(low, high)]

    if query == "score by paste band":
        filtered["paste_band"] = pd.cut(
            filtered["paste_char_ratio"],
            bins=[-np.inf, 0.05, 0.15, 0.30, np.inf],
            labels=["0–5%", "5–15%", "15–30%", "30%+"],
        )
        grouped = (
            filtered.groupby("paste_band", observed=False)["ai_score"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_ai_score", "count": "count"})
        )
        chart_data = grouped[["avg_ai_score"]].rename(columns={"avg_ai_score": "Average AI score"})
        table = grouped.reset_index().rename(columns={"paste_band": "band"})
        x_label = "paste band"
    elif query == "score by time on problem":
        filtered["time_band"] = pd.cut(
            filtered["time_on_problem_min"],
            bins=[-np.inf, 15, 30, 60, np.inf],
            labels=["0–15 min", "15–30 min", "30–60 min", "60+ min"],
        )
        grouped = (
            filtered.groupby("time_band", observed=False)["ai_score"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_ai_score", "count": "count"})
        )
        chart_data = grouped[["avg_ai_score"]].rename(columns={"avg_ai_score": "Average AI score"})
        table = grouped.reset_index().rename(columns={"time_band": "time band"})
        x_label = "time band"
    elif query == "typing pattern analysis":
        st.subheader("Typing pattern analysis")
        st.caption("Average suspicion score by behavioral telemetry bands.")
        pattern_feature = st.selectbox(
            "Typing pattern",
            ["inter_key_std_ms", "backspace_ratio", "paste_char_ratio", "time_to_first_key_ms"],
            format_func=lambda value: DISPLAY_NAMES[value],
        )
        if filtered.empty:
            st.warning("No rows match the current filters.")
            return
        if filtered[pattern_feature].nunique() < 2:
            st.info("Not enough variation in this feature for pattern analysis.")
            return
        bands = pd.qcut(
            filtered[pattern_feature],
            q=min(4, filtered[pattern_feature].nunique()),
            duplicates="drop",
        )
        table = (
            filtered.assign(pattern_band=bands)
            .groupby("pattern_band", observed=False)["ai_score"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_ai_score", "count": "submissions"})
            .reset_index()
        )
        table["pattern_band"] = table["pattern_band"].astype(str)
        st.bar_chart(
            table.set_index("pattern_band")[["avg_ai_score"]],
            y_label="Average AI score",
        )
        st.dataframe(table.round(2), hide_index=True, use_container_width=True)
        return
    elif query == "feature correlation with AI score":
        st.subheader("Correlation between telemetry features and AI score")
        correlations = filtered[[*MODEL_FEATURES, "ai_score"]].corr()["ai_score"].drop("ai_score")
        correlations.index = [DISPLAY_NAMES[name] for name in correlations.index]
        st.bar_chart(correlations.rename("Pearson correlation"), y_label="Correlation")
        st.dataframe(
            correlations.rename("Pearson correlation").to_frame().round(3),
            use_container_width=True,
        )
        return
    else:
        st.subheader("Filtered submissions")
        st.caption(f"{len(filtered)} rows match the selected score and feature filters.")
        if filtered.empty:
            st.warning("No rows match the current filters.")
            return
        graph_feature = st.selectbox(
            "Graph feature",
            MODEL_FEATURES,
            format_func=lambda value: DISPLAY_NAMES[value],
        )
        graph_data = filtered[[graph_feature, "ai_score"]].rename(
            columns={graph_feature: DISPLAY_NAMES[graph_feature], "ai_score": "AI score"}
        )
        st.scatter_chart(graph_data, x=DISPLAY_NAMES[graph_feature], y="AI score")
        st.dataframe(filtered, hide_index=True, use_container_width=True)
        return

    st.subheader(f"Bar chart · avg ai_score by {x_label}")
    st.bar_chart(chart_data, height=230, y_label="Average AI score")
    st.subheader("Result table")
    st.caption(f"{' · '.join(table.columns)}")
    st.dataframe(table.round(2), hide_index=True, use_container_width=True)
    st.caption(f"{len(filtered)} repaired rows are included after filtering.")


def predict_page(df: pd.DataFrame) -> None:
    model, scaler = load_artifacts()
    st.title("Page 2 · Predict")
    st.subheader("Telemetry inputs")
    st.caption("Enter the telemetry captured for one submission.")

    defaults = {
        "total_keystrokes": int(df["total_keystrokes"].median()),
        "inter_key_std_ms": int(df["inter_key_std_ms"].median()),
        "backspace_ratio": float(df["backspace_ratio"].median()),
        "paste_char_ratio": float(df["paste_char_ratio"].median()),
        "time_on_problem_min": float(df["time_on_problem_min"].median()),
        "time_to_first_key_ms": int(df["time_to_first_key_ms"].median()),
    }
    inputs = {}
    columns = st.columns(2)
    for index, feature in enumerate(MODEL_FEATURES):
        with columns[index % 2]:
            if feature in {"backspace_ratio", "paste_char_ratio"}:
                inputs[feature] = st.number_input(
                    DISPLAY_NAMES[feature],
                    min_value=0.0,
                    max_value=1.0,
                    value=defaults[feature],
                    step=0.01,
                    format="%.3f",
                )
            else:
                inputs[feature] = st.number_input(
                    DISPLAY_NAMES[feature],
                    min_value=0.0,
                    value=float(defaults[feature]),
                    step=1.0,
                )

    if st.button("Predict", type="primary", use_container_width=False):
        values = pd.DataFrame([inputs], columns=MODEL_FEATURES)
        scaled = scaler.transform(values)
        prediction = float(model.predict(scaled)[0])
        score = int(np.clip(np.rint(prediction), 0, 100))
        st.session_state["prediction"] = score

    if "prediction" in st.session_state:
        st.markdown(
            f'<div class="score-card">Suspicion score: {st.session_state["prediction"]}/100</div>',
            unsafe_allow_html=True,
        )
        display_neighbor_count = min(4, model._fit_X.shape[0])
        st.subheader(f"{display_neighbor_count} closest past submissions")
        values = pd.DataFrame([inputs], columns=MODEL_FEATURES)
        scaled_input = scaler.transform(values)
        distances, model_indices = model.kneighbors(
            scaled_input, n_neighbors=display_neighbor_count
        )

        # The model was fitted on a training split, so map its internal row
        # indices back to the corresponding rows in the complete CSV.
        training_rows = scaler.inverse_transform(model._fit_X)
        csv_rows = df[MODEL_FEATURES].to_numpy(dtype=float)
        nearest_indices = []
        for training_index in model_indices[0]:
            matches = np.where(
                np.isclose(csv_rows, training_rows[training_index], rtol=1e-5, atol=1e-8).all(axis=1)
            )[0]
            if len(matches) == 0:
                raise ValueError("A model neighbor could not be matched to the current CSV.")
            nearest_indices.append(matches[0])

        nearest = df.iloc[nearest_indices].copy()
        nearest.insert(0, "rank", range(1, len(nearest) + 1))
        nearest.insert(2, "distance", distances[0])
        nearest.insert(3, "used", ["Yes" if rank <= model.n_neighbors else "Context"
                                   for rank in nearest["rank"]])
        nearest = nearest[["rank", "ai_score", "distance", "used"]].rename(
            columns={
                "rank": "#",
                "ai_score": "AI score",
                "distance": "Euclidean distance",
                "used": "Used in prediction",
            }
        )
        st.dataframe(
            nearest.round({"Euclidean distance": 3}),
            hide_index=True,
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(page_title="AI code detector", page_icon="🔎", layout="wide")
    df = load_data()
    inject_styles()
    page = sidebar()
    if page == "Explore":
        explore_page(df)
    else:
        predict_page(df)


if __name__ == "__main__":
    main()
