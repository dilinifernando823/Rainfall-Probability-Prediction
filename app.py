from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"

RAW_FEATURES = [
    "day",
    "pressure",
    "maxtemp",
    "temparature",
    "mintemp",
    "dewpoint",
    "humidity",
    "cloud",
    "sunshine",
    "winddirection",
    "windspeed",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply exactly the same feature engineering used in the notebook."""
    data = df.copy()
    eps = 1e-6

    if "day" in data.columns:
        data["day_sin"] = np.sin(2 * np.pi * data["day"] / 365.0)
        data["day_cos"] = np.cos(2 * np.pi * data["day"] / 365.0)
        data["day_halfyear_sin"] = np.sin(4 * np.pi * data["day"] / 365.0)
        data["day_halfyear_cos"] = np.cos(4 * np.pi * data["day"] / 365.0)

    if "winddirection" in data.columns:
        wind_rad = np.deg2rad(data["winddirection"])
        data["winddirection_sin"] = np.sin(wind_rad)
        data["winddirection_cos"] = np.cos(wind_rad)
        data["winddirection_sector"] = (data["winddirection"] // 45).astype(float)

    data["temp_range"] = data["maxtemp"] - data["mintemp"]
    data["temp_mean_gap"] = data["maxtemp"] - data["temparature"]
    data["temp_min_gap"] = data["temparature"] - data["mintemp"]
    data["dewpoint_gap"] = data["temparature"] - data["dewpoint"]
    data["dewpoint_temp_ratio"] = data["dewpoint"] / (data["temparature"] + eps)
    data["max_dew_spread"] = data["maxtemp"] - data["dewpoint"]

    data["humidity_cloud"] = data["humidity"] * data["cloud"]
    data["humidity_sunshine"] = data["humidity"] * data["sunshine"]
    data["cloud_sunshine_ratio"] = data["cloud"] / (data["sunshine"] + eps)
    data["sunshine_cloud_ratio"] = data["sunshine"] / (data["cloud"] + eps)
    data["cloud_minus_sunshine"] = data["cloud"] - data["sunshine"]
    data["humidity_minus_sunshine"] = data["humidity"] - data["sunshine"]
    data["humidity_plus_cloud"] = data["humidity"] + data["cloud"]
    data["clear_sky_index"] = data["sunshine"] / (data["cloud"] + 1.0)

    data["rain_weather_index"] = (
        0.45 * data["humidity"] + 0.40 * data["cloud"] - 0.35 * data["sunshine"]
    )
    data["dryness_index"] = 100.0 - data["humidity"]
    data["dry_cloud_balance"] = data["dryness_index"] - data["cloud"]
    data["saturation_proxy"] = 1.0 / (data["dewpoint_gap"].abs() + 0.5)

    data["pressure_dev"] = data["pressure"] - 1013.0
    data["pressure_windspeed"] = data["pressure"] * data["windspeed"]
    data["pressure_humidity"] = data["pressure"] * data["humidity"]
    data["pressure_cloud"] = data["pressure"] * data["cloud"]
    data["windspeed_humidity"] = data["windspeed"] * data["humidity"]
    data["windspeed_cloud"] = data["windspeed"] * data["cloud"]

    data["wind_humidity_ratio"] = data["windspeed"] / (data["humidity"] + eps)
    data["wind_pressure_ratio"] = data["windspeed"] / (data["pressure"] + eps)
    data["moisture_pressure_index"] = data["humidity"] / (data["pressure"] + eps)
    data["cloud_pressure_index"] = data["cloud"] / (data["pressure"] + eps)

    data["dew_humidity_index"] = data["dewpoint"] * data["humidity"]
    data["dew_cloud_index"] = data["dewpoint"] * data["cloud"]

    data["humidity_sq"] = data["humidity"] ** 2
    data["cloud_sq"] = data["cloud"] ** 2
    data["sunshine_sq"] = data["sunshine"] ** 2
    data["dewpoint_gap_sq"] = data["dewpoint_gap"] ** 2

    return data


@st.cache_resource
def load_artifacts():
    config_path = ARTIFACT_DIR / "deployment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "deployment_config.json was not found. Run the notebook's "
            "'Export Streamlit Artifacts' cell first."
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    imputer = joblib.load(ARTIFACT_DIR / "imputer.pkl")
    models = {
        name: joblib.load(ARTIFACT_DIR / f"{name}_model.pkl")
        for name in config["model_names"]
    }

    scalers: Dict[str, object] = {}
    for name in config.get("scaled_models", []):
        scaler_path = ARTIFACT_DIR / f"{name}_scaler.pkl"
        if scaler_path.exists():
            scalers[name] = joblib.load(scaler_path)

    meta_model = None
    meta_path = ARTIFACT_DIR / "meta_model.pkl"
    if meta_path.exists():
        meta_model = joblib.load(meta_path)

    return config, imputer, models, scalers, meta_model


def prepare_input(raw_df: pd.DataFrame, config: dict, imputer) -> np.ndarray:
    missing = [column for column in RAW_FEATURES if column not in raw_df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    clean = raw_df[RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
    engineered = engineer_features(clean)

    feature_columns = config["feature_columns"]
    missing_engineered = [
        column for column in feature_columns if column not in engineered.columns
    ]
    if missing_engineered:
        raise ValueError(
            "Engineered input is missing model features: "
            + ", ".join(missing_engineered)
        )

    ordered = engineered[feature_columns]
    transformed = imputer.transform(ordered)
    return transformed


def predict_probability(
    raw_df: pd.DataFrame,
    config: dict,
    imputer,
    models: dict,
    scalers: dict,
    meta_model,
) -> np.ndarray:
    X = prepare_input(raw_df, config, imputer)

    base_predictions = {}
    for name in config["model_names"]:
        model_input = scalers[name].transform(X) if name in scalers else X
        base_predictions[name] = models[name].predict_proba(model_input)[:, 1]

    final_approach = config["final_approach"]

    if final_approach == "weighted_blend":
        weights = config["blend_weights"]
        probability = sum(
            float(weights[name]) * base_predictions[name]
            for name in config["model_names"]
        )
    elif final_approach == "stacked_ensemble":
        if meta_model is None:
            raise RuntimeError("meta_model.pkl is required for stacked prediction.")
        stacked = np.column_stack(
            [base_predictions[name] for name in config["model_names"]]
        )
        probability = meta_model.predict_proba(stacked)[:, 1]
    elif final_approach == "single_best":
        probability = base_predictions[config["single_best_model"]]
    else:
        raise ValueError(f"Unsupported final approach: {final_approach}")

    return np.clip(probability, 0.0, 1.0)


def validate_weather_values(values: dict) -> list[str]:
    errors = []
    if not 1 <= values["day"] <= 365:
        errors.append("Day must be between 1 and 365.")
    if values["mintemp"] > values["temparature"]:
        errors.append("Minimum temperature cannot exceed mean temperature.")
    if values["temparature"] > values["maxtemp"]:
        errors.append("Mean temperature cannot exceed maximum temperature.")
    if not 0 <= values["humidity"] <= 100:
        errors.append("Humidity must be between 0 and 100.")
    if not 0 <= values["cloud"] <= 100:
        errors.append("Cloud cover must be between 0 and 100.")
    if not 0 <= values["winddirection"] <= 360:
        errors.append("Wind direction must be between 0 and 360 degrees.")
    if values["windspeed"] < 0:
        errors.append("Wind speed cannot be negative.")
    if values["sunshine"] < 0:
        errors.append("Sunshine duration cannot be negative.")
    return errors


st.set_page_config(
    page_title="Rainfall Probability Prediction",
    page_icon="🌧️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(14, 165, 233, 0.12), transparent 34%),
            linear-gradient(180deg, #f5f9ff 0%, #f8fafc 42%, #ffffff 100%);
    }
    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.24);
        padding: 1rem 1.15rem;
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.045);
    }
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 1.25rem;
        border-radius: 18px;
    }
    .hero {
        padding: 1.8rem 2rem;
        border-radius: 22px;
        color: #f8fafc;
        background: linear-gradient(120deg, #0f172a 0%, #123b5d 58%, #0369a1 100%);
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.16);
        margin-bottom: 1.5rem;
    }
    .hero-kicker {
        color: #7dd3fc;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .hero h1 {
        color: #ffffff;
        margin: 0.3rem 0;
        font-size: clamp(1.8rem, 4vw, 2.7rem);
    }
    .hero p {
        color: #dbeafe;
        margin: 0;
        max-width: 760px;
    }
    </style>
    <section class="hero">
        <div class="hero-kicker">Weather intelligence</div>
        <h1>Rainfall probability</h1>
        <p>Explore a model-based rainfall estimate from weather measurements,
        or score a complete CSV of records.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    config, imputer, models, scalers, meta_model = load_artifacts()
except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
    st.error(str(exc))
    st.info(
        "Place the exported model artifacts inside the `artifacts` folder "
        "next to app.py."
    )
    st.stop()

with st.sidebar:
    st.header("Model settings")
    st.caption("These settings affect the displayed class, not the probability.")
    st.write("Ensemble", config["final_approach"].replace("_", " ").title())
    st.write("Base models", ", ".join(config["model_names"]))
    if "validation_auc" in config:
        st.metric("Validation ROC-AUC", f'{config["validation_auc"]:.5f}')
    threshold = st.slider(
        "Rainfall classification threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "The probability is unchanged. A record is classified as rainfall "
            "when its probability is at or above this threshold."
        ),
    )

manual_tab, batch_tab, about_tab = st.tabs(
    ["Single record", "Batch CSV", "How it works"]
)

with manual_tab:
    st.subheader("Weather conditions")
    st.caption(
        "Enter the measurements for one record. All fields are required; "
        "the day value uses the day-of-year scale (1-365)."
    )

    with st.form("rainfall_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 🌡️ Temperature & pressure")
            day = st.number_input(
                "Day of year", 1, 365, 180, 1, help="Calendar day, from 1 to 365."
            )
            pressure = st.number_input("Pressure", value=1013.0, step=0.1, format="%.1f")
            maxtemp = st.number_input(
                "Maximum temperature", value=28.0, step=0.1, format="%.1f"
            )
            temparature = st.number_input(
                "Mean temperature", value=24.0, step=0.1, format="%.1f"
            )

        with col2:
            st.markdown("##### 💧 Moisture & cloud")
            mintemp = st.number_input(
                "Minimum temperature", value=20.0, step=0.1, format="%.1f"
            )
            dewpoint = st.number_input(
                "Dew point", value=19.0, step=0.1, format="%.1f"
            )
            humidity = st.number_input(
                "Humidity (%)", 0.0, 100.0, 75.0, 1.0
            )
            cloud = st.number_input(
                "Cloud cover (%)", 0.0, 100.0, 65.0, 1.0
            )

        with col3:
            st.markdown("##### ☀️ Sun & wind")
            sunshine = st.number_input(
                "Sunshine duration",
                value=5.0,
                min_value=0.0,
                step=0.1,
                help="Use the same unit as the model's training data.",
            )
            winddirection = st.number_input(
                "Wind direction (degrees)", 0.0, 360.0, 180.0, 1.0
            )
            windspeed = st.number_input(
                "Wind speed",
                value=15.0,
                min_value=0.0,
                step=0.1,
                help="Use the same unit as the model's training data.",
            )

        submitted = st.form_submit_button(
            "Calculate rainfall probability", type="primary", use_container_width=True
        )

    if submitted:
        values = {
            "day": day,
            "pressure": pressure,
            "maxtemp": maxtemp,
            "temparature": temparature,
            "mintemp": mintemp,
            "dewpoint": dewpoint,
            "humidity": humidity,
            "cloud": cloud,
            "sunshine": sunshine,
            "winddirection": winddirection,
            "windspeed": windspeed,
        }

        validation_errors = validate_weather_values(values)
        if validation_errors:
            for message in validation_errors:
                st.warning(message)
        else:
            input_df = pd.DataFrame([values])
            try:
                probability = float(
                    predict_probability(
                        input_df, config, imputer, models, scalers, meta_model
                    )[0]
                )
            except (KeyError, RuntimeError, ValueError) as exc:
                st.error(f"Could not calculate a prediction: {exc}")
            else:
                predicted_class = int(probability >= threshold)
                result_col1, result_col2 = st.columns([1, 2])
                with result_col1:
                    st.metric("Rainfall probability", f"{probability:.1%}")
                with result_col2:
                    st.metric(
                        "Classification",
                        "Rainfall" if predicted_class else "No rainfall",
                        help=f"Classification threshold: {threshold:.0%}",
                    )
                st.progress(
                    probability,
                    text=f"Probability: {probability:.1%} | Threshold: {threshold:.0%}",
                )
                st.caption(
                    "This is a model estimate, not an official forecast or weather warning."
                )
                with st.expander("Review submitted measurements"):
                    st.dataframe(input_df, use_container_width=True, hide_index=True)

with batch_tab:
    st.subheader("Score a CSV file")
    st.write(
        "Upload one row per weather record. Predictions are generated for each row; "
        "an optional `id` column is retained in the output."
    )
    with st.expander("Required CSV columns"):
        st.code(", ".join(RAW_FEATURES), language="text")
        st.caption("Column names must match exactly. Additional columns are ignored.")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="batch_csv",
        help="The CSV must contain all required weather columns.",
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not read this CSV file: {exc}")
        else:
            missing_columns = [column for column in RAW_FEATURES if column not in batch_df]
            if missing_columns:
                st.error("Missing required columns: " + ", ".join(missing_columns))
            elif batch_df.empty:
                st.warning("This CSV has a header but no data rows.")
            else:
                st.caption(
                    f"File: {uploaded_file.name} · "
                    f"{len(batch_df):,} rows · {len(batch_df.columns):,} columns"
                )
                with st.expander("Preview input data", expanded=False):
                    st.dataframe(batch_df.head(10), use_container_width=True)

                try:
                    probabilities = predict_probability(
                        batch_df, config, imputer, models, scalers, meta_model
                    )
                except (KeyError, RuntimeError, ValueError) as exc:
                    st.error(f"Prediction failed: {exc}")
                else:
                    result = pd.DataFrame()
                    if "id" in batch_df.columns:
                        result["id"] = batch_df["id"].to_numpy()
                    result["rainfall_probability"] = probabilities
                    result["rainfall_prediction"] = (
                        probabilities >= threshold
                    ).astype(int)

                    rainfall_count = int(result["rainfall_prediction"].sum())
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    metric_col1.metric("Records scored", f"{len(result):,}")
                    metric_col2.metric("Rainfall classified", f"{rainfall_count:,}")
                    metric_col3.metric(
                        "No rainfall classified", f"{len(result) - rainfall_count:,}"
                    )
                    st.caption(
                        f"Classification threshold: {threshold:.0%}. "
                        "Downloaded probability values range from 0 to 1."
                    )
                    display_result = result.copy()
                    display_result["rainfall_probability"] = display_result[
                        "rainfall_probability"
                    ].map(lambda value: f"{value:.1%}")
                    st.dataframe(
                        display_result.head(20),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "Download predictions as CSV",
                        data=result.to_csv(index=False).encode("utf-8"),
                        file_name="rainfall_predictions.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True,
                    )

with about_tab:
    st.subheader("How the estimate is produced")
    st.markdown(
        """
        1. Enter a weather record or upload a CSV with the required weather fields.
        2. The app checks required columns and validates single-record measurements.
        3. It creates the engineered features used by the trained model.
        4. It applies the saved imputer and model-specific scaling, when required.
        5. The configured ensemble produces a rainfall probability.
        6. The selected threshold determines the displayed classification.
        """
    )
    st.subheader("Model and responsible use")
    st.write(
        f"The current ensemble is **{config['final_approach'].replace('_', ' ').title()}** "
        f"with **{len(config['model_names'])} base model(s)**. "
        "The probability is a model estimate and is not an official forecast or "
        "weather-warning service. Do not use it for safety-critical decisions."
    )
