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

st.title("🌧️ Rainfall Probability Prediction")
st.caption(
    "A Streamlit implementation of the trained rainfall ensemble. "
    "The application applies the same feature engineering, imputation, "
    "model preprocessing and final ensemble logic used in the notebook."
)

try:
    config, imputer, models, scalers, meta_model = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.info(
        "Place the exported model artifacts inside the `artifacts` folder "
        "next to app.py."
    )
    st.stop()

with st.sidebar:
    st.header("Model Information")
    st.write("Final approach:", config["final_approach"].replace("_", " ").title())
    st.write("Base models:", ", ".join(config["model_names"]))
    if "validation_auc" in config:
        st.metric("Validation ROC-AUC", f'{config["validation_auc"]:.5f}')
    threshold = st.slider(
        "Classification threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="The probability itself is unchanged; this only controls the displayed class.",
    )

manual_tab, batch_tab, about_tab = st.tabs(
    ["Single Prediction", "Batch CSV Prediction", "About"]
)

with manual_tab:
    st.subheader("Enter Weather Measurements")

    with st.form("rainfall_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            day = st.number_input("Day of year", 1, 365, 180, 1)
            pressure = st.number_input(
                "Pressure", value=1013.0, step=0.1, format="%.1f"
            )
            maxtemp = st.number_input(
                "Maximum temperature", value=28.0, step=0.1, format="%.1f"
            )
            temparature = st.number_input(
                "Mean temperature", value=24.0, step=0.1, format="%.1f"
            )

        with col2:
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
            sunshine = st.number_input(
                "Sunshine duration", value=5.0, min_value=0.0, step=0.1
            )
            winddirection = st.number_input(
                "Wind direction (degrees)", 0.0, 360.0, 180.0, 1.0
            )
            windspeed = st.number_input(
                "Wind speed", value=15.0, min_value=0.0, step=0.1
            )

        submitted = st.form_submit_button(
            "Predict Rainfall Probability", use_container_width=True
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
            probability = float(
                predict_probability(
                    input_df, config, imputer, models, scalers, meta_model
                )[0]
            )
            predicted_class = int(probability >= threshold)

            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Rainfall probability", f"{probability:.2%}")
            metric_col2.metric(
                "Prediction",
                "Rainfall expected" if predicted_class == 1 else "No rainfall expected",
            )

            st.progress(probability)
            st.dataframe(input_df, use_container_width=True, hide_index=True)

with batch_tab:
    st.subheader("Predict Multiple Records")
    st.write(
        "Upload a CSV containing the eleven weather columns shown below. "
        "An optional `id` column is preserved in the downloaded results."
    )
    st.code(", ".join(RAW_FEATURES), language="text")

    uploaded_file = st.file_uploader(
        "Upload input CSV", type=["csv"], key="batch_csv"
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview")
            st.dataframe(batch_df.head(20), use_container_width=True)

            probabilities = predict_probability(
                batch_df, config, imputer, models, scalers, meta_model
            )

            result = pd.DataFrame()
            if "id" in batch_df.columns:
                result["id"] = batch_df["id"]
            result["rainfall_probability"] = probabilities
            result["rainfall_prediction"] = (
                probabilities >= threshold
            ).astype(int)

            st.success(f"Generated predictions for {len(result):,} rows.")
            st.dataframe(result.head(20), use_container_width=True)

            st.download_button(
                "Download Predictions",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="rainfall_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with about_tab:
    st.subheader("Application Workflow")
    st.markdown(
        """
        1. The user enters weather measurements or uploads a CSV.
        2. The application validates the required fields.
        3. The same engineered features used during training are created.
        4. Features are placed in the saved training order.
        5. The saved median imputer and any model-specific scaler are applied.
        6. All saved base models generate rainfall probabilities.
        7. The selected ensemble generates the final probability.
        """
    )
    st.subheader("Important Note")
    st.write(
        "This application demonstrates the trained machine-learning workflow. "
        "It is not an official weather-warning service and should not be used "
        "for safety-critical decisions."
    )
