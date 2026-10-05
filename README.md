# Rainfall Probability Prediction — Streamlit Application

## Folder structure

```text
Rainfall_Streamlit_App/
├── app.py
├── requirements.txt
├── export_artifacts_cell.py
└── artifacts/
```

## 1. Copy the folder

Extract or copy `Rainfall_Streamlit_App` to:

```text
D:\CI\Rainfall_Streamlit_App
```

## 2. Export the trained model artifacts

Open `Rainfall_prediction.ipynb`.

After all model-training cells have completed, run the code contained in:

```text
export_artifacts_cell.py
```

This creates the required model files inside:

```text
D:\CI\Rainfall_Streamlit_App\artifacts
```

The application cannot make predictions until these trained artifacts exist.

## 3. Create and activate a virtual environment

Open Command Prompt or the VS Code terminal:

```bat
cd /d D:\CI\Rainfall_Streamlit_App
python -m venv .venv
.venv\Scripts\activate
```

## 4. Install dependencies

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Run the application

```bat
streamlit run app.py
```

Streamlit normally opens:

```text
http://localhost:8501
```

## Application features

- Single-record rainfall probability prediction
- Input validation
- Batch CSV prediction
- Downloadable prediction results
- Saved-model and ensemble loading
- The same notebook feature engineering and preprocessing
- Support for weighted blend, stacking, or the best single model

## Required batch CSV columns

```text
day, pressure, maxtemp, temparature, mintemp, dewpoint,
humidity, cloud, sunshine, winddirection, windspeed
```

An optional `id` column is preserved in the output.
