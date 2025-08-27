from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib
from scipy.stats import skew, kurtosis
from realtime import read_realtime_data
from pathlib import Path
from io import BytesIO

# Load model and scaler once at startup
MODEL_PATH = Path(__file__).parent / "model.pkl"
SCALER_PATH = Path(__file__).parent / "scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    # Defer error to requests so the server can still start
    model = None
    scaler = None


def extract_features_df(data: np.ndarray, axis_label: str) -> pd.DataFrame:
    return pd.DataFrame({
        f"{axis_label}_mean": np.mean(data, axis=1),
        f"{axis_label}_median": np.median(data, axis=1),
        f"{axis_label}_std": np.std(data, axis=1),
        f"{axis_label}_var": np.var(data, axis=1),
        f"{axis_label}_min": np.min(data, axis=1),
        f"{axis_label}_max": np.max(data, axis=1),
        f"{axis_label}_range": np.ptp(data, axis=1),
        f"{axis_label}_skewness": skew(data, axis=1, nan_policy='omit'),
        f"{axis_label}_kurtosis": kurtosis(data, axis=1, nan_policy='omit'),
        f"{axis_label}_rms": np.sqrt(np.mean(np.square(data), axis=1)),
    }).dropna()


def process_csv_bytes(content: bytes) -> list[int]:
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")

    try:
        df = pd.read_csv(BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or unreadable CSV file.")

    required_cols = [
        'Acceleration x (m/s^2)',
        'Acceleration y (m/s^2)',
        'Acceleration z (m/s^2)',
        'Time (s)'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing)}")

    # Interpolate with 2nd order polynomial
    df.interpolate(method="polynomial", order=2, inplace=True)

    window = 49
    xMA = df['Acceleration x (m/s^2)'].rolling(window=window).mean()
    yMA = df['Acceleration y (m/s^2)'].rolling(window=window).mean()
    zMA = df['Acceleration z (m/s^2)'].rolling(window=window).mean()

    time_series = df["Time (s)"]
    avg_delta = time_series.diff().mean()
    if pd.isna(avg_delta) or avg_delta <= 0:
        raise HTTPException(status_code=400, detail="Invalid time series in CSV.")
    num_entries = int(5 / avg_delta)
    if num_entries <= 0:
        raise HTTPException(status_code=400, detail="Not enough data for 5-second windows.")
    num_windows = len(df) // num_entries
    if num_windows == 0:
        raise HTTPException(status_code=400, detail="CSV too short for one window.")

    x_segments, y_segments, z_segments = [], [], []
    for i in range(num_windows):
        start, end = i * num_entries, (i + 1) * num_entries
        x_segments.append(xMA.iloc[start:end].values)
        y_segments.append(yMA.iloc[start:end].values)
        z_segments.append(zMA.iloc[start:end].values)

    x_df = extract_features_df(np.array(x_segments), 'x')
    y_df = extract_features_df(np.array(y_segments), 'y')
    z_df = extract_features_df(np.array(z_segments), 'z')
    features = pd.concat([x_df, y_df, z_df], axis=1)
    features_scaled = scaler.transform(features)
    preds = model.predict(features_scaled)
    return preds.tolist()


app = FastAPI(title="Movement Classifier", version="1.0.0")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/realtime", response_class=HTMLResponse)
async def realtime_page(request: Request):
    return templates.TemplateResponse("realtime.html", {"request": request})


@app.post("/api/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    content = await file.read()
    preds = process_csv_bytes(content)
    labeled = ["Walking" if p == 0 else "Jumping" for p in preds]
    return {"predictions": preds, "labels": labeled}


@app.get("/api/predict/realtime")
async def predict_realtime():
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")
    try:
        x_vals, y_vals, z_vals = read_realtime_data()
        x_df = extract_features_df(np.array([x_vals]), 'x')
        y_df = extract_features_df(np.array([y_vals]), 'y')
        z_df = extract_features_df(np.array([z_vals]), 'z')
        features = pd.concat([x_df, y_df, z_df], axis=1)
        features_scaled = scaler.transform(features)
        pred = int(model.predict(features_scaled)[0])
        label = "Walking" if pred == 0 else "Jumping"
        return {"prediction": pred, "label": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Realtime error: {e}")


@app.get("/api/health")
async def health():
    return {"ok": True, "model_loaded": model is not None, "scaler_loaded": scaler is not None}


# If run directly: uvicorn server:app --reload
