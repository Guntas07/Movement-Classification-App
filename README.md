# Movement Classification App

This project classifies user movement (Walking vs Jumping) using a Logistic Regression model trained on team-collected sensor data. It supports both a desktop UI (Tkinter) and a redesigned modern web UI (FastAPI + custom CSS + Chart.js).

Model accuracy on >300k labeled samples: ~98%.

## Apps
- Desktop UI: `app.py` (Tkinter, plots with Matplotlib)
- Web UI: `server.py` with:
  - CSV Studio at `/` for drag-and-drop CSV classification
  - Realtime Lab at `/realtime` for polling the PhyPhox stream

## Frontend refresh
The web experience now includes:
- A dark glassmorphism design system in `static/styles.css`
- A polished CSV upload workflow in `static/app.js`
- A realtime polling dashboard in `static/realtime.js`
- Chart.js visualizations for both uploaded and live predictions
- Responsive layouts for desktop and mobile

## Quickstart (Web UI)
1) Python deps (suggested):
   - `pip install fastapi uvicorn[standard] jinja2 python-multipart pandas numpy scipy scikit-learn joblib selenium`
2) Run the server:
   - `uvicorn server:app --reload`
3) Open the UI:
   - Visit `http://127.0.0.1:8000` for CSV upload/classification
   - Visit `http://127.0.0.1:8000/realtime` for live polling (uses your PhyPhox stream)

The backend reuses the existing model (`model.pkl`) and scaler (`scaler.pkl`). CSVs should include columns:
`Acceleration x (m/s^2)`, `Acceleration y (m/s^2)`, `Acceleration z (m/s^2)`, `Time (s)`.

## Realtime Notes
Realtime classification uses the existing `realtime.py` Selenium reader pointing at a local PhyPhox web view. Update the address in `realtime.py` to match your network setup.

## Desktop UI
If you prefer the original desktop app: `python app.py`
