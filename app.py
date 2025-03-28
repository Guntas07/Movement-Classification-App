import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
from scipy.stats import skew, kurtosis
from realtime import read_realtime_data, close_driver

# Load model and scaler that were created in the train_logistic file
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Realtime prediction history
realtime_preds = []


# Feature extraction from 5 second windows
def extract_features_df(data, axis_label):
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
        f"{axis_label}_rms": np.sqrt(np.mean(np.square(data), axis=1))
    }).dropna()


# CSV Prediction from user input
def process_and_predict(filepath):
    df = pd.read_csv(filepath)

    # Fix any missing values using second order polynomial interpolation
    df.interpolate(method="polynomial", order=2, inplace=True)

    # Keep the MA length to 49 as before and extract the MA's
    window = 49
    xMA = df['Acceleration x (m/s^2)'].rolling(window=window).mean()
    yMA = df['Acceleration y (m/s^2)'].rolling(window=window).mean()
    zMA = df['Acceleration z (m/s^2)'].rolling(window=window).mean()

    # Get time data for calculations
    time_series = df["Time (s)"]
    avg_delta = time_series.diff().mean()
    num_entries = int(5 / avg_delta)
    num_windows = len(df) // num_entries

    # Segment the signals into 5 second windows using the variables calculated above
    x_segments, y_segments, z_segments = [], [], []
    for i in range(num_windows):
        start, end = i * num_entries, (i + 1) * num_entries
        x_segments.append(xMA.iloc[start:end].values)
        y_segments.append(yMA.iloc[start:end].values)
        z_segments.append(zMA.iloc[start:end].values)

    # Extract the 10 features from each axis
    x_df = extract_features_df(np.array(x_segments), 'x')
    y_df = extract_features_df(np.array(y_segments), 'y')
    z_df = extract_features_df(np.array(z_segments), 'z')

    # Combine them into one dataframe
    features = pd.concat([x_df, y_df, z_df], axis=1)

    # Use the pre-fitted scaler to standardize the features
    features_scaled = scaler.transform(features)
    # Predict the activity for each 5-second window and return it (array)
    preds = model.predict(features_scaled)
    return preds


# --- Realtime Prediction Loop ---
def update_realtime():
    # Try in case of an error like phone disconnecting or website glitching
    try:
        # Calls read data from the other file to get the last 5 seconds of data
        x_vals, y_vals, z_vals = read_realtime_data()
        # Extract the 10 statistical features
        x_df = extract_features_df(np.array([x_vals]), 'x')
        y_df = extract_features_df(np.array([y_vals]), 'y')
        z_df = extract_features_df(np.array([z_vals]), 'z')
        # Combine all feature dataframes into one to scale and predict
        features = pd.concat([x_df, y_df, z_df], axis=1)
        features_scaled = scaler.transform(features)
        # Model returns an array, but you only need the first (and only) part of it
        pred = model.predict(features_scaled)[0]
        # Stores predictoin to list of past predictions
        realtime_preds.append(pred)
        # Updates the gui plot
        show_realtime_plot(realtime_preds)
        # Update the label at the top of the screen to say walking or running
        update_label(pred)
    except Exception as e:
        print("Realtime error:", e)
    finally:
        # After 5 seconds, call the function again (this recursively runs indefinitely)
        root.after(5000, update_realtime)


# UI Functions
# Let the user upload a CSV
def choose_file():
    path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if path:
        label_path.config(text=f"Selected: {path.split('/')[-1]}")
        process_file(path)


def process_file(path):
    try:
        # Calls the process and predict pipeline
        preds = process_and_predict(path)
        # Creates a dataframe for walking or jumping
        output_df = pd.DataFrame({
            'Window': range(len(preds)),
            'Label': ['Walking' if p == 0 else 'Jumping' for p in preds]
        })
        # Outputs that dataframe into a csv file with the original title + _output
        out_path = path.replace(".csv", "_output.csv")
        output_df.to_csv(out_path, index=False)
        # Show user where it is saved
        messagebox.showinfo("Done", f"Output saved to:\n{out_path}")
        # Show the plot
        show_static_plot(preds)
    except Exception as e:
        messagebox.showerror("Error", str(e))


def show_static_plot(preds):
    # Clears the previous plot from the gui
    for widget in frame_static_plot.winfo_children():
        widget.destroy()

    # Plots the figure using matplotlib
    fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
    ax.plot(preds, drawstyle='steps-post')
    ax.set_title("Predicted Activity Over Time")
    ax.set_ylabel("Activity")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Walking", "Jumping"])
    ax.set_xlabel("5-sec Window #")
    ax.set_xticks(range(len(preds)))
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout(pad=2)
    canvas = FigureCanvasTkAgg(fig, master=frame_static_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def show_realtime_plot(preds):
    # Clears the previous plot from the gui
    for widget in frame_realtime_plot.winfo_children():
        widget.destroy()

    # Plots the figure
    fig, ax = plt.subplots(figsize=(8, 3), dpi=100)

    ax.plot(preds, drawstyle='steps-post', label="Prediction")
    ax.set_ylabel("Activity")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Walking", "Jumping"])
    ax.set_xlabel("5-sec Window #")
    ax.set_xticks(range(len(preds)))
    ax.set_title("Realtime Activity Classification")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    fig.tight_layout(pad=2)
    canvas = FigureCanvasTkAgg(fig, master=frame_realtime_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def update_label(pred):
    label_text = "Jumping" if pred == 1 else "Walking"
    current_prediction_label.config(text=f"Current Prediction: {label_text}")


# GUI Setup
root = tk.Tk()
root.title("Activity Classifier")
root.geometry("840x600")

# Tab Control
tab_control = ttk.Notebook(root)
tab_static = ttk.Frame(tab_control)
tab_realtime = ttk.Frame(tab_control)
tab_control.add(tab_static, text='CSV Classification')
tab_control.add(tab_realtime, text='Realtime Classification')
tab_control.pack(expand=1, fill='both')

# Static tab content
title_static = tk.Label(tab_static, text="Jump vs Walk Classifier", font=("Helvetica", 18, "bold"))
title_static.pack(pady=20)

btn_file = tk.Button(tab_static, text="Choose CSV File", command=choose_file, font=("Helvetica", 12), padx=10, pady=5)
btn_file.pack(pady=5)

label_path = tk.Label(tab_static, text="No file selected", font=("Helvetica", 10))
label_path.pack(pady=5)

frame_static_plot = tk.Frame(tab_static, padx=10, pady=10, bd=1, relief="solid")
frame_static_plot.pack(pady=20, fill=tk.BOTH, expand=True)

# Realtime tab content
title_realtime = tk.Label(tab_realtime, text="Live Classification (5s Window)", font=("Helvetica", 18, "bold"))
title_realtime.pack(pady=10)

current_prediction_label = tk.Label(tab_realtime, text="Current Prediction: N/A", font=("Helvetica", 14))
current_prediction_label.pack(pady=5)

frame_realtime_plot = tk.Frame(tab_realtime, padx=10, pady=10, bd=1, relief="solid")
frame_realtime_plot.pack(pady=20, fill=tk.BOTH, expand=True)

# Start realtime loop
root.after(1000, update_realtime)

# On close cleanup
root.protocol("WM_DELETE_WINDOW", lambda: (close_driver(), root.destroy()))

root.mainloop()
