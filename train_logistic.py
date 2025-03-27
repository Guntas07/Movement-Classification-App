import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import h5py
import matplotlib.pyplot as plt
import joblib


# --- Feature extraction helper ---
def extract_features_df(data, axis_label):
    from scipy.stats import skew, kurtosis

    def rms(data):
        return np.sqrt(np.mean(np.square(data), axis=1))

    mean = np.mean(data, axis=1)
    median = np.median(data, axis=1)
    std = np.std(data, axis=1)
    var = np.var(data, axis=1)
    min_val = np.min(data, axis=1)
    max_val = np.max(data, axis=1)
    range_val = max_val - min_val
    skewness = skew(data, axis=1, nan_policy='omit')
    kurt = kurtosis(data, axis=1, nan_policy='omit')
    rms_val = rms(data)

    feature_df = pd.DataFrame({
        f"{axis_label}_mean": mean,
        f"{axis_label}_median": median,
        f"{axis_label}_std": std,
        f"{axis_label}_var": var,
        f"{axis_label}_min": min_val,
        f"{axis_label}_max": max_val,
        f"{axis_label}_range": range_val,
        f"{axis_label}_skewness": skewness,
        f"{axis_label}_kurtosis": kurt,
        f"{axis_label}_rms": rms_val
    })

    return feature_df.dropna()


# --- Load data from HDF5 and prepare train/test sets ---
X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

with h5py.File('hdf5_data.h5', 'r') as hdf:
    for split in ['Train', 'Test']:
        group = hdf[f'Train_Test Data/{split}']
        for name in group:
            dataset = group[name]
            x = dataset['x'][:]
            y = dataset['y'][:]
            z = dataset['z'][:]
            labels = dataset['label'][:]

            x_df = extract_features_df(x, 'x')
            y_df = extract_features_df(y, 'y')
            z_df = extract_features_df(z, 'z')

            combined_df = pd.concat([x_df, y_df, z_df], axis=1).dropna()
            valid_len = len(combined_df)
            if valid_len == 0:
                continue

            labels = labels[:valid_len]

            if split == 'Train':
                X_train_list.append(combined_df)
                y_train_list.append(labels)
            else:
                X_test_list.append(combined_df)
                y_test_list.append(labels)

X_train = pd.concat(X_train_list, ignore_index=True)
y_train = np.concatenate(y_train_list)
X_test = pd.concat(X_test_list, ignore_index=True)
y_test = np.concatenate(y_test_list)

# --- Normalize features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train logistic regression model ---
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# --- Evaluate model ---
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
