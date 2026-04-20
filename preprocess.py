import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

path = "UCI HAR Dataset/"

# Load the data
X_train = pd.read_csv(path + "train/X_train.txt", sep="\s+", header=None)
y_train = pd.read_csv(path + "train/y_train.txt", sep="\s+", header=None)

X_test = pd.read_csv(path + "test/X_test.txt", sep="\s+", header=None)
y_test = pd.read_csv(path + "test/y_test.txt", sep="\s+", header=None)

# Convert labels from 1–6 → 0–5
y_train = y_train[0].values - 1
y_test = y_test[0].values - 1

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for Streamlit later
joblib.dump(scaler, "scaler.joblib")

# Reshape for LSTM: (samples, timesteps=1, features=561)
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_lstm  = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Save preprocessed arrays
np.save("X_train_lstm.npy", X_train_lstm)
np.save("X_test_lstm.npy", X_test_lstm)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Preprocessing complete!")
print("X_train_lstm:", X_train_lstm.shape)
print("X_test_lstm:", X_test_lstm.shape)
