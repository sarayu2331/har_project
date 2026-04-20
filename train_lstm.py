import numpy as np
import json
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Load preprocessed data
X_train = np.load("X_train_lstm.npy")
X_test = np.load("X_test_lstm.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Build LSTM model
model = Sequential([
    LSTM(128, input_shape=(1, 561)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(6, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
checkpoint = ModelCheckpoint("best_lstm_model.h5", save_best_only=True, monitor='val_accuracy')
earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[checkpoint, earlystop],
    verbose=1
)

# Save final model
model.save("final_lstm_model.h5")

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print("Test Accuracy:", acc)
print("Test Loss:", loss)

# Experiment Tracking
experiment_log = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_architecture": "LSTM",
    "dataset": "UCI HAR",
    "test_accuracy": float(acc),
    "test_loss": float(loss)
}

with open("experiment_tracking.json", "a") as file:
    json.dump(experiment_log, file)
    file.write("\n")