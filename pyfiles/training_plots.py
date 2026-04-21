import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load training history from model
history = pd.read_csv("history.csv")  # we will generate this in next step if missing

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_plots.png", dpi=200)
print("Saved training_plots.png")
