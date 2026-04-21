import pandas as pd

# Path to dataset
path = "UCI HAR Dataset/"

# Load training files
X_train = pd.read_csv(path + "train/X_train.txt", sep="\s+", header=None)
y_train = pd.read_csv(path + "train/y_train.txt", sep="\s+", header=None)

# Load testing files
X_test = pd.read_csv(path + "test/X_test.txt", sep="\s+", header=None)
y_test = pd.read_csv(path + "test/y_test.txt", sep="\s+", header=None)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
