
# NeuroMove HAR: Human Activity Recognition System

NeuroMove is an end-to-end Human Activity Recognition (HAR) framework that integrates Deep Learning (LSTM) with modern MLOps and DevOps engineering standards. The system classifies human motion data from smartphone sensors with high precision and serves predictions through a cloud-hosted interactive dashboard.

## 🚀 System Overview
The core of NeuroMove is a Long Short-Term Memory (LSTM) neural network trained on the UCI HAR dataset. It achieves a **94.03% test accuracy** across six activity classes: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, and Laying.

## 🛠️ Technical Stack
* **Deep Learning:** TensorFlow & Keras (LSTM Architecture)
* **Data Science:** Scikit-Learn (StandardScaler), Pandas, NumPy
* **Visualization:** Streamlit, Plotly (3D Biomechanical Skeleton & Signal Waveforms)
* **DevOps:** Docker (Containerization), GitHub Actions (CI/CD), Render (Cloud Hosting)
* **MLOps:** JSON Experiment Tracking, Serialized Scaler State (.joblib)

## 📁 Project Structure
```text
├── .github/workflows/       # GitHub Actions CI/CD configuration
├── models/                  # Serialized model (.h5) and scaler (.joblib)
├── src/                     # Core logic for preprocessing and inference
├── app.py                   # Streamlit dashboard application
├── Dockerfile               # Production container build recipe
├── requirements.txt         # Project dependencies
└── experiment_tracking.json # Automated MLOps audit ledger
```

## ⚙️ Installation & Deployment

### 🐳 Run with Docker (Recommended)
1. **Build the image:**
   ```bash
   docker build -t neuromove-har .
   ```
2. **Launch the service:**
   ```bash
   docker run -p 8501:8501 neuromove-har
   ```
3. **Access:** Open `http://localhost:8501` in your browser.

### 🐍 Local Development Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Streamlit:**
   ```bash
   streamlit run app.py
   ```

## 🔄 MLOps & DevOps Workflow
* **Continuous Integration:** GitHub Actions automatically lints Python code and verifies the Docker build on every push to the `main` branch.
* **State Preservation:** The `scaler.joblib` artifact ensures that the exact normalization parameters used in training are applied in production, preventing training-serving skew.
* **Auditability:** Every training run is logged to `experiment_tracking.json`, capturing hyperparameters, loss, and accuracy for project reproducibility.
* **Continuous Deployment:** Successful builds are automatically deployed to the **Render Cloud Platform**.

## 📊 Performance Summary
* **Test Accuracy:** 94.03%
* **Test Loss:** 0.1848
* **Model Format:** HDF5 (.h5)
* **Architecture:** Multi-layer LSTM with Dropout Regularization

## 📜 Acknowledgments
* **Dataset:** UCI Machine Learning Repository (Human Activity Recognition Using Smartphones).
* **Deployment:** Infrastructure powered by Render.
