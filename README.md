# 🛡️ Credit Card Anomaly Detection

> **Unsupervised Deep Learning System for Fraud Detection**  
> *Built with TensorFlow, Keras, and Python.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?style=for-the-badge&logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Active-green?style=for-the-badge)

## 📖 Overview

This project implements an **Autoencoder Neural Network** to detect anomalies (potential fraud) in credit card transactions. Unlike traditional supervised classification, this system learns the representation of *normal* transactions and flags any transaction with a high **Reconstruction Error** (MSE) as an anomaly.

### Key Features
- **Autoencoder Architecture**: Compresses and reconstructs input data to learn latent patterns.
- **Robust Preprocessing**: Uses **RobustScaler** to handle outliers effectively.
- **Advanced Regularization**: Incorporates **Batch Normalization** and **Dropout** for training stability.
- **Recall-Oriented Thresholding**: Automatically calculates a threshold to target **90% Recall**, ensuring high fraud detection rates.
- **Live Simulation**: Includes a demo script (`predict.py`) to simulate real-time fraud detection.
- **Modular Design**: Clean separation of data loading, preprocessing, training, and evaluation logic.

---

## 📂 Project Structure

```text
credit-card-anomaly-detection/
├── config/
│   └── config.yaml          # ⚙️ Hyperparameters and file paths
├── data/
│   └── raw/                 # 💾 Raw dataset (creditcard.csv)
├── models/
│   └── saved/               # 🧠 Trained models, scalers, and stats
│       ├── autoencoder_model.keras
│       ├── scaler.pkl
│       └── threshold.txt
├── src/                     # 📦 Source Code
│   ├── modules/
│   │   ├── autoencoder.py   # Robust Autoencoder (Dense + BatchNorm + Dropout)
│   │   ├── data_loader.py   # Data ingestion pipeline
│   │   ├── preprocessor.py  # RobustScaler integration
│   │   ├── trainer.py       # Training loop with Callbacks
│   │   └── utils.py         # Metrics & Plotting
│   └── __init__.py
├── main.py                  # 🚀 Entry point for training
├── predict.py               # 🔮 Entry point for inference/demo
├── test_data.py             # 🧪 Data integrity check script
├── report.tex               # 📄 Latex Project Report
├── requirements.txt         # 📋 Dependencies list
└── README.md                # 📄 Project documentation
```

---

## ⚡ Installation

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Requires Python 3.10+ and pip.*

---

## 🚀 Usage Guide

### 1. Verification (Optional)
Ensure the dataset exists and the pipeline is ready:
```bash
python test_data.py
```

### 2. Model Training
Train the Autoencoder on normal transactions. This step handles:
- Data splitting (Train/Test)
- Normalization (RobustScaler)
- Model training with Early Stopping
- Automatic thresholding (Targeting 90% Recall)

```bash
python main.py
```
*Output will be saved to `models/saved/`.*

### 3. Inference & Demo
Run the prediction simulation to test the model on random samples from the dataset. It mimics a live production environment.

```bash
python predict.py
```

**Sample Output:**
```text
🔍 STARTING LIVE TRANSACTION DEMO
==================================================

--- Transaction #1 (ACTUAL NORMAL) ---
Reconstruction Error: 0.3421
Threshold Limit:      3.0000
Result: ✅ Transaction Safe
System Accuracy: Correct

--- Transaction #2 (ACTUAL FRAUD) ---
Reconstruction Error: 15.8912
Threshold Limit:      3.0000
Result: 🚨 ANOMALY DETECTED
System Accuracy: Correct
```

---

## ⚙️ Configuration

Control the entire pipeline via `config/config.yaml`. No need to edit code to change parameters!

```yaml
paths:
  raw_data: "data/raw/creditcard.csv"
  model_save: "models/saved/autoencoder_model.keras"

model:
  input_dim: 29       # Features (V1-V28 + Amount)
  encoding_dim: 14    # Latent space size
  hidden_dim: 7       # Bottleneck size
  learning_rate: 0.001

train:
  epochs: 50
  batch_size: 32
  patience: 10        # Early stopping patience
```

---

## 📊 Results & Artifacts

After training, check `outputs/plots/` (created automatically) for:
- 📉 **Reconstruction Error Distribution**: Histogram distinguishing Normal vs Fraud errors.
- 📈 **ROC Curve**: True Positive Rate vs False Positive Rate.
- 🟦 **Confusion Matrix**: Visual breakdown of classifications.

---

## 🤝 Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
