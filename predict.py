# predict.py
from turtle import clear
import yaml
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import os

class FraudDetector:
    def __init__(self, config_path="config/config.yaml"):
        # 1. Load Config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # 2. Load Artifacts
        print("[INFO] Loading system artifacts...")
        self.model = tf.keras.models.load_model(self.config['paths']['model_save'])
        self.scaler = joblib.load(self.config['paths']['scaler_save'])
        
        # Load the optimized threshold
        with open("models/saved/threshold.txt", "r") as f:
            self.threshold = float(f.read())
            
        print(f"[INFO] System Ready. Threshold set to: {self.threshold:.4f}")

    def preprocess(self, input_data):
        """
        Scales the input data using the saved scaler.
        """
        # Ensure input is 2D array
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        return self.scaler.transform(input_data)

    def predict(self, sample):
        """
        Returns: Is_Fraud (bool), Anomaly_Score (float), Status (str)
        """
        # 1. Preprocess
        scaled_sample = self.preprocess(sample)
        
        # 2. Reconstruct
        reconstruction = self.model.predict(scaled_sample, verbose=0)
        
        # 3. Calculate Error (MSE)
        mse = np.mean(np.power(scaled_sample - reconstruction, 2), axis=1)[0]
        
        # 4. Compare with Threshold
        is_fraud = mse > self.threshold
        
        return is_fraud, mse

def simulate_live_data():
    """
    Simulates live transactions by picking random samples from the dataset.
    """
    # Load dataset just to pick samples
    df = pd.read_csv("data/raw/creditcard.csv")
    
    # Separate Fraud and Normal for demonstration purposes
    fraud_samples = df[df['Class'] == 1].drop(['Time', 'Class'], axis=1).values
    normal_samples = df[df['Class'] == 0].drop(['Time', 'Class'], axis=1).values
    
    detector = FraudDetector()
    
    print("\n" + "="*50)
    print("      🔍 STARTING LIVE TRANSACTION DEMO")
    print("="*50)
    
    # Simulate 5 Transactions
    for i in range(5):
        # Randomly pick fraud or normal (50% chance for demo excitement)
        if np.random.rand() > 0.5:
            idx = np.random.randint(0, len(fraud_samples))
            sample = fraud_samples[idx]
            ground_truth = "ACTUAL FRAUD"
        else:
            idx = np.random.randint(0, len(normal_samples))
            sample = normal_samples[idx]
            ground_truth = "ACTUAL NORMAL"
            
        print(f"\n--- Transaction #{i+1} ({ground_truth}) ---")
        
        # Run Prediction
        is_fraud, score = detector.predict(sample)
        
        # Output Result
        status = "🚨 ANOMALY DETECTED" if is_fraud else "✅ Transaction Safe"
        color = "\033[91m" if is_fraud else "\033[92m" # Red or Green
        reset = "\033[0m"
        
        print(f"Reconstruction Error: {score:.4f}")
        print(f"Threshold Limit:      {detector.threshold:.4f}")
        print(f"Result: {color}{status}{reset}")
        
        if (is_fraud and "FRAUD" in ground_truth) or (not is_fraud and "NORMAL" in ground_truth):
             print("System Accuracy: Correct")
        else:
             print("System Accuracy: Missed/False Alarm")

if __name__ == "__main__":
    simulate_live_data()