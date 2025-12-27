# evaluate.py
import yaml
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from src.data_loader import DataLoader
from src.utils import ModelEvaluator

def main():
    # 1. Load Config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print("[INFO] Loading system artifacts...")
    
    # 2. Load Model & Scaler
    model = tf.keras.models.load_model(config['paths']['model_save'])
    # Scaler is needed if we are processing raw data, but DataLoader handles preprocessing.
    # However, DataLoader fits a NEW scaler if we call load_and_preprocess without careful logic.
    # Actually, DataLoader.load_and_preprocess fits/transforms based on training logic.
    # For evaluation, we must strictly use the SAVED scaler.
    
    # Let's verify DataLoader logic. It uses Preprocessor.
    # We should manually handle data loading to ensure we don't re-fit.
    
    # Load raw data
    df = pd.read_csv(config['paths']['raw_data'])
    
    # Drop Time
    if 'Time' in df.columns:
        df = df.drop(['Time'], axis=1)
        
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    
    # Load Scaler
    scaler = joblib.load(config['paths']['scaler_save'])
    X_scaled = scaler.transform(X) # Transform ALL data or just test?
    
    # Let's use the standard split to be consistent with training
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_size'], 
        random_state=config['data']['random_state'],
        stratify=y
    )
    
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Load Custom Threshold
    try:
        with open("models/saved/threshold.txt", "r") as f:
            threshold = float(f.read().strip())
        print(f"[INFO] Using Customer Threshold: {threshold}")
    except Exception as e:
        print(f"[ERROR] Could not read threshold.txt: {e}")
        return

    # 4. Predict
    print("[INFO] Running Inference on Test Set...")
    reconstructions = model.predict(X_test_scaled, verbose=0)
    
    # 5. Evaluate
    evaluator = ModelEvaluator(model, config)
    mse = evaluator.calculate_mse(X_test_scaled, reconstructions)
    
    # Print Report
    evaluator.print_report(y_test, mse, threshold)
    
    # Generate Plots
    print(f"[INFO] Generating diagrams with threshold {threshold}...")
    evaluator.plot_reconstruction_error(mse, y_test, threshold)
    evaluator.plot_roc(y_test, mse)

if __name__ == "__main__":
    main()
