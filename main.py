# main.py
import yaml
import numpy as np
import tensorflow as tf
from src.data_loader import DataLoader
from src.autoencoder import AutoencoderBuilder
from src.trainer import AutoencoderTrainer
from src.utils import ModelEvaluator

def main():
    # 1. Load Configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Load & Preprocess Data
    loader = DataLoader(config['paths']['raw_data'], config)
    X_train, X_test, y_test = loader.load_and_preprocess()

    # 3. Build & Train Model
    input_dim = X_train.shape[1]
    model = AutoencoderBuilder.build(input_dim, config)
    
    # Check if model exists to skip training (Optional, mostly for dev)
    # For now, we train every time as requested.
    trainer = AutoencoderTrainer(model, config)
    trainer.train(X_train, X_test)

    # 4. Predict & Calculate Error
    print("[INFO] Performing inference on Test Set...")
    reconstructions = model.predict(X_test)
    
    evaluator = ModelEvaluator(model, config)
    mse = evaluator.calculate_mse(X_test, reconstructions)
    
    # 5. Determine Threshold & Evaluate
    # We use y_test here only to find the best cut-off for the report
    best_threshold = evaluator.find_best_threshold(mse, y_test)
    
    # 6. Generate Plots and Reports
    evaluator.plot_reconstruction_error(mse, y_test, best_threshold)
    evaluator.plot_roc(y_test, mse)
    evaluator.print_report(y_test, mse, best_threshold)

    # Save the threshold to config or a separate file for the demo script
    print(f"[INFO] Saving threshold {best_threshold} for production...")
    with open("models/saved/threshold.txt", "w") as f:
        f.write(str(best_threshold))

if __name__ == "__main__":
    main()