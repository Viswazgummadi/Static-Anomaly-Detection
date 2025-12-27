# optimize_threshold.py
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix
from src.data_loader import DataLoader
from src.autoencoder import AutoencoderBuilder

def calculate_cost(y_true, y_pred, cost_fn, cost_fp):
    cm = confusion_matrix(y_true, y_pred)
    # cm structure: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    return (fn * cost_fn) + (fp * cost_fp)

def main():
    # 1. Load Config & Data
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    loader = DataLoader(config['paths']['raw_data'], config)
    # We only need the Test set to find the threshold
    _, X_test, y_test = loader.load_and_preprocess()
    
    # 2. Load Model
    print("[INFO] Loading Model...")
    input_dim = X_test.shape[1]
    model = AutoencoderBuilder.build(input_dim, config)
    model.load_weights(config['paths']['model_save'])
    
    # 3. Get Reconstruction Errors
    print("[INFO] Predicting...")
    reconstructions = model.predict(X_test, verbose=0)
    mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
    
    # 4. Analyze Thresholds
    thresholds = np.linspace(np.min(mse), np.percentile(mse, 99), 100)
    precisions = []
    recalls = []
    f1_scores = []
    costs = []
    
    # BUSINESS ASSUMPTIONS (CHANGE THESE TO MATCH REALITY)
    COST_OF_MISSED_FRAUD = 500.00  # Losing the money
    COST_OF_FALSE_ALARM = 5.00     # Cost of SMS/Support call
    
    print("[INFO] Optimizing...")
    for th in thresholds:
        y_pred = (mse > th).astype(int)
        
        # Calculate Metrics
        p, r, _ = precision_recall_curve(y_test, mse) # This returns arrays, slightly inefficient for loop
        # Simpler way for scalar metrics:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        cost = (fn * COST_OF_MISSED_FRAUD) + (fp * COST_OF_FALSE_ALARM)
        
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        costs.append(cost)

    # 5. Find Winners
    best_f1_idx = np.argmax(f1_scores)
    best_cost_idx = np.argmin(costs)
    
    print("\n" + "="*40)
    print("OPTIMIZATION RESULTS")
    print("="*40)
    print(f"Strategy 1: Best F1 Score")
    print(f"   Threshold: {thresholds[best_f1_idx]:.4f}")
    print(f"   F1: {f1_scores[best_f1_idx]:.4f}")
    print("-" * 40)
    print(f"Strategy 2: Minimum Business Cost")
    print(f"   Assumption: Missed Fraud=${COST_OF_MISSED_FRAUD}, False Alarm=${COST_OF_FALSE_ALARM}")
    print(f"   Threshold: {thresholds[best_cost_idx]:.4f}")
    print(f"   Total Cost: ${costs[best_cost_idx]:,.2f}")
    print("="*40)
    
    # 6. Plotting
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Precision/Recall/F1
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, precisions, label='Precision (Low False Alarms)', linestyle='--')
    plt.plot(thresholds, recalls, label='Recall (Catching Fraud)', color='green')
    plt.plot(thresholds, f1_scores, label='F1 Score', color='red')
    plt.axvline(thresholds[best_f1_idx], color='k', linestyle=':', label='Best F1')
    plt.title("Metric-Based Optimization")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    
    # Subplot 2: Cost Curve
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, costs, color='purple', label='Total Cost ($)')
    plt.axvline(thresholds[best_cost_idx], color='k', linestyle=':', label='Min Cost')
    plt.title("Financial Cost Optimization")
    plt.xlabel("Threshold")
    plt.ylabel("Cost ($)")
    plt.legend()
    
    output_path = "outputs/plots/threshold_optimization.png"
    plt.savefig(output_path)
    print(f"[INFO] Analysis Plot saved to {output_path}")

if __name__ == "__main__":
    main()