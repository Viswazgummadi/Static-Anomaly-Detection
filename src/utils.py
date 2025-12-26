# src/utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_recall_curve, 
                             roc_curve, auc, classification_report, f1_score)

class ModelEvaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.save_dir = config['paths']['plots']
        import os
        os.makedirs(self.save_dir, exist_ok=True)

    def calculate_mse(self, X_true, X_pred):
        """
        Calculate Mean Squared Error per sample.
        """
        mse = np.mean(np.power(X_true - X_pred, 2), axis=1)
        return mse

    def find_best_threshold(self, mse, y_true):
        """
        Finds a threshold that ensures we catch at least 90% of the fraud (High Recall).
        """
        print("[INFO] Searching for threshold (Target: 90% Recall)...")
        
        # Combine error and labels
        results = pd.DataFrame({'mse': mse, 'y': y_true})
        
        # Sort by Error (Highest first) because High Error = Fraud
        results = results.sort_values(by='mse', ascending=False)
        
        total_fraud = results['y'].sum()
        target_fraud_count = int(total_fraud * 0.90) # We want to catch 90%
        
        current_fraud_count = 0
        best_th = 0
        
        # Iterate down the list
        for index, row in results.iterrows():
            if row['y'] == 1:
                current_fraud_count += 1
            
            # Once we have caught 90% of the fraud, stop and set the threshold here
            if current_fraud_count >= target_fraud_count:
                best_th = row['mse']
                break
                
        print(f"[RESULT] Threshold set to: {best_th:.4f} (Caught {current_fraud_count}/{total_fraud} frauds)")
        return best_th

    def plot_reconstruction_error(self, mse, y_true, threshold):
        """
        Visualizes the error distribution for Normal vs Fraud.
        """
        plt.figure(figsize=(10, 6))
        
        # Plot Normal
        sns.histplot(mse[y_true==0], bins=50, kde=True, color='blue', label='Normal', alpha=0.6)
        # Plot Fraud
        sns.histplot(mse[y_true==1], bins=50, kde=True, color='red', label='Fraud', alpha=0.6)
        
        plt.axvline(threshold, color='k', linestyle='--', label=f'Threshold: {threshold:.2f}')
        plt.title('Reconstruction Error: Normal vs Fraud')
        plt.xlabel('Reconstruction Error (MSE)')
        plt.ylabel('Count')
        plt.yscale('log') # Log scale because normal counts are huge
        plt.legend()
        plt.savefig(f"{self.save_dir}/reconstruction_error.png")
        print(f"[INFO] Error plot saved to {self.save_dir}")

    def plot_roc(self, y_true, mse):
        fpr, tpr, _ = roc_curve(y_true, mse)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(f"{self.save_dir}/roc_curve.png")
        print(f"[INFO] ROC plot saved to {self.save_dir}")

    def print_report(self, y_true, mse, threshold):
        y_pred = (mse > threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        print("\n" + "="*30)
        print("FINAL EVALUATION REPORT")
        print("="*30)
        print(f"Confusion Matrix:\n{cm}")
        print("-" * 30)
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
        
        # Plot Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        plt.title("Confusion Matrix")
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.savefig(f"{self.save_dir}/confusion_matrix.png")