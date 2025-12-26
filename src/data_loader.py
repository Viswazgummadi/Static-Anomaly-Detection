# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessor import DataPreprocessor

class DataLoader:
    def __init__(self, filepath, config):
        self.filepath = filepath
        self.config = config
        self.preprocessor = DataPreprocessor()

    def load_and_preprocess(self):
        print(f"[INFO] Loading data from {self.filepath}...")
        df = pd.read_csv(self.filepath)
        
        # 1. Drop Time column (usually irrelevant for simple Autoencoders)
        if 'Time' in df.columns:
            df = df.drop(['Time'], axis=1)
            
        # 2. Split Features and Labels
        # Class 0 = Normal, Class 1 = Fraud
        X = df.drop(['Class'], axis=1)
        y = df['Class']

        # 3. Split into Train and Test sets (Standard split first)
        # We start with a standard split to ensure our Test set has both Fraud and Non-Fraud
        X_train_all, X_test, y_train_all, y_test = train_test_split(
            X, y, 
            test_size=self.config['data']['test_size'], 
            random_state=self.config['data']['random_state'],
            stratify=y # Maintain ratio of fraud in test set
        )
        
        # 4. Filter Training Data: KEEP ONLY NORMAL TRANSACTIONS (Class 0)
        # The Autoencoder trains ONLY on normal data to learn "normality"
        train_mask = (y_train_all == 0)
        X_train_normal = X_train_all[train_mask]
        
        # We don't need y_train_normal because the target is the input itself (X -> X)
        
        print(f"[INFO] Training Data (Normal Only): {X_train_normal.shape}")
        print(f"[INFO] Test Data (Mixed): {X_test.shape}")

        # 5. Scaling
        # Fit scaler ONLY on training data to avoid data leakage
        print("[INFO] Scaling data...")
        X_train_scaled = self.preprocessor.fit_transform(X_train_normal)
        X_test_scaled = self.preprocessor.transform(X_test)
        
        # Save the scaler for later use in production
        self.preprocessor.save_scaler(self.config['paths']['scaler_save'])

        return X_train_scaled, X_test_scaled, y_test