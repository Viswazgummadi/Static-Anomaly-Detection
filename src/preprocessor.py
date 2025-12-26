# src/preprocessor.py
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler 
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()        
        
    def fit(self, data):
        self.scaler.fit(data)
        
    def transform(self, data):
        return self.scaler.transform(data)
    
    def fit_transform(self, data):
        return self.scaler.fit_transform(data)
    
    def save_scaler(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"[INFO] Scaler saved to {path}")
        
    def load_scaler(self, path):
        self.scaler = joblib.load(path)
        print(f"[INFO] Scaler loaded from {path}")