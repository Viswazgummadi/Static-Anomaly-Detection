# test_data.py
import yaml
from src.data_loader import DataLoader

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize Loader
loader = DataLoader(config['paths']['raw_data'], config)

# Run pipeline
X_train, X_test, y_test = loader.load_and_preprocess()

print("-" * 30)
print(f"X_train shape (Should be ~227k, 29): {X_train.shape}")
print(f"X_test shape (Should be ~57k, 29): {X_test.shape}")
print("Data Loading Successful!")
