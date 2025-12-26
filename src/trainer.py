# src/trainer.py
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class AutoencoderTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self, X_train, X_test):
        """
        Train the model with Early Stopping and Checkpointing.
        Note: We pass X_train as both input AND target (Autoencoder).
        """
        epochs = self.config['train']['epochs']
        batch_size = self.config['train']['batch_size']
        patience = self.config['train']['patience']
        model_save_path = self.config['paths']['model_save']

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        # Callbacks for production-grade training
        checkpoint = ModelCheckpoint(
            model_save_path, 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=True, 
            mode='min'
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=patience, 
            verbose=1,
            restore_best_weights=True
        )

        print(f"[INFO] Starting training for {epochs} epochs...")
        
        history = self.model.fit(
            X_train, X_train,  # Input == Target
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_test, X_test), # Validate reconstruction on test set
            callbacks=[checkpoint, early_stop],
            verbose=1
        )
        
        print(f"[INFO] Training complete. Best model saved to {model_save_path}")
        return history