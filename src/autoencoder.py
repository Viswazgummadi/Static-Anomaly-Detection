# src/autoencoder.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1

class AutoencoderBuilder:
    @staticmethod
    def build(input_dim, config):
        """
        Builds a Robust Autoencoder.
        """
        encoding_dim = config['model']['encoding_dim'] 
        hidden_dim = config['model']['hidden_dim']
        learning_rate = config['model']['learning_rate']

        model = Sequential([
            Input(shape=(input_dim,)),
            
            # Encoder
            Dense(encoding_dim, activation="relu"),
            BatchNormalization(),  # Keeps values stable
            Dropout(0.2),          # Prevents overfitting
            
            # Bottleneck (Latent Space)
            # activity_regularizer=l1(10e-5) forces the model to ignore noise
            Dense(hidden_dim, activation="relu", activity_regularizer=l1(10e-5), name="bottleneck"),
            
            # Decoder
            Dense(encoding_dim, activation="relu"),
            Dropout(0.2),
            
            # Output
            Dense(input_dim, activation="linear")
        ])

        optimizer = Adam(learning_rate=learning_rate)
        
        # We stick to MSE, but the internal layers are now much smarter
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        return model