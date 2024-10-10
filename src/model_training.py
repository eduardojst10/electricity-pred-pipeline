import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input


def lstm_model(input_shape, output_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    # First LSTM layer (with return sequences to pass to next LSTM)
    model.add(tf.keras.layers.LSTM(200, activation='relu', input_shape=input_shape, return_sequences=True))
    
    # Add a dropout layer for regularization
    model.add(Dropout(0.3))
    
    # Second LSTM layer
    model.add(tf.keras.layers.LSTM(50, activation='relu', return_sequences=True))
    
    # Dropout again
    model.add(Dropout(0.4))
    
    # Third LSTM layer (if you want even more depth)
    model.add(tf.keras.layers.LSTM(20, activation='relu'))
    
    # Dropout
    model.add(Dropout(0.3))
    
    # Dense layer to reduce dimensionality and combine features
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    
    # Output layer
    model.add(tf.keras.layers.Dense(units=output_shape))
    
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model



def lstm_model2(input_shape, output_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    # First LSTM layer (with return sequences to pass to next LSTM)
    model.add(tf.keras.layers.LSTM(100, activation='relu', input_shape=input_shape, return_sequences=True))
    
    # Add a dropout layer for regularization
    model.add(Dropout(0.3))
    
    # Second LSTM layer
    model.add(tf.keras.layers.LSTM(50, activation='relu', return_sequences=True))
    
    # Dropout again
    model.add(Dropout(0.3))
    
    # Third LSTM layer (if you want even more depth)
    model.add(tf.keras.layers.LSTM(20, activation='relu'))
    
    # Dropout
    model.add(Dropout(0.3))
    
    # Dense layer to reduce dimensionality and combine features
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    
    # Output layer
    model.add(tf.keras.layers.Dense(units=output_shape))
    
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def lstm_model_tuner(hp):
    model = Sequential()

    # Input Layer
    input_shape = (96, 370)  # Example input shape (adjust based on your data)
    model.add(Input(shape=input_shape))
    
    # First LSTM Layer
    model.add(LSTM(units=hp.Int('units_lstm1', min_value=50, max_value=200, step=50),
                   activation='relu', return_sequences=True))
    model.add(Dropout(hp.Float('dropout_lstm1', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Second LSTM Layer
    model.add(LSTM(units=hp.Int('units_lstm2', min_value=50, max_value=150, step=50),
                   activation='relu', return_sequences=True))
    model.add(Dropout(hp.Float('dropout_lstm2', min_value=0.2, max_value=0.5, step=0.1)))

    # Third LSTM Layer
    model.add(LSTM(units=hp.Int('units_lstm3', min_value=20, max_value=100, step=20),
                   activation='relu'))
    model.add(Dropout(hp.Float('dropout_lstm3', min_value=0.2, max_value=0.5, step=0.1)))

    # Dense Layer
    model.add(Dense(units=hp.Int('dense_units', min_value=20, max_value=100, step=20), activation='relu'))
    
    # Output Layer
    output_shape = 370  # Example output shape (based on your task)
    model.add(Dense(units=output_shape))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4]))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model