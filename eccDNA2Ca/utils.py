"""
Main module for eccDNA2Ca: an ensemble learning model combining XGBoost and CNN
for linking eccDNA to cancer.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from Bio import SeqIO
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import xgboost as xgb

# ------------------------------
# Utility functions
# ------------------------------
def load_sequences(fasta_file):
    seq = SeqIO.parse(open(fasta_file), "fasta")
    return [str(fasta.seq) for fasta in seq]

def string_to_array(my_string):
    my_string = re.sub('[^ACGT]', 'N', my_string)
    return np.array(list(my_string))

def one_hot(array, label_encoder):
    array = np.where(~np.isin(array, ['A', 'C', 'G', 'T']), 'N', array)
    integer_encoded = label_encoder.transform(array)
    onehot_encoder = OneHotEncoder(sparse_output=False, dtype=int)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return onehot_encoder.fit_transform(integer_encoded)

def build_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, kernel_size=3, activation='relu', kernel_initializer='he_normal'),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation='relu', kernel_initializer='he_normal'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=['accuracy'])
    return model

# ------------------------------
# Main training and prediction class
# ------------------------------
class eccDNA2Ca:
    def __init__(self, data_path=None, fasta_path=None, model_dir="models"):
        self.data_path = data_path
        self.fasta_path = fasta_path
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.label_encoder = LabelEncoder().fit(['A', 'C', 'G', 'T', 'N'])

    def train(self):
        # Load and prepare tabular features for XGBoost
        features = pd.read_excel(self.data_path)
        x_xgb = features.iloc[:,:-1]
        y = np.array(features.iloc[:,-1])
        scaler = StandardScaler()
        x_scaled = pd.DataFrame(scaler.fit_transform(x_xgb), columns=x_xgb.columns)

        xgb_model = xgb.XGBClassifier(objective='binary:logistic',
                                      n_estimators=50, colsample_bytree=1,
                                      subsample=0.9, gamma=0, learning_rate=0.1,
                                      max_depth=4, random_state=42)
        xgb_model.fit(x_scaled, y)

        # Save XGBoost model and scaler
        xgb_model.save_model(os.path.join(self.model_dir, "xgb_model.json"))
        joblib.dump(scaler, os.path.join(self.model_dir, "scaler.pkl"))

        # Load and prepare sequences for CNN
        sequences = load_sequences(self.fasta_path)
        one_hot_matrices = [one_hot(string_to_array(seq), self.label_encoder) for seq in sequences]
        padded_matrices = pad_sequences(one_hot_matrices, maxlen=25000, padding='post', dtype='float32', value=0)

        cnn_model = build_cnn(padded_matrices.shape[1:])
        cnn_model.fit(padded_matrices, y, epochs=50, batch_size=32,
                      callbacks=[EarlyStopping(patience=10, monitor='loss')], verbose=0)

        cnn_model.save(os.path.join(self.model_dir, "cnn_model.h5"))

    def predict(self, fasta_file=None, xgb_features_df=None):
        # Load models
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(os.path.join(self.model_dir, "xgb_model.json"))
        cnn_model = load_model(os.path.join(self.model_dir, "cnn_model.h5"))

        # Check if fasta_file and xgb_features_df are provided
        if fasta_file is not None:
            # Process CNN input
            sequences = load_sequences(fasta_file)
            one_hot_matrices = [one_hot(string_to_array(seq), self.label_encoder) for seq in sequences]
            padded_matrices = pad_sequences(one_hot_matrices, maxlen=25000, padding='post', dtype='float32', value=0)
            cnn_probs = cnn_model.predict(padded_matrices).flatten()
        else:
            cnn_probs = np.array([])  # No CNN input, set to empty array

        if xgb_features_df is not None:
            # Process XGB input
            scaler = joblib.load(os.path.join(self.model_dir, "scaler.pkl"))
            xgb_features_df = pd.DataFrame(scaler.transform(xgb_features_df), columns=xgb_features_df.columns)
            xgb_probs = xgb_model.predict_proba(xgb_features_df)[:, 1]
        else:
            xgb_probs = np.array([])  # No XGBoost input, set to empty array

        # If both are provided, apply soft voting
        if len(cnn_probs) > 0 and len(xgb_probs) > 0:
            # Soft voting ensemble
            ensemble_probs = (xgb_probs + cnn_probs) / 2
            return ensemble_probs
        elif len(cnn_probs) > 0:
            return cnn_probs  # Only CNN prediction
        elif len(xgb_probs) > 0:
            return xgb_probs  # Only XGBoost prediction
        else:
            raise ValueError("At least one of fasta_file or xgb_features_df must be provided for prediction.")