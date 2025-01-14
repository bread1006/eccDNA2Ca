from Bio import SeqIO
import re,os,random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_extraction import FeatureHasher
from tensorflow.keras.regularizers import l2


class eccDNA2Ca:
    def __init__(self, fasta_path, label_path, sheet_name, model_type="CNN", epochs=None, batch_size=32, npatience=20,
                  k=4, l2=0.0001, lr=0.0001):
        self.fasta_path = fasta_path
        self.label_path = label_path
        self.sheet_name = sheet_name
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.npatience = npatience
        self.k = k  # k-mer size for LSTM
        self.l2 = l2
        self.lr = lr
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        if epochs is None:
            if self.model_type == "CNN":
                self.epochs = 50  # Set default epochs for CNN
            elif self.model_type == "LSTM":
                self.epochs = 60  # Set default epochs for LSTM
            else:
                raise ValueError("Model type should be either 'CNN' or 'LSTM'.")
        else:
            self.epochs = epochs

    def load_data(self):
        if self.model_type == "CNN":
            #If using cnn, process the sequence through one-hot
            seq = SeqIO.parse(self.fasta_path, 'fasta')
            sequences = [str(fasta.seq) for fasta in seq]

            label_encoder = LabelEncoder()
            label_encoder.fit_transform(np.array(["A", 'C', 'G', 'T']))

            def string_to_array(my_string):
                my_string = re.sub("[^ACGT]", "N", my_string)
                my_array = np.array(list(my_string))
                return my_array

            def one_hot(array):
                integer_encoded = label_encoder.transform(array)
                onehot_encoder = OneHotEncoder(sparse_output=False, dtype=int)
                integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
                onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
                return onehot_encoded

            one_hot_matrices = [one_hot(string_to_array(seq)) for seq in sequences]
            max_length = 25000
            padded_matrices = pad_sequences(one_hot_matrices, maxlen=max_length, padding="post", value=0,
                                            dtype='float32')

            label = np.array(pd.read_excel(self.label_path, sheet_name=self.sheet_name).iloc[:, -1])

            self.X_train = padded_matrices
            self.y_train = label

        elif self.model_type == "LSTM":
            # If using LSTM, process the sequence through kmer-hashing
            seq = SeqIO.parse(self.fasta_path, "fasta")
            sequences = [str(fasta.seq) for fasta in seq]

            def generate_kmers(sequence, k):
                return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

            kmer_list = [generate_kmers(seq, self.k) for seq in sequences]

            vectorizer = FeatureHasher(n_features=4 ** self.k, input_type="string")
            hashed_vectors = vectorizer.transform(kmer_list).toarray()
            hashed_vectors = np.expand_dims(hashed_vectors, axis=-1)

            label = np.array(pd.read_excel(self.label_path, sheet_name=self.sheet_name).iloc[:, -1])

            self.X_train = hashed_vectors
            self.y_train = label

        else:
            raise ValueError("Model type should be either 'CNN' or 'LSTM'.")

    def get_cnn_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(32, kernel_size=3, activation="relu", kernel_initializer="he_normal"),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation="relu", kernel_initializer="he_normal"),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation="relu", kernel_initializer="he_normal"),
            Dropout(0.2),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def get_lstm_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=True, kernel_regularizer=l2(self.l2)),
            LSTM(64, kernel_regularizer=l2(self.l2)),
            Dense(32, activation="tanh", kernel_regularizer=l2(self.l2)),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer=Adam(learning_rate=self.lr), loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def train(self):
        if self.model_type == "CNN":
            self.model = self.get_cnn_model(input_shape=self.X_train.shape[1:])
        elif self.model_type == "LSTM":
            self.model = self.get_lstm_model(input_shape=self.X_train.shape[1:])
        else:
            raise ValueError("Model type must be either 'CNN' or 'LSTM'.")

        early_stop = EarlyStopping(patience=self.npatience, monitor="loss", mode="min")
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                       callbacks=[early_stop], verbose=0)

        # Save the trained model
        if not os.path.exists("TrainedModels"):
            os.makedirs("TrainedModels")

        model_path = f"TrainedModels/eccDNA2Ca_{self.model_type}_module.h5"
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        y_pred_class = (y_pred > 0.5).astype(int)
        return y_pred,y_pred_class


fasta_path = "D:/lina/papers/eccPred/Data_cancer/eccDNA.fasta"
label_path = "D:/lina/papers/eccPred/Data_cancer/Data_cancer.xlsx"
sheet_name = "annotation"

seed=42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
model1 = eccDNA2Ca(fasta_path=fasta_path, label_path=label_path, sheet_name=sheet_name, model_type="CNN")
model1.load_data()
model1.train()

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
model2 = eccDNA2Ca(fasta_path=fasta_path, label_path=label_path, sheet_name=sheet_name, model_type="LSTM")
model2.load_data()
model2.train()