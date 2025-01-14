import re
import numpy as np
import argparse
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction import FeatureHasher

# Function to load and process sequences
def load_and_process_sequences(fasta_path, model_type="CNN", k=4):
    seq = SeqIO.parse(fasta_path, 'fasta')
    sequences = [str(fasta.seq) for fasta in seq]

    if model_type == "CNN":
        # One-hot encoding for CNN model
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
        padded_matrices = pad_sequences(one_hot_matrices, maxlen=max_length, padding="post", value=0, dtype='float32')

        return padded_matrices

    elif model_type == "LSTM":
        # k-mer feature hashing for LSTM model
        def generate_kmers(sequence, k):
            return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

        kmer_list = [generate_kmers(seq, k) for seq in sequences]

        vectorizer = FeatureHasher(n_features=4 ** k, input_type="string")
        hashed_vectors = vectorizer.transform(kmer_list).toarray()
        hashed_vectors = np.expand_dims(hashed_vectors, axis=-1)

        return hashed_vectors

    else:
        raise ValueError("Model type should be either 'CNN' or 'LSTM'.")


# Load trained model and make predictions
def load_model_and_predict(model_path, fasta_path, model_type="CNN", k=4):
    model = load_model(model_path)
    X_new = load_and_process_sequences(fasta_path, model_type=model_type, k=k)
    y_pred = model.predict(X_new)
    y_pred_class = (y_pred > 0.5).astype(int)

    return y_pred, y_pred_class


# Main function to parse arguments and call the prediction function
def main():
    # Argument parsing for the command-line interface
    parser = argparse.ArgumentParser(description="Predict eccDNA sequences using trained models.")
    parser.add_argument("fasta_file", type=str, help="Path to the FASTA file containing the sequences.")
    parser.add_argument("--model", choices=["CNN", "LSTM"], default="CNN", help="Choose the model type.")

    args = parser.parse_args()

    model_path = f"TrainedModels/eccDNA2Ca_{args.model}_module.h5"

    print(f"Predicting using the {args.model} model...")
    y_pred, y_pred_class = load_model_and_predict(model_path, args.fasta_file, model_type=args.model)

    # Output
    print("Predicted probabilities:\n", y_pred)
    print("Predicted classes:\n", y_pred_class)


if __name__ == "__main__":
    main()