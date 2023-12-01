import argparse
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import unidecode
import Levenshtein as lev

def getArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fullpath_denomination_labo",
                        help="Complete path to the file containing lab articles' names", type=str)
    parser.add_argument("--fullpath_denomination_sap",
                        help="Complete path to the file containing SAP articles' names", type=str)
    parser.add_argument("--fullpath_output",
                        help="Complete path to the output file containing Levenshtein distances", type=str)
    parser.add_argument("--model_output_path",
                        help="Complete path to save the trained model", type=str, default="levenshtein_model.h5")

    args = parser.parse_args()

    print("--fullpath_denomination_labo:", args.fullpath_denomination_labo)
    print("--fullpath_denomination_sap:", args.fullpath_denomination_sap)
    print("--fullpath_output:", args.fullpath_output)
    print("--model_output_path:", args.model_output_path)

    return args

def preprocess_text(text):
    text = text.upper()
    text = unidecode.unidecode(text)
    text = re.sub('[^0-9A-Z]*', '', text)
    return text

def create_model(input_dim):
    model = keras.Sequential([
        keras.layers.Embedding(input_dim=input_dim, output_dim=32, input_length=100),
        keras.layers.LSTM(64),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def train_model(model, X_train, y_train, epochs=5):
    model.fit(X_train, y_train, epochs=epochs, verbose=1)

def main():
    args = getArguments()

    labo_data = np.loadtxt(args.fullpath_denomination_labo, dtype=str, delimiter='\t', usecols=(0, 1, 2))
    sap_data = np.loadtxt(args.fullpath_denomination_sap, dtype=str, delimiter='\t', usecols=(0, 1))

    labo_texts = [preprocess_text(row[2]) for row in labo_data]
    sap_texts = [preprocess_text(row[1]) for row in sap_data]

    levenshtein_distances = []

    for labo_text in labo_texts:
        distances = [lev.distance(labo_text, sap_text) / len(labo_text) for sap_text in sap_texts]
        min_distance = min(distances)
        levenshtein_distances.append(min_distance)

    levenshtein_distances = np.array(levenshtein_distances).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(labo_texts, levenshtein_distances, test_size=0.2, random_state=42)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    vocab_size = len(label_encoder.classes_)

    X_train_padded = keras.preprocessing.sequence.pad_sequences(
        label_encoder.transform(X_train),
        padding="post",
        truncating="post",
        maxlen=100
    )

    model = create_model(vocab_size)
    train_model(model, X_train_padded, y_train_encoded)

    # Save the model
    model.save(args.model_output_path)

    # Evaluate the model on the test set
    X_test_padded = keras.preprocessing.sequence.pad_sequences(
        label_encoder.transform(X_test),
        padding="post",
        truncating="post",
        maxlen=100
    )
    y_test_encoded = label_encoder.transform(y_test)

    predictions = model.predict(X_test_padded)
    predictions_unscaled = label_encoder.inverse_transform(predictions.flatten())
    mae = mean_absolute_error(y_test, predictions_unscaled)
    print(f"Mean Absolute Error on Test Set: {mae}")

    # Save the Levenshtein distances to the output file
    np.savetxt(args.fullpath_output, levenshtein_distances, fmt='%.2f')

if __name__ == "__main__":
    main()
