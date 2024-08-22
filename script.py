import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pdfplumber
import sys

def extract_text_from_pdfs(dir):
    texts = []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(subdir, file)
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    texts.append(text)
    return texts

def predict_labels(texts):
    try:
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')

        if padded_sequences.size == 0:
            raise ValueError("Padded sequences are empty.")

        predictions = model.predict(padded_sequences)
        predicted_labels = np.argmax(predictions, axis=1)

        label_names = label_encoder.inverse_transform(predicted_labels)
        return label_names
    except Exception as e:
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    test_data_dir = sys.argv[1]

    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    model = tf.keras.models.load_model('best_model.keras')

    texts = extract_text_from_pdfs(test_data_dir)

    if not texts:
        raise ValueError("No texts extracted from PDFs.")

    predicted_labels = predict_labels(texts)

    pdf_files = [os.path.basename(file) for file in os.listdir(test_data_dir) if file.endswith('.pdf')]
    results_df = pd.DataFrame({'pdf_file': pdf_files, 'predicted_label': predicted_labels, 'text': texts})
    results_df.to_csv('predicted_labels.csv', index=False)
