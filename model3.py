import pandas as pd
import numpy as np
from tensorflow import keras  # Import keras directly from tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import joblib
# print(tf.__version__)

# Load the LabelEncoder
encoder = joblib.load('all_weights/final_lstm_dataset3/encoder.pkl')

# Load the model weights
model = keras.models.load_model('all_weights/final_lstm_dataset3/lstm_model.h5') 

test_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")['input_str']

# Step 2: Convert input_str to list of integers
test_seq_X = test_seq_X.apply(lambda x: [int(char) for char in x])

# Step 3: Pad sequences (though here all are length 50, padding is not really required)
test_seq_X = np.array(test_seq_X.tolist())  # Shape: (num_samples, 50)

predictions = model.predict(test_seq_X)
# Convert predictions to binary classes (assuming output is a probability)
binary_predictions = (predictions > 0.5).astype(int)  # Threshold at 0.5 for binary classification


# Save binary predictions to a file
np.savetxt('pred_textseq.txt', binary_predictions, fmt='%d')

print("Encoder and model weights loaded successfully.")


