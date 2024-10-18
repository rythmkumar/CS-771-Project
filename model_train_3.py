import pandas as pd
import numpy as np
from tensorflow import keras  # Import keras directly from tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# read text sequence dataset
train_seq_df = pd.read_csv("/content/drive/MyDrive/datasets/train/train_text_seq.csv")
val_seq_df=pd.read_csv("/content/drive/MyDrive/datasets/valid/valid_text_seq.csv")

# Step 1: Load the dataset

# Step 2: Convert input_str to list of integers
train_seq_df['input_str'] = train_seq_df['input_str'].apply(lambda x: [int(char) for char in x])

# Step 3: Pad sequences (though here all are length 50, padding is not really required)
train_seq_X = np.array(train_seq_df['input_str'].tolist())  # Shape: (num_samples, 50)

# Step 4: Label encode the output labels (binary labels)
encoder = LabelEncoder()
train_seq_Y = encoder.fit_transform(train_seq_df['label'])  # 0 or 1

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_seq_X, train_seq_Y, test_size=0.2, random_state=42)

# # Step 6: Model Summary to check the number of trainable parameters
# Define the RNN model
model = keras.Sequential([ # Now 'keras' is recognized
    keras.layers.Embedding(input_dim=10, output_dim=96,input_length=50),  # Adjust input_dim based on your vocabulary size
    keras.layers.LSTM(16),  # Keep the number of units low to limit parameters
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation = 'relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
# Step 7: Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Step 9: Train the model
model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_test, y_test))

# Step 10: Pad sequences (though here all are length 50, padding is not really required)
val_seq_X = np.array(val_seq_df['input_str'].tolist())  # Shape: (num_samples, 50)
# Step 11: Convert input_str to list of integers
val_seq_df['input_str'] = val_seq_df['input_str'].apply(lambda x: [int(char) for char in x])

# Step 12: Pad sequences (though here all are length 50, padding is not really required)
val_seq_X = np.array(val_seq_df['input_str'].tolist())  # Shape: (num_samples, 50)
# Step 13: Label encode the output labels (binary labels)
val_seq_Y = encoder.transform(val_seq_df['label'])
# Evaluate the model on the new validation set
loss, accuracy = model.evaluate(val_seq_X, val_seq_Y)
print('Validation accuracy:', accuracy)
