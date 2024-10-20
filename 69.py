import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from tqdm import tqdm
from tensorflow import keras  # Import keras directly from tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
import joblib

#Step 1

dEmot = pd.read_csv("datasets/test/test_emoticon.csv")

EmX = np.array(dEmot.iloc[:,0].values)

emojis = ['😛','😑','🛐','😣','🙯','🚼','🙼']

for i in range(len(EmX)):
  for j in range(len(emojis)):
    EmX[i] = EmX[i].replace(emojis[j], '')

le = pickle.load(open("all_weights/final_svm_dataset1/le.pkl", 'rb'))

for i in range(0, len(EmX)):
  for j in range(0, len(EmX[i])):
    if EmX[i][j] not in le.classes_:
      EmX[i] = EmX[i][:j] + '0' + EmX[i][(j + 1):]

TotalEm = []
for i in range(0, len(EmX)):
  for j in range(0, len(EmX[i])):
    TotalEm.append(EmX[i][j])
Transformation = le.transform(TotalEm)

SVM =  joblib.load("all_weights/final_svm_dataset1/SVM_D1.pkl")

New_X = []
for i in range(0, len(EmX)):
  A = []
  for j in range(0, len(EmX[i])):
    V = Transformation[i*len(EmX[i])+j]
    L = []
    for k in range(0, len(le.classes_)):
      if k == V:
        L.append(1)
      else:
        L.append(0)
    A.extend(L)
  New_X.append(A)

def transform(x):
  C = []
  TotalX = []
  for i in range(len(x)):
    for j in range(len(emojis)):
      x[i] = x[i].replace(emojis[j], '')
  for i in range(0, len(x)):
    for j in range(0, len(EmX[i])):
      TotalX.append(x[i][j])
  Transformation = le.transform(TotalX)
  for i in range(0, len(x)):
    A = []
    for j in range(0, len(EmX[i])):
      V = Transformation[i*len(EmX[i])+j]
      L = []
      for k in range(0, len(le.classes_)):
        if k == V:
          L.append(1)
        else:
         L.append(0)
      A.extend(L)
    C.append(A)
  return np.array(C)

New_X = np.array(New_X)

X_test = New_X

Y_test = np.array(SVM.predict(X_test))

# Save the predictions to a text file
with open('pred_emoticon.txt', 'w') as f:
    for pred in Y_test:
        f.write(f"{pred}\n")


test_feat_X = np.load("datasets/test/test_feature.npz", allow_pickle=True)['features']

# Load the saved model and scaler
model = joblib.load('all_weights/final_logistic_dataset2/logistic_regression_model.pkl')     # Replace with the path to your model file
scaler = joblib.load('all_weights/final_logistic_dataset2/scaler_model.pkl')   # Replace with the path to your scaler file

# test_feat_X select the desired indexes
test_feat_X=test_feat_X[:, [1, 7, 12]]
# reshaped_val_feat_X=val_feat_select_X.reshape(val_feat_select_X.shape[0], -1
scaled_test_data=scaler.transform(test_feat_X.reshape(test_feat_X.shape[0], -1))
# Predict the values using the loaded model
predictions = model.predict(scaled_test_data)

# Save the predictions to a text file
with open('pred_deepfeat.txt', 'w') as f:
    for pred in predictions:
        f.write(f"{pred}\n")

print("Predictions saved to pred_deepfeat.txt")

with open('pred_combined.txt', 'w') as f:
    for pred in predictions:
        f.write(f"{pred}\n")

print("Predictions saved to pred_combined.txt")



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

