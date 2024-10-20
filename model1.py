# Loading library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tqdm import tqdm
import pickle
import joblib

# Loading the Testing dataset
dEmot = pd.read_csv("datasets/test/test_emoticon.csv")

EmX = np.array(dEmot.iloc[:,0].values)

# Emojis which appear in every datapoint
emojis = ['😛','😑','🛐','😣','🙯','🚼','🙼']

for i in range(len(EmX)):
  for j in range(len(emojis)):
    EmX[i] = EmX[i].replace(emojis[j], '')

# Loading the Labels for One Hot Encoding
le = pickle.load(open("all_weights/final_svm_dataset1/le.pkl", 'rb'))

# If a new emoji comes which is not in the training dataset, we set it to '0'.
for i in range(0, len(EmX)):
  for j in range(0, len(EmX[i])):
    if EmX[i][j] not in le.classes_:
      EmX[i] = EmX[i][:j] + '0' + EmX[i][(j + 1):]

TotalEm = []
for i in range(0, len(EmX)):
  for j in range(0, len(EmX[i])):
    TotalEm.append(EmX[i][j])
Transformation = le.transform(TotalEm)

# Loading the model weights
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

# Predicting with the SVM Model
Y_test = np.array(SVM.predict(X_test))

# Save the predictions to a text file
with open('pred_emoticon.txt', 'w') as f:
    for pred in Y_test:
        f.write(f"{pred}\n")
