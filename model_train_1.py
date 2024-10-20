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

# Learning with Prototype Model
class LwP:
  def __init__(self):
    self.u0 = None
    self.u1 = None
  def fit(self, X, Y):
    self.u0 = np.mean(X[Y == 0])
    self.u1 = np.mean(X[Y == 1])
  def predict(self, X):
    Y = []
    for i in range(0, len(X)):
      if np.linalg.norm(X[i] - self.u0) < np.linalg.norm(X[i] - self.u1):
        Y.append(0)
      else:
        Y.append(1)
    return np.array(Y)

# Loading the training dataset
dEmot = pd.read_csv("datasets/train/train_emoticon.csv")

EmX = np.array(dEmot.iloc[:,0].values)
EmY = np.array(dEmot.iloc[:,1].values)

# Removing emojis which appear in every datapoint
emojis = ['😛','😑','🛐','😣','🙯','🚼','🙼']

for i in range(len(EmX)):
  for j in range(len(emojis)):
    EmX[i] = EmX[i].replace(emojis[j], '')

# Finding the One Hot Encoding transformation
TotalEm = []
for i in range(0, len(EmX)):
  for j in range(0, len(EmX[i])):
    TotalEm.append(EmX[i][j])
le = LabelEncoder()
NewEm = TotalEm
NewEm.extend(['0'])
le.fit(NewEm)
Transformation = le.transform(TotalEm)

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

pickle.dump(le, open("le.pkl", 'wb'))

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

X_train = New_X
Y_train = EmY

# Fitting SVM Model
SVCClf = SVC(kernel = 'linear',gamma = 'scale', shrinking = False,)
SVCClf.fit(X_train, Y_train)

# Fitting Logistic Regression
LogReg = LogisticRegression()
LogReg.fit(X_train, Y_train)

# Fitting Random Forest
RF = RandomForestClassifier(n_estimators = 100)
RF.fit(X_train, Y_train)

# Fitting Learning With Prototypes
LP = LwP()
LP.fit(X_train, Y_train)

# Saving the models
joblib.dump(SVCClf, "SVM_D1.pkl")

joblib.dump(RF, "Random_Forest_D1.pkl")

joblib.dump(LogReg, "Logistic_D1.pkl")

joblib.dump(LP, "LwP_D1.pkl")
