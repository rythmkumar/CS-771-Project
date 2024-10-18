# read feature dataset
#Step 1
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

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

