import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
train_feat_X = train_feat['features']
train_feat_Y = train_feat['label']

val_feat=np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
val_feat_X = val_feat['features']
val_feat_Y = val_feat['label']

train_feat_select_X=train_feat_X[:, [1, 7, 12]]

# Step 1: Reshape the data (num_samples, 13, 786) to (num_samples, 3 * 786)
# train_feat_select_X=train_feat_X[:, [1, 7, 12]]
reshaped_train_feat_X = train_feat_select_X.reshape(train_feat_select_X.shape[0], -1)  # Shape: (num_samples, 3 * 786)

# Step 2: Standardize the data
scaler = StandardScaler()
scaled_train_feat_X = scaler.fit_transform(reshaped_train_feat_X)

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_train_feat_X, train_feat_Y, test_size=0.2, random_state=42)

# Step 4: Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)



val_feat_select_X=val_feat_X[:, [1, 7, 12]]
# reshaped_val_feat_X=val_feat_select_X.reshape(val_feat_select_X.shape[0], -1
X_val_test=scaler.transform(val_feat_select_X.reshape(val_feat_select_X.shape[0], -1))

# Step 5: Predict on the validation set
y_pred = model.predict(X_val_test)

# Step 6: Evaluate the model's accuracy
accuracy = accuracy_score(val_feat_Y, y_pred)
print(f"Model Accuracy on validation when trained on 80% train dataset: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(val_feat_Y, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# F1 Score
f1 = f1_score(val_feat_Y, y_pred)
print(f"F1 Score: {f1:.2f}")

# Step 8: Print the number of weights (including the intercept)
n_weights = model.coef_.size + model.intercept_.size
print(f"Number of weights in the model: {n_weights}")
