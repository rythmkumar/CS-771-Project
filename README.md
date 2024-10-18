### Directory Structure
```
mini-project-1/
├── datasets/
│   ├── test/
│   │   ├── test_emoticon.csv
│   │   ├── test_feature.npz
│   │   └── test_text_seq.csv
│   ├── train/
│   │   ├── train_emoticon.csv
│   │   ├── train_feature.npz
│   │   └── train_text_seq.csv
│   └── valid/
│       ├── valid_emoticon.csv
│       ├── valid_feature.npz
│       └── valid_text_seq.csv
├── all_weights/
│   ├── final_logistic_dataset2/
│   │   ├── logistic_regression_model.pkl
│   │   └── scaler_model.pkl
│   ├── final_lstm_dataset3/
│   │   ├── encoder.pkl
│   │   └── lstm_model.h5
│   └── final_svm_dataset1/
│       ├── le.pkl
│       └── SVM_D1.pkl
├── predicted_values/
│   ├── pred_combined.txt
│   ├── pred_emoticon.txt
│   ├── pred_feat.txt
│   └── pred_textseq.txt
├── model_train_1.py
├── model_train_2.py
├── model_train_3.py
├── README.md
└── 69.py
```

## Installation Requirements

To set up the environment for this project, please install the following libraries:

- **TensorFlow**: Version 2.17
- **pickle**
- **scikit-learn**
- **tqdm**
- **pandas**
- **numpy**
- **joblib**

## Project Structure

This project includes a well-organized directory structure as follows:

### 1. Datasets

The `datasets/` folder contains all training, testing, and validation datasets for the three types of datasets:

- **train/**: Contains training datasets.
- **valid/**: Contains validation datasets.
- **test/**: Contains testing datasets.

### 2. Model Weights

The `all_weights/` folder contains the weights for the models that are necessary for predicting on the test datasets. Each subdirectory corresponds to different models.

### 3. Predicted Values

The `predicted_values/` folder includes the `.txt` files that store the predicted values for the test data, namely:
- `pred_feat.txt`
- `pred_emoticon.txt`
- `pred_textseq.txt`
- `pred_combined.txt`

### 4. Model Training Scripts

The following scripts contain the code to train models on the respective datasets:

- **model_train_1.py**: Trains models on Dataset 1 (Emoticons).
- **model_train_2.py**: Trains models on Dataset 2 (Deep Features).
- **model_train_3.py**: Trains models on Dataset 3 (Text Sequence).

### 5. Prediction Generation

The `69.py` file is the script that you need to run to generate the four prediction files mentioned above.
