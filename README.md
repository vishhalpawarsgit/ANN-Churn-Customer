# Customer Churn Prediction Using Artificial Neural Network

A machine learning project that predicts whether a bank customer is likely to churn using an **Artificial Neural Network (ANN)** built with **TensorFlow/Keras**.

The project includes data validation, preprocessing, ANN model training, evaluation, and hyperparameter tuning using `GridSearchCV`.

---

## 📌 Project Overview

Customer churn prediction helps organizations identify customers who are likely to leave their services.

In this project, the **Churn Modelling dataset** is used to build a binary classification model that predicts the `Exited` status of a customer.

### Prediction Target

* `0` → Customer did not churn
* `1` → Customer churned

The project follows the complete machine learning workflow:

```text
Dataset
   ↓
Data Validation
   ↓
Data Preprocessing
   ↓
Train/Test Split
   ↓
Feature Scaling & Encoding
   ↓
ANN Model
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Hyperparameter Tuning
   ↓
Save Best Model
```

---

# 📂 Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── Churn_Modelling.csv
│
├── artifacts/
│   ├── preprocessor.pkl
│   ├── ann_model.h5
│   └── tuned_ann_model.h5
│
├── main.py
│
├── README.md
│
└── requirements.txt
```

> The `artifacts/` directory is automatically created by the Python script if it does not already exist.

---

# 📊 Dataset

The project uses the **Churn Modelling** dataset.

The dataset contains customer information such as credit score, geography, age, balance, number of products, activity status, and estimated salary.

## Dataset Columns

| Column            | Description                                         |
| ----------------- | --------------------------------------------------- |
| `RowNumber`       | Row identifier                                      |
| `CustomerId`      | Unique customer identifier                          |
| `Surname`         | Customer surname                                    |
| `CreditScore`     | Customer credit score                               |
| `Geography`       | Customer's country/geography                        |
| `Gender`          | Customer gender                                     |
| `Age`             | Customer age                                        |
| `Tenure`          | Number of years the customer has been with the bank |
| `Balance`         | Customer account balance                            |
| `NumOfProducts`   | Number of products used by the customer             |
| `HasCrCard`       | Whether the customer has a credit card              |
| `IsActiveMember`  | Whether the customer is an active member            |
| `EstimatedSalary` | Estimated customer salary                           |
| `Exited`          | Target variable indicating customer churn           |

---

# 🛠️ Technologies Used

The project is built using the following technologies:

* Python
* Pandas
* Scikit-learn
* TensorFlow
* Keras
* SciKeras
* Pickle

### Main Libraries

```python
from pathlib import Path
import pickle
import pandas as pd

import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from scikeras.wrappers import KerasClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

Navigate into the project directory:

```bash
cd customer-churn-prediction
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

Install the required packages:

```bash
pip install pandas scikit-learn tensorflow scikeras
```

Alternatively, create a `requirements.txt` file containing:

```text
pandas
scikit-learn
tensorflow
scikeras
```

Then install:

```bash
pip install -r requirements.txt
```

---

# 📁 Dataset Setup

Create a `data` directory in the project root:

```text
data/
```

Place the dataset inside the directory:

```text
data/Churn_Modelling.csv
```

The expected path is:

```text
project/
├── data/
│   └── Churn_Modelling.csv
```

If the dataset is not found, the program will raise a `FileNotFoundError`.

---

# 🔍 Data Validation

Before model training, the script performs several validation checks.

The following information is displayed:

* Number of rows
* Number of columns
* Column names
* First five records
* Data types
* Non-null counts
* Summary statistics
* Missing values
* Duplicate rows

The script also validates that all required columns are present.

If required columns are missing, a `ValueError` is raised.

---

# 🧹 Data Preprocessing

## Removing Unnecessary Columns

The following columns are removed before model training:

```text
RowNumber
CustomerId
Surname
```

These columns are not useful for predicting customer churn.

The target variable `Exited` is separated from the feature set.

```python
X = data.drop(
    columns=["RowNumber", "CustomerId", "Surname", "Exited"]
)

y = data["Exited"]
```

---

# 🔢 Feature Selection

## Numerical Features

The following numerical features are used:

```text
CreditScore
Age
Tenure
Balance
NumOfProducts
HasCrCard
IsActiveMember
EstimatedSalary
```

These features are standardized using `StandardScaler`.

```python
StandardScaler()
```

---

## 🔤 Categorical Features

The following categorical features are used:

```text
Geography
Gender
```

They are converted into numerical values using one-hot encoding:

```python
OneHotEncoder(drop="first")
```

Using `drop="first"` removes the first category to avoid redundant dummy variables.

---

# ✂️ Train/Test Split

The dataset is divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Split Configuration

| Parameter      | Value |
| -------------- | ----: |
| Training data  |   80% |
| Testing data   |   20% |
| Random state   |    42 |
| Stratification |   Yes |

Using `stratify=y` helps maintain a similar target-class distribution in both datasets.

---

# 🔄 Preprocessing Pipeline

A `ColumnTransformer` is used to apply different preprocessing techniques to numerical and categorical features.

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(drop="first"),
            categorical_features
        )
    ]
)
```

The preprocessing pipeline is fitted using the training data:

```python
X_train_processed = preprocessor.fit_transform(X_train)
```

The same fitted preprocessor is then used to transform the test data:

```python
X_test_processed = preprocessor.transform(X_test)
```

This ensures that the test data is transformed using the same preprocessing configuration as the training data.

---

# 💾 Saving the Preprocessor

The fitted preprocessing pipeline is saved using Python's `pickle` module.

```text
artifacts/preprocessor.pkl
```

This is important because the same preprocessing steps must be applied when making predictions on new customer data.

---

# 🧠 Artificial Neural Network

The project uses an Artificial Neural Network for binary classification.

The ANN architecture consists of:

```text
Input Layer
     ↓
Hidden Layer
     ↓
Hidden Layer
     ↓
Output Layer
```

The number of hidden layers and neurons can be changed during hyperparameter tuning.

---

# 🏗️ ANN Architecture

The ANN is created using the following function:

```python
def build_ann(
    neurons=64,
    layers=2,
    learning_rate=0.001
):
```

### Default Configuration

| Parameter                |               Value |
| ------------------------ | ------------------: |
| Hidden layers            |                   2 |
| Neurons per hidden layer |                  64 |
| Activation               |                ReLU |
| Output neurons           |                   1 |
| Output activation        |             Sigmoid |
| Optimizer                |                Adam |
| Learning rate            |               0.001 |
| Loss                     | Binary Crossentropy |

---

# 🔥 Activation Functions

## ReLU

The hidden layers use the ReLU activation function:

```python
Dense(neurons, activation="relu")
```

ReLU introduces non-linearity into the network and is commonly used in hidden layers of neural networks.

## Sigmoid

The output layer uses:

```python
Dense(1, activation="sigmoid")
```

The sigmoid function produces a probability between `0` and `1`, making it suitable for binary classification.

---

# 🎯 Model Compilation

The model is compiled using the Adam optimizer:

```python
optimizer = Adam(
    learning_rate=learning_rate
)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

### Loss Function

`binary_crossentropy` is used because the problem is a binary classification problem.

---

# 🚀 Model Training

The baseline model is trained using:

```python
history = model.fit(
    X_train_processed,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)
```

### Training Configuration

| Parameter        |   Value |
| ---------------- | ------: |
| Maximum epochs   |     100 |
| Batch size       |      32 |
| Validation split |     20% |
| Early stopping   | Enabled |

---

# ⏹️ Early Stopping

Early stopping is used to help prevent overfitting.

```python
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

The training process monitors validation loss.

If validation loss does not improve for five consecutive epochs, training stops and the best model weights are restored.

---

# 📈 Model Evaluation

The trained ANN is evaluated on the test dataset.

```python
test_loss, test_accuracy = model.evaluate(
    X_test_processed,
    y_test
)
```

The model then generates churn probabilities:

```python
probabilities = model.predict(
    X_test_processed
)

probabilities = probabilities.flatten()
```

These probabilities are converted into binary predictions using a threshold of `0.5`:

```python
predictions = (
    probabilities >= 0.5
).astype(int)
```

---

# 📊 Evaluation Metrics

The following metrics are calculated:

## Accuracy

Measures the percentage of total predictions that are correct.

```python
accuracy_score(
    y_test,
    predictions
)
```

## Precision

Measures how many customers predicted as churners actually churned.

```python
precision_score(
    y_test,
    predictions
)
```

## Recall

Measures how many actual churners were correctly identified.

```python
recall_score(
    y_test,
    predictions
)
```

## F1 Score

The F1 score provides a balance between precision and recall.

```python
f1_score(
    y_test,
    predictions
)
```

## ROC AUC

ROC AUC measures the model's ability to distinguish between churned and non-churned customers.

```python
roc_auc_score(
    y_test,
    probabilities
)
```

---

# 💾 Saving the Baseline Model

The trained baseline ANN is saved as:

```text
artifacts/ann_model.h5
```

The model can later be loaded using TensorFlow/Keras.

---

# 🔎 Hyperparameter Tuning

The project uses `GridSearchCV` with `SciKeras` to search for better ANN hyperparameters.

Hyperparameters being tested:

```python
param_grid = {
    "neurons": [32, 64],
    "layers": [1, 2],
    "learning_rate": [0.001, 0.01],
    "batch_size": [32],
    "epochs": [50]
}
```

---

# 🧪 Grid Search Configuration

The grid search uses 3-fold cross-validation:

```python
grid_search = GridSearchCV(
    estimator=model_classifier,
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,
    n_jobs=1,
    verbose=1
)
```

The best combination of hyperparameters is selected based on accuracy.

---

# 🔧 Hyperparameters Tested

| Hyperparameter   | Values      |
| ---------------- | ----------- |
| Neurons          | 32, 64      |
| Hidden layers    | 1, 2        |
| Learning rate    | 0.001, 0.01 |
| Batch size       | 32          |
| Epochs           | 50          |
| Cross-validation | 3-fold      |
| Scoring          | Accuracy    |

This results in:

```text
2 × 2 × 2 × 1 × 1 = 8
```

different hyperparameter combinations.

With 3-fold cross-validation, the grid search performs:

```text
8 × 3 = 24
```

model fits.

---

# 🏆 Best Model

After grid search, the best estimator is obtained using:

```python
best_model = grid_search_result.best_estimator_
```

The best hyperparameters can be displayed using:

```python
best_params = grid_search_result.best_params_

print(best_params)
```

The underlying Keras model is then extracted:

```python
tuned_keras_model = best_model.model_
```

---

# 💾 Saving the Tuned Model

The tuned model is saved to:

```text
artifacts/tuned_ann_model.h5
```

This model represents the best ANN configuration found by the grid search based on the selected scoring metric.

---

# 📁 Generated Artifacts

After successfully running the project, the `artifacts/` directory will contain:

```text
artifacts/
│
├── preprocessor.pkl
├── ann_model.h5
└── tuned_ann_model.h5
```

### `preprocessor.pkl`

Stores the fitted preprocessing pipeline.

### `ann_model.h5`

Stores the baseline ANN model.

### `tuned_ann_model.h5`

Stores the best model obtained from hyperparameter tuning.

---

# ▶️ How to Run

Make sure the project structure is correct:

```text
customer-churn-prediction/
│
├── data/
│   └── Churn_Modelling.csv
│
├── artifacts/
│
├── main.py
│
└── README.md
```

Run the Python script:

```bash
python main.py
```

The script will display:

1. Dataset information
2. Missing values
3. Duplicate rows
4. Target distribution
5. Training/testing shapes
6. Preprocessed data shapes
7. ANN architecture
8. Training progress
9. Baseline evaluation metrics
10. Best hyperparameters
11. Model artifact locations

---

# ⚠️ Important Implementation Notes

## Grid Search Toggle

The script contains:

```python
RUN_GRID_SEARCH = True
```

When set to `True`, hyperparameter tuning is performed.

When set to `False`, the grid search section is skipped.

However, the current implementation later accesses `grid_search_result` outside the `if RUN_GRID_SEARCH:` block.

Therefore, if grid search is disabled, the code should be adjusted to avoid referencing `grid_search_result`.

---

## Metrics Artifact

The code defines:

```python
METRICS_PATH = ARTIFACTS_DIR / "metrics.pkl"
```

but the current implementation does not save the calculated metrics to this file.

A metrics dictionary can be saved using:

```python
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc
}

with open(METRICS_PATH, "wb") as f:
    pickle.dump(metrics, f)
```

---

## Tuned Model Evaluation

The current code evaluates the baseline model before performing hyperparameter tuning.

The tuned model is saved after grid search, but its test-set metrics are not explicitly calculated.

For a complete evaluation workflow, the tuned model should also be evaluated on `X_test_processed` and `y_test`.

---

# 🔮 Future Improvements

The project can be extended with the following features:

* Add a separate prediction/inference script.
* Add confusion matrix visualization.
* Add ROC curve visualization.
* Add Precision-Recall curve.
* Plot training and validation accuracy.
* Plot training and validation loss.
* Save evaluation metrics to `metrics.pkl`.
* Evaluate the tuned model on the test dataset.
* Add model versioning.
* Add unit tests.
* Add logging.
* Add an API using FastAPI or Flask.
* Create a Streamlit web application.
* Add Docker support.
* Add CI/CD using GitHub Actions.
* Deploy the model to a cloud platform.

---

# 📌 Key Learning Outcomes

This project demonstrates how to:

* Load and validate a real-world dataset.
* Perform exploratory data inspection.
* Handle numerical and categorical features.
* Build a preprocessing pipeline using Scikit-learn.
* Scale numerical features.
* Encode categorical variables.
* Build an ANN using TensorFlow/Keras.
* Use ReLU and sigmoid activation functions.
* Train a neural network.
* Apply early stopping.
* Evaluate binary classification models.
* Calculate accuracy, precision, recall, F1, and ROC AUC.
* Integrate Keras with Scikit-learn using SciKeras.
* Perform hyperparameter tuning using `GridSearchCV`.
* Save trained models and preprocessing artifacts.

---

# 👨‍💻 Author

**Your Name**

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

# 📄 License

This project is intended for educational and demonstration purposes.

Add an appropriate open-source license such as **MIT License** if you plan to distribute the project publicly.
