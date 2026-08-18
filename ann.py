## Import required libraries

from pathlib import Path
import pickle
import pandas as pd

import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from scikeras.wrappers import KerasClassifier  ## this is used as a wrapper for keras model to be used in sklearn pipeline

## Keras components for building the ANN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

## 2. Project configuration

BASE_DIR = Path(__file__).resolve().parent  ## Get the base directory of the project
DATA_PATH = BASE_DIR / "data" / "Churn_Modelling.csv"  ## Path to the dataset
ARTIFACTS_DIR = BASE_DIR / "artifacts"  ## Directory to save artifacts like models, scalers, encodings etc.

## Create the artifacts directory if it doesn't exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

## Files that will be generated during the process
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"  ## File to save the preprocessor
MODEL_PATH = ARTIFACTS_DIR / "ann_model.h5"  ## File to save the trained ANN model
METRICS_PATH = ARTIFACTS_DIR / "metrics.pkl"  ## File to save the evaluation metrics

## 3. load the dataset

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please ensure the dataset is placed in the correct directory.")

## Load the dataset
data = pd.read_csv(DATA_PATH)


## 4. Understand the dataset

print("\n" + "-"*50)
print("Dataset Overview: ")
print("\n" + "-"*50)

## display number of rows and columns
print(f"Number of rows: {data.shape[0]}")
print(f"Number of columns: {data.shape[1]}")
print("\n" + "-"*50)

## Display all the column names
print("\nColumn Names: ")
print(data.columns.tolist())
print("\n" + "-"*50)

## Display the first few rows of the dataset
print("\nFirst 5 rows of the dataset: ")
print(data.head())
print("\n" + "-"*50)

## display the data types of each column and non null count
print("\nData Types and Non-Null Counts: ")
print(data.info())
print("\n" + "-"*50)

## display the summary statistics of the dataset
print("\nSummary Statistics: ")
print(data.describe())
print("\n" + "-"*50)

## Check for missing values in the dataset
print("\nMissing Values in Each Column: ")
print(data.isnull().sum())
print("\n" + "-"*50)

## Check for duplicate rows in the dataset
print("\nDuplicate Rows in the Dataset: ")
print(data.duplicated().sum())
print("\n" + "-"*50)

## If duplicates are found, drop them
if data.duplicated().sum() > 0:
    data = data.drop_duplicates()
    print(f"Dropped {data.duplicated().sum()} duplicate rows.")

## 5. Validate required columns in the dataset

required_columns = {
    "RowNumber",
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited"
}

## Find columns that are missing from the dataset
missing_columns = required_columns - set(data.columns)

if missing_columns:
    raise ValueError(f"The following required columns are missing from the dataset: {missing_columns}")

## 6. Data Preprocessing

print("\n" + "-"*50)
print("\n Target distribution: ")
print(data["Exited"].value_counts())

print("\n Target percentage distribution: ")
print(data["Exited"].value_counts(normalize=True) * 100)

## 7. Remove unnecessary columns
## Row number, CustomerId, and Surname are not useful for prediction, so we drop them

X = data.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])  ## Features
y = data["Exited"]  ## Target variable

## 8. Define numerical and categorical features

numerical_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
categorical_features = ["Geography", "Gender"]

## 9. Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

## Print the shapes of the training and testing sets
print("\n" + "-"*50)
print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

## 10. Preprocessing pipeline

## Create a processing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),  ## Scale numerical features
        ("cat", OneHotEncoder(drop="first"), categorical_features)  ## One-hot encode categorical features, drop first to avoid dummy variable trap
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)  ## Fit and transform the training data
X_test_processed = preprocessor.transform(X_test)  ## Transform the testing data

print("\n" + "-"*50)
print("Preprocessing completed. Processed training and testing data shapes:")
print(f"X_train_processed: {X_train_processed.shape}, X_test_processed: {X_test_processed.shape}")

## 11. Save the preprocessor for future use

with open(PREPROCESSOR_PATH, "wb") as f:
    pickle.dump(preprocessor, f)

print(f"\nPreprocessor saved at {PREPROCESSOR_PATH}")

## 12. Build the ANN model

## Define a function to create the ANN model
def build_ann(neurons = 64, layers = 2, learning_rate = 0.001):
    model = Sequential()  ## Sequential means the model will have a linear stack of layers
    model.add(Input(shape=(X_train_processed.shape[1],)))  ## Input layer with the number of features as input shape
    for _ in range(layers):  ## Add hidden layers
        model.add(Dense(neurons, activation="relu"))  ## Hidden layer 1 with ReLU activation))
    ## output layer with 1 neuron and sigmoid activation for binary classification
    model.add(Dense(1, activation="sigmoid"))  ## Output layer with sigmoid activation for binary classification

    ## Optimizer : it controls how the model is updated based on the data it sees and its loss function. Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iteratively based on training data.
    optimizer = Adam(learning_rate=learning_rate)  ## Adam optimizer with specified learning rate

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])  ## Compile the model with binary crossentropy loss and accuracy metric

    return model

## 13. Create Baseline ANN model

print("\n" + "-"*50)
print("Creating baseline ANN model with default parameters...")
print("\n" + "-"*50)

model = build_ann()  ## Create the ANN model with default parameters

## Display model summary
print("\n" + "-"*50)
print("Model Summary: ")
model.summary()  ## Display the model architecture

early_stopping = EarlyStopping(
    monitor="val_loss", 
    patience=5, 
    restore_best_weights=True)  ## Early stopping to prevent overfitting

## 14. Train the model

print("\n" + "-"*50)
print("Training the model...")
print("\n" + "-"*50)

history = model.fit(
    X_train_processed, 
    y_train, 
    validation_split=0.2,  ## Use 20% of training data for validation
    epochs=100,  ## Train for 100 epochs
    batch_size=32,  ## Batch size of 32
    callbacks=[early_stopping],  ## Use early stopping callback
    verbose=1  ## Show progress bar
)

## 15. Evaluate the model

print("\n" + "-"*50)
print("Evaluating the model on the test set...")
print("\n" + "-"*50)

test_loss, test_accuracy = model.evaluate(
    X_test_processed, 
    y_test, 
    verbose=1)  ## Evaluate the model on the test set

## 16. Make predictions on the test set

probablities = model.predict(X_test_processed)  ## Get predicted probabilities
probablities = probablities.flatten()  ## Flatten the probabilities array

## 17. Convert probabilities to binary predictions using a threshold of 0.5
predictions = (probablities >= 0.5).astype(int)  ## Convert probabilities to binary predictions

## 18. Calculate evaluation metrics

accuracy = accuracy_score(y_test, predictions)  ## Calculate accuracy
precision = precision_score(y_test, predictions)  ## Calculate precision
recall = recall_score(y_test, predictions)  ## Calculate recall
f1 = f1_score(y_test, predictions)  ## Calculate F1 score
roc_auc = roc_auc_score(y_test, probablities)  ## Calculate ROC AUC

print("\n" + "-"*50)
print("Evaluation Metrics: ")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

## 19. Save the trained model
model.save(MODEL_PATH)  ## Save the trained model to disk
print(f"\nTrained model saved at {MODEL_PATH}")


## 20. Hyperparameter tuning using GridSearchCV

## The baseline model is build with default parameters. Now we will perform hyperparameter tuning to find the best combination of parameters for our ANN model.
## Using neurons, layers, and learning rate as hyperparameters to tune.

RUN_GRID_SEARCH = True

if RUN_GRID_SEARCH:

    print("\n" + "-" * 50)
    print("Starting hyperparameter tuning using GridSearchCV...")
    print("-" * 50)

    param_grid = {
        "neurons": [32, 64],
        "layers": [1, 2],
        "learning_rate": [0.001, 0.01],
        "batch_size": [32],
        "epochs": [50]
    }

    model_classifier = KerasClassifier(
        model=build_ann,
        neurons=64,
        layers=2,
        learning_rate=0.001,
        verbose=0
    )

    grid_search = GridSearchCV(
        estimator=model_classifier,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=1,
        verbose=1
    )

    grid_search_result = grid_search.fit(
        X_train_processed,
        y_train
    )

    best_params = grid_search_result.best_params_

    print("\n" + "-" * 50)
    print("Best Hyperparameters found:")
    print(best_params)

    best_model = grid_search_result.best_estimator_

    tuned_keras_model = best_model.model_

    tuned_model_path = ARTIFACTS_DIR / "tuned_ann_model.h5"

    tuned_keras_model.save(tuned_model_path)

    print(f"\nTuned model saved at {tuned_model_path}")
    
## 21. Get the best model from the grid search and evaluate it on the test set

best_model = grid_search_result.best_estimator_  ## Get the best model from the grid search

tuned_keras_model = best_model.model_  ## Get the underlying Keras model from the best estimator

## 22. Save the tuned model
tuned_model_path = ARTIFACTS_DIR / "tuned_ann_model.h5"  ## Path to save the tuned model
tuned_keras_model.save(tuned_model_path)  ## Save the tuned model to disk

print(f"\nTuned model saved at {tuned_model_path}")