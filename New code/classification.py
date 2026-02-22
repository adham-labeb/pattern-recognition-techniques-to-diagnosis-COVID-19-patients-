#### Importing the libraries ####
import pandas as pd
import numpy as np
import glob
import cv2
import time
import warnings
from statistics import mean
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import cross_validate , train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score


warnings.filterwarnings('ignore')


###########################     Data Input     ####################################
X = []
y = []

for file in glob.glob('Normal/*.png'):
    images = cv2.imread(file,0)
    Resize_image = cv2.resize(images, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    Reshape_image=Resize_image.reshape(1,256*256)
    X.append(Reshape_image)
    y.append(0)

for file in glob.glob('COVID/*.png'):
    images = cv2.imread(file,0)
    Resize_image = cv2.resize(images, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    Reshape_image=Resize_image.reshape(1,256*256)
    X.append(Reshape_image)
    y.append(1)

#####################################################################################

X = np.array(X)
X =X.reshape(X.shape[0],256*256)

y = np.array(y)
y = y.reshape(y.shape[0],1)



###########################     Evaluation Pipeline     ####################################


def timer_decorator(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        exec_time = end_time - start_time
        return result, exec_time
    return wrapper

@timer_decorator
def train_model(model, X_train, y_train):
    """Trains the model and returns fitted model."""
    model.fit(X_train, y_train.ravel())
    return model

@timer_decorator
def test_model(model, X_test):
    """Generates predictions."""
    return model.predict(X_test)

def evaluate_train_test_split(model, X, y):
    """Evaluates using train_test_split."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    fitted_model, train_time = train_model(model, X_train, y_train)
    y_pred, test_time = test_model(fitted_model, X_test)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    
    return acc, rec, prec, train_time, test_time

def evaluate_cross_val(model, X, y):
    """Evaluates using 3-fold cross validation."""
    scoring = ['accuracy', 'recall', 'precision']
    # cv_results_ returns 'fit_time' and 'score_time' inherently
    scores = cross_validate(model, X, y.ravel(), cv=3, scoring=scoring, return_train_score=False)
    
    acc = np.mean(scores['test_accuracy'])
    rec = np.mean(scores['test_recall'])
    prec = np.mean(scores['test_precision'])
    train_time = np.mean(scores['fit_time'])
    test_time = np.mean(scores['score_time'])
    
    return acc, rec, prec, train_time, test_time

# Define models, normalizations, and splitting strategies
models = {
    'LogisticRegression': LR(),
    'RandomForestClassifier': RFC(),
    'GaussianNB': GNB(),
    'MLPClassifier': MLPC(solver='adam', hidden_layer_sizes=(8, 3))
}

normalizations = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(feature_range=(0, 1))
}

results = []

for norm_name, scaler in normalizations.items():
    # Apply normalization
    X_scaled = scaler.fit_transform(X)
    
    for model_name, model in models.items():
        
        # 1. Train-Test Split
        acc, rec, prec, train_time, test_time = evaluate_train_test_split(model, X_scaled, y)
        results.append({
            'Model Name': f"{model_name} + {norm_name} + train_test_split",
            'Accuracy': acc,
            'Recall': rec,
            'Precision': prec,
            'Train Time': train_time,
            'Test Time': test_time
        })
        
        # 2. Cross Validation (cv=3)
        acc, rec, prec, train_time, test_time = evaluate_cross_val(model, X_scaled, y)
        results.append({
            'Model Name': f"{model_name} + {norm_name} + cross_val_score",
            'Accuracy': acc,
            'Recall': rec,
            'Precision': prec,
            'Train Time': train_time,
            'Test Time': test_time
        })

# Create DataFrame and export to CSV
results_df = pd.DataFrame(results)
print(results_df)

csv_filename = "classification_results.csv"
results_df.to_csv(csv_filename, index=False)
print(f"\nSuccessfully exported results to {csv_filename}!")
