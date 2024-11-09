# CT Lung Image Classification Project
for the data set you can visit : https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

## Project Overview
This project aims to classify CT images of lungs as either normal or COVID-19 positive. The goal is to determine the best classifier, normalization technique, and data split method to accurately classify future images. For this project, we label normal cases as `0` and COVID-19 cases as `1`.

## Pattern Recognition Workflow

1. **Data Acquisition**:
   - CT images are resized to 256x256 pixels, transformed into 1D arrays, and organized into two arrays: `X` (features) and `y` (labels).
   
2. **Preprocessing**:
   - **Normalization Techniques**: Two normalization techniques are tested:
     - **Min-Max Normalization** (scales data to a range of 0-1)
     - **Z-score Normalization** (standardizes features by removing the mean and scaling to unit variance)

3. **Data Splitting**:
   - **Random Split (Holdout)**: Data is split randomly into training and testing sets (80% train, 20% test).
   - **Cross-Validation Split**: Data is split into `k=3` folds for cross-validation.

4. **Feature Extraction**:
   - Features are represented as discriminative and invariant vectors extracted from the images.

5. **Classification Models**:
   - Four classifiers are used in the study:
     - **Logistic Regression (LR)**
     - **Random Forest (RF)**
     - **Naive Bayes (NB)**
     - **Artificial Neural Network (ANN)**

6. **Performance Metrics**:
   - Accuracy, Precision, Recall, and execution time are evaluated for each model.

---

## Results Summary

### Z-score Normalization + Random Split

| Model      | Train Accuracy | Train Recall | Train Precision | Train Time (s) | Test Accuracy | Test Recall | Test Precision | Test Time (s) |
|------------|----------------|--------------|-----------------|----------------|---------------|-------------|----------------|---------------|
| Random Forest (RF) | 1.00           | 1.00         | 1.00            | 53.45          | 0.93          | 0.93         | 0.93           | 1.99          |
| Naive Bayes (NB)   | 0.76           | 0.82         | 0.76            | 7.53           | 0.75          | 0.75         | 0.82           | 2.25          |
| Artificial Neural Network (ANN) | 0.80           | 0.80         | 0.83            | 58.66          | 0.79          | 0.79         | 0.84           | 6.81          |
| Logistic Regression (LR) | 1.00           | 1.00         | 1.00            | 24.55          | 0.84          | 0.84         | 0.88           | 1.79          |

![pattern-recognition](https://github.com/adham-labeb/pattern-recognition-techniques-to-diagnosis-COVID-19-patients-/blob/main/Algorithm%20name%20%20Z-score%20normalization%20%2B%20random%20split.png)
### Z-score Normalization + Cross-Validation Split

| Model      | Accuracy | Recall  | Precision | Time (s) |
|------------|----------|---------|-----------|----------|
| Random Forest (RF) | 0.91     | 0.76    | 0.90      | 276.04   |
| Naive Bayes (NB)   | 0.74     | 0.82    | 0.52      | 403.41   |
| Artificial Neural Network (ANN) | 0.89     | 0.76    | 0.84      | 275.34   |
| Logistic Regression (LR) | 0.71     | 0.86    | 0.46      | 388.41   |

![pattern-recognition](https://github.com/adham-labeb/pattern-recognition-techniques-to-diagnosis-COVID-19-patients-/blob/main/Algorithm%20name%20%20Z-score%20normalization%20%2B%20Cross%20validation%20split.png)
![pattern-recognition](https://github.com/adham-labeb/pattern-recognition-techniques-to-diagnosis-COVID-19-patients-/blob/main/Algorithm%20name%20%20Z-score%20normalization%20%2B%20Cross%20validation%20split%202.png)
### Min-Max Normalization + Random Split

| Model      | Train Accuracy | Train Recall | Train Precision | Train Time (s) | Test Accuracy | Test Recall | Test Precision | Test Time (s) |
|------------|----------------|--------------|-----------------|----------------|---------------|-------------|----------------|---------------|
| Random Forest (RF) | 1.00           | 1.00         | 1.00            | 48.45          | 0.93          | 0.93         | 0.93           | 1.99          |
| Naive Bayes (NB)   | 0.76           | 0.82         | 0.76            | 7.53           | 0.75          | 0.75         | 0.82           | 2.25          |
| Artificial Neural Network (ANN) | 0.61           | 0.72         | 0.73            | 63.66          | 0.79          | 0.79         | 0.84           | 6.81          |
| Logistic Regression (LR) | 0.97           | 0.97         | 0.97            | 18.37          | 0.86          | 0.86         | 0.88           | 1.62          |

![pattern-recognition](https://github.com/adham-labeb/pattern-recognition-techniques-to-diagnosis-COVID-19-patients-/blob/main/Algorithm%20name%20%20min%20-%20max%20normalization%20%2B%20random%20split%20.png)
### Min-Max Normalization + Cross-Validation Split

| Model      | Accuracy | Recall  | Precision | Time (s) |
|------------|----------|---------|-----------|----------|
| Random Forest (RF) | 0.92     | 0.75    | 0.93      | 550.21   |
| Naive Bayes (NB)   | 0.84     | 0.81    | 0.85      | 428.23   |
| Artificial Neural Network (ANN) | 0.77     | 0.09    | 0.28      | 708.24   |
| Logistic Regression (LR) | 0.86     | 0.68    | 0.77      | 507.41   |

![pattern-recognition](https://github.com/adham-labeb/pattern-recognition-techniques-to-diagnosis-COVID-19-patients-/blob/main/Algorithm%20name%20%20min%20max%20normalization%20%2B%20Cross%20validation%20split.png)
![pattern-recognition](https://github.com/adham-labeb/pattern-recognition-techniques-to-diagnosis-COVID-19-patients-/blob/main/Algorithm%20name%20%20min%20max%20normalization%20%2B%20Cross%20validation%20split%202.png)
---

## Conclusion and Insights

- **Best Normalization Technique**: Z-score normalization is preferred as it handles outliers and reduces noise.
- **Best Classifier**: The Random Forest classifier achieved the highest accuracy across all tests.
- **Fastest Algorithm**: The Naive Bayes classifier was the fastest in 3 out of 4 tests.
- **Overfitting Observed**:
  - Logistic Regression caused overfitting in both the Min-Max and Z-score holdout (random) split methods.

---

## Usage Instructions

To run this code:

1. Place your images in folders labeled as "normal" (e.g., `C:\images`) and "COVID-19" (e.g., `C:\images-covid`).
2. Install the required libraries:
   ```bash
   pip install numpy opencv-python-headless scikit-learn
