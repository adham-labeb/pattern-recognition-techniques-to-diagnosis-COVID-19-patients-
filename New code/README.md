# Classification Pipeline Explanation

This document explains the steps performed in `classification.py` and details the machine learning models, normalization scaling techniques, and data splitting methods utilized.

Note: this code runs on a small subset of the data, so the results are different from the results in the main README file. I do this so I can test if the code works or not. If the entire dataset is used, the result will be the same as in the main README file

---

## 1. Data Input and Preprocessing

```python
for file in glob.glob('Normal/*.png'):
    images = cv2.imread(file,0) # Reads the image in grayscale (0 parameter)
    Resize_image = cv2.resize(images, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    Reshape_image=Resize_image.reshape(1,256*256)
    X.append(Reshape_image)
    y.append(0) # 0 for Normal
```

**What it does:**
1. **Loading Images:** It reads `.png` images from the `Normal` and `COVID` directories using OpenCV (`cv2`). The images are read as grayscale arrays.
2. **Resizing:** The images are resized to a fixed uniform dimension of $256 \times 256$ pixels to ensure that every image provides the exact same number of features (pixels) to the models.
3. **Reshaping (Flattening):** It flattens the 2D image array of $256 \times 256$ into a 1D array (vector) of $65,536$ elements. This is because traditional machine learning classifiers require 1D arrays of features per sample.
4. **Target Labels (`y`):** It assigns a label of `0` to Normal images and `1` to COVID images.

Once all images are loaded, `X` and `y` are converted to NumPy arrays for efficient numerical operations.

---

## 2. Decorators and Metric Tracking

```python
def timer_decorator(func):
# ...
```
**What it does:**
A decorator wraps a function to modify its behavior without changing its source code. Here, `@timer_decorator` calculates the exact time the enclosed function takes to execute. We use it to record the **Train Time** (how long it takes a model to learn) and **Test Time** (how long it takes a model to make predictions) explicitly during the `train_test_split` phase.

---

## 3. Data Normalization Techniques

Features (pixel intensities) natively range between 0 and 255. Normalization adjusts these values to identical numeric scales so models don't weigh larger numbers more heavily.

### A. StandardScaler (Z-score normalization)
```python
StandardScaler()
```
* **How it works:** It shifts the distribution of each feature to have a mean ($\mu$) of $0$ and a standard deviation ($\sigma$) of $1$. The formula is $z = \frac{x - \mu}{\sigma}$.
* **What it does to the data:** Pixel values are centered around zero. Some values will become negative, and most will fall between -3 and 3. This works well for algorithms that assume data is normally distributed (like Logistic Regression or GaussianNB).

### B. MinMaxScaler
```python
MinMaxScaler(feature_range=(0, 1))
```
* **How it works:** It scales all features to lie entirely within a fixed range, in this case, between `0` and `1`. The formula is $X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$.
* **What it does to the data:** The darkest pixel (0) remains `0.0`, and the brightest pixel (255) becomes `1.0`. Relationships between pixels are preserved proportionally. It is highly effective for Neural Networks and distance-based algorithms since it bounds the parameter space tightly.

---

## 4. Classifiers (Machine Learning Models)

### A. Logistic Regression (`LR`)
* **How it works:** Despite its name, this is a linear classification algorithm. It calculates a weighted sum of the input features and passes it through a Sigmoid (logistic) function to map the prediction to a probability between 0 and 1.
* **Characteristics:** Very fast to train and test. Serves as an excellent baseline model.

### B. Random Forest Classifier (`RF`)
* **How it works:** An ensemble learning method. It creates a "forest" of many Decision Trees during training. Each tree is built using a random subset of the data and a random subset of the features. The final prediction is made by taking a majority vote of all the individual trees.
* **Characteristics:** Highly accurate and resistant to overfitting compared to single decision trees. Does not strictly require normalized data, but we normalize here to keep the pipeline consistent.

### C. Gaussian Naive Bayes (`NB`)
* **How it works:** A probabilistic classifier based on applying Bayes' theorem with the "naive" assumption of conditional independence between every pair of features (i.e., it assumes pixel 1 is completely independent of pixel 2). "Gaussian" means it assumes the continuous pixel values follow a normal (Gaussian) distribution.
* **Characteristics:** Extremely fast training time and handles high-dimensional data well.

### D. Multi-Layer Perceptron (`MLPClassifier`)
* **How it works:** An Artificial Neural Network (Feedforward Neural Network). 
  * `hidden_layer_sizes=(8, 3)` means it has an input layer (65,536 neurons for pixels), passes through two hidden layers (one with 8 neurons, then one with 3 neurons), and finally an output layer.
  * `solver='adam'` is the optimization algorithm used to aggressively update the network weights iteratively based on training data.
* **Characteristics:** Can model complex, non-linear relationships. Highly dependent on normalized data (which is why `MinMaxScaler` often pairs well with it).

---

## 5. Splitting and Evaluation Techniques

To know how well a model performs on unseen data, we evaluate it. We evaluate 3 metrics:
* **Accuracy:** Total percentage of correct predictions.
* **Recall:** Out of all actual COVID cases, how many did the model correctly find? (Crucial for medical diagnosis).
* **Precision:** Out of all times the model claimed it found COVID, how many were actually COVID?

We calculate these metrics using two different splitting strategies:

### A. Train/Test Split (`train_test_split`)
```python
train_test_split(X, y, test_size=0.2, random_state=1)
```
* **How it works:** It shuffles the dataset and randomly slices it into two distinct pieces. Here, 80% is used for **Training** the model, and the remaining 20% is held out strictly for **Testing**. `random_state=1` ensures the split is identical every time you run the script so results are reproducible.
* **Pros:** Fast and simple. Predicts performance on unseen data well.

### B. Cross-Validation (`cross_val_score` / `cross_validate` with `cv=3`)
* **How it works:** Instead of relying on a single random split, Cross-Validation splits the data into $K=3$ equal "folds". 
  * Iteration 1: Train on Fold 2+3, Test on Fold 1.
  * Iteration 2: Train on Fold 1+3, Test on Fold 2.
  * Iteration 3: Train on Fold 1+2, Test on Fold 3.
  * Finally, it averages the metrics from all 3 iterations.
* **Pros:** Provides a more reliable, robust estimate of model performance because every single data point gets to be in a test set exactly once. Prevents "getting lucky/unlucky" with a single 80/20 train/test split. It takes exactly 3 times longer to run than a standard train/test split.

---

## 6. Execution Loop

1. The script initializes an empty `results` list.
2. It loops through both normalization methods (`StandardScaler`, `MinMaxScaler`).
3. For each normalized dataset, it loops through the 4 classifiers.
4. For each classifier, it runs the `train_test_split` logic, computes metrics + times, and appends a row to `results`.
5. It then runs the `cross_validation` logic on the same classifier, computes average metrics + times, and appends the second row to `results`.
6. Lastly, `results` is converted into a Pandas DataFrame and exported cleanly to `classification_results.csv`.
