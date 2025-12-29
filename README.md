# Multi-Output Classification Fitness Recommendations

## Project Overview

This project demonstrates **multi-output classification** for personalized fitness recommendations using three different machine learning paradigms:

* Classical machine learning with **scikit-learn**
* Deep learning with **PyTorch**
* Deep learning with **TensorFlow (Keras)**

Given a small set of categorical user attributes, the models jointly predict:

1. **Exercise Schedule**
2. **Meal Plan**

The repository is designed as a **comparative and educational reference**, showing how the *same problem* can be solved across frameworks while preserving identical data preprocessing and modeling logic.

---

## Dataset Description

The dataset (`GYM.csv`) is a structured, categorical fitness dataset with **80,000 rows**.

### Input Features

* `Gender`
* `Goal`
* `BMI Category`

### Target Labels

* `Exercise Schedule`
* `Meal Plan`

Each row represents a deterministic mapping from the three input features to a specific exercise schedule and meal plan.

### Deterministic Nature of the Data

The dataset is **highly deterministic**:

* Each unique `(Gender, Goal, BMI Category)` combination maps to a fixed output pair.
* There is little to no ambiguity or overlap between classes.

As a consequence:

* Most models converge extremely fast.
* Near-perfect or perfect classification metrics are expected.
* High accuracy reflects **memorization**, not real-world generalization.

This behavior is intentional and useful for studying **multi-output learning mechanics** rather than model robustness.

---

## Problem Type

This is a **multi-output, multi-class classification** problem:

* Both targets are categorical
* Each target has multiple discrete classes
* Outputs must be predicted **simultaneously**

The problem is *not* regression and *not* multilabel classification.

---

## Data Preprocessing Pipeline

All notebooks follow the same preprocessing strategy to prevent data leakage.

### 1. Train/Test Split

The dataset is split **before encoding**:

* `train_test_split(test_size=0.2, random_state=42)`

This ensures test data remains unseen during encoder fitting.

### 2. Feature Encoding

* `OneHotEncoder(sparse_output=False, handle_unknown="ignore")`
* Encoder is **fit only on training data**
* Both train and test sets are transformed afterward

Dense output is intentionally used to simplify neural network inputs.

### 3. Label Encoding

Each target is encoded independently:

* `LabelEncoder` for `Exercise Schedule`
* `LabelEncoder` for `Meal Plan`

This preserves independent class spaces for each output head.

---

## scikit-learn Implementation

Notebook: `SkLearn.ipynb`

### Models Evaluated

* K-Nearest Neighbors
* Logistic Regression (`MultiOutputClassifier`)
* Decision Tree
* Random Forest
* Gradient Boosting
* Naive Bayes (`MultiOutputClassifier`)
* XGBoost (`XGBClassifier` wrapped with `MultiOutputClassifier`)

Each model is evaluated using `classification_report` for both outputs.

### Results

All models achieve **perfect precision, recall, and F1-score**, which is expected due to the deterministic structure of the dataset.

---

## PyTorch Implementation

Notebook: `PyTorch_Fitness.ipynb`

### Model Architecture

* Shared feedforward trunk:

  * Dense(128) → ReLU
  * Dense(64) → ReLU
* Two independent output heads:

  * Exercise head (logits)
  * Meal plan head (logits)

### Training

* Loss function: `CrossEntropyLoss`
* Combined loss:

```
loss = loss_exercise + 2.0 * loss_meal
```

* Optimizer: Adam
* Learning rate: 0.01

### Observations

* Extremely fast convergence
* Loss reaches zero after few epochs
* Perfect evaluation metrics on test set

---

## TensorFlow (Keras) Implementation

Notebook: `TensorFlow_Fitness.ipynb`

### Model Architecture

* Functional API
* Shared dense layers
* Two output layers producing logits

```python
exercise_output = layers.Dense(
    n_exercise, activation=None, name="exercise"
)(shared)

meal_output = layers.Dense(
    n_meal, activation=None, name="meal"
)(shared)
```

### Training

* Separate loss tracked per output
* Same logical structure as PyTorch model


Convergence is immediate and results are identical.

---

## Key Takeaways

* Multi-output classification can be implemented consistently across frameworks
* Deterministic datasets lead to deceptively perfect metrics
* Framework choice affects training speed and verbosity, not outcomes
* Shared-trunk / multi-head architectures map naturally to this problem

---

## Repository Structure

```
├── PyTorch_Fitness.ipynb
├── TensorFlow_Fitness.ipynb
├── SkLearn.ipynb
├── GYM.csv
├── README.md
├── LICENSE
```

---

## Limitations

* Dataset lacks noise and ambiguity
* No assessment of real-world generalization
* Models primarily memorize categorical mappings

This repository is best viewed as an **architectural and educational reference**, not a production-ready fitness recommender.

---

## License

This project is licensed under the **MIT License**.

You are free to use, modify, and redistribute this software under the terms of the MIT License.
