# Breast Cancer Prediction

A beginner-friendly machine learning project that classifies breast tumors as **malignant** or **benign** using the Wisconsin Diagnostic Breast Cancer dataset included with scikit-learn.

> This project is for learning and demonstration only. It is not a medical diagnostic tool.

## Project Goal

The notebook demonstrates the complete basic workflow of a supervised classification project:

1. Load and inspect the dataset.
2. Organize the data with pandas.
3. Separate the input features from the target label.
4. Split the data into training and test sets.
5. Train a Logistic Regression classifier.
6. Generate predictions on unseen test data.
7. Evaluate performance using accuracy, a confusion matrix, precision, recall, and F1-score.

## Dataset

The project uses `load_breast_cancer()` from scikit-learn.

- **Samples:** 569
- **Input features:** 30 numerical measurements
- **Target labels:**
  - `0` = malignant
  - `1` = benign
- **Class distribution:**
  - 212 malignant samples
  - 357 benign samples

Examples of the measured features include tumor radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

## Model

The current notebook uses **Logistic Regression**, a supervised classification algorithm that estimates the probability of each class and converts that probability into a final class prediction.

The data is divided into:

- **80% training data**
- **20% test data**
- `random_state=42` for a reproducible split

## Current Results

The saved notebook produced the following test results:

- **Accuracy:** 95.6%
- **Test samples:** 114
- **Confusion matrix:**

```text
[[39, 4],
 [ 1, 70]]
```

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Malignant (0) | 0.97 | 0.91 | 0.94 | 43 |
| Benign (1) | 0.95 | 0.99 | 0.97 | 71 |

These results describe performance on this specific train/test split and should not be interpreted as clinical validation.

## Known Limitation

The current notebook produces a `ConvergenceWarning` because Logistic Regression reaches its default iteration limit before fully converging. The input features also have very different numerical scales.

A planned improvement is to use `StandardScaler` in a machine learning pipeline and increase `max_iter` if needed. Scaling must be fitted on the training data only to avoid data leakage.

The split can also be improved by passing `stratify=y` directly to `train_test_split()` so both classes maintain similar proportions in the training and test sets.

## Requirements

- Python 3
- NumPy
- pandas
- scikit-learn
- Jupyter Notebook or Google Colab

Install the main dependencies with:

```bash
pip install numpy pandas scikit-learn notebook
```

## How to Run

1. Clone or download this repository.
2. Open `Breast_Cancer.ipynb` in Jupyter Notebook or Google Colab.
3. Run the notebook from top to bottom.
4. Review the predictions and evaluation metrics.

## Repository Contents

- `Breast_Cancer.ipynb` — data preparation, model training, prediction, and evaluation.
- `README.md` — project overview and usage instructions.

## Next Steps

- Add `StandardScaler` using a scikit-learn pipeline.
- Use a stratified train/test split.
- Compare Logistic Regression with other classifiers.
- Add cross-validation.
- Plot the confusion matrix and other evaluation results.
- Improve notebook organization with separate explanation and code cells.
