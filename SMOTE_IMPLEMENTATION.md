# SMOTE Implementation for Data Balancing

## Overview

SMOTE (Synthetic Minority Oversampling Technique) has been integrated into the loan eligibility prediction pipeline to address the class imbalance issue in the dataset.

## Problem

The original dataset is highly imbalanced:
- **Class 0 (Not Eligible)**: 35,000 samples (77.8%)
- **Class 1 (Eligible)**: 10,000 samples (22.2%)
- **Ratio**: 3.5:1

This imbalance causes the model to be biased towards predicting "Not Eligible", leading to poor performance on the minority class.

## Solution: SMOTE

SMOTE has been implemented to balance the training data by generating synthetic samples of the minority class.

### Implementation Details

1. **Location**: `app/ml_pipeline/preprocessing.py`
2. **Function**: `prepare_data()` with `apply_smote=True` parameter
3. **When Applied**: SMOTE is applied **only to training data** after train-test split to avoid data leakage
4. **Balance Ratio**: 1:1 (equal number of samples for both classes)

### Key Features

- **Safe k_neighbors calculation**: Automatically adjusts k_neighbors based on minority class size
- **Error handling**: Falls back to original data if SMOTE fails
- **Data type preservation**: Maintains DataFrame/Series structure after resampling
- **Reproducibility**: Uses random_state for consistent results

### Usage

SMOTE is enabled by default when calling `prepare_data()`:

```python
# With SMOTE (default)
X_train, X_test, y_train, y_test, preprocessor = prepare_data(apply_smote=True)

# Without SMOTE
X_train, X_test, y_train, y_test, preprocessor = prepare_data(apply_smote=False)
```

### Expected Results

**Before SMOTE:**
- Training set: ~28,000 (Class 0) vs ~8,000 (Class 1)
- Ratio: ~3.5:1

**After SMOTE:**
- Training set: ~28,000 (Class 0) vs ~28,000 (Class 1)
- Ratio: 1:1 (balanced)

### Installation

Install the required package:

```bash
pip install imbalanced-learn>=0.11.0
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### Benefits

1. **Improved Model Performance**: Better recall for the minority class (eligible loans)
2. **Reduced Bias**: Model won't be biased towards predicting "Not Eligible"
3. **Better Generalization**: More balanced training leads to better performance on both classes
4. **Fair Predictions**: More accurate predictions for eligible applicants

### Notes

- SMOTE is applied **only to training data**, not test data
- Test data remains imbalanced to reflect real-world distribution
- The synthetic samples are generated based on k-nearest neighbors in feature space
- Original data distribution is preserved in the test set for realistic evaluation

## Next Steps

After implementing SMOTE:

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Retrain models**: `python train_model.py`
3. **Compare results**: Check if model performance improves, especially for Class 1 (Eligible)

The training script will automatically show before/after SMOTE distributions and the balanced class ratios.



