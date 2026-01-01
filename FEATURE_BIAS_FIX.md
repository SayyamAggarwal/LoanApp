# Feature Bias Fix: Ensuring Fair Feature Representation

## Problem Identified

The model was showing excessive bias towards the `person_home_ownership` feature, particularly:
- When users selected "OWN", the model always predicted "Not Eligible"
- The feature had disproportionate impact on predictions
- Other features were not getting fair representation

### Root Causes

1. **Label Encoding Bias**: Using `LabelEncoder` created ordinal encoding (0, 1, 2, 3) which introduced artificial ordering
   - OWN might be encoded as 2, MORTGAGE as 1, RENT as 3
   - This created false relationships between categories

2. **Data Imbalance in Feature**: The data showed:
   - OWN: 92.48% Not Eligible (only 7.52% Eligible)
   - MORTGAGE: 88.40% Not Eligible (only 11.60% Eligible)
   - RENT: 67.60% Not Eligible (32.40% Eligible)
   - The model learned this strong correlation and over-relied on it

3. **Lack of Regularization**: Models could over-rely on single features without constraints

## Solutions Implemented

### 1. OneHot Encoding (Primary Fix)

**Changed from**: `LabelEncoder` (ordinal encoding)
**Changed to**: `OneHotEncoder` (binary encoding)

**Benefits**:
- Each category gets its own binary feature
- No artificial ordering between categories
- Fair representation: each category is treated independently
- Example: Instead of `person_home_ownership = 2`, we get:
  - `person_home_ownership_MORTGAGE = 0`
  - `person_home_ownership_OWN = 1`
  - `person_home_ownership_OTHER = 0`
  - `person_home_ownership_RENT = 0` (dropped as reference)

**Implementation**:
- Applied to all categorical features: `person_education`, `person_home_ownership`, `loan_intent`, `previous_loan_defaults_on_file`
- Uses `drop='first'` to avoid multicollinearity
- Handles unknown categories gracefully

### 2. Regularization Added

Added regularization parameters to prevent over-reliance on single features:

**XGBoost**:
- `min_child_weight=3`: Prevents overfitting on small samples
- `reg_alpha=0.1`: L1 regularization (feature selection)
- `reg_lambda=1.0`: L2 regularization (feature shrinkage)
- `max_delta_step=1`: Helps with imbalanced classes
- `scale_pos_weight`: Handles class imbalance

**LightGBM**:
- `min_child_samples=20`: Regularization
- `reg_alpha=0.1`: L1 regularization
- `reg_lambda=1.0`: L2 regularization
- `class_weight='balanced'`: Handles class imbalance

**CatBoost**:
- `l2_leaf_reg=3`: L2 regularization
- `class_weights`: Handles class imbalance

### 3. Class Weights

Added class weights to ensure balanced learning:
- Prevents model from being biased towards majority class
- Ensures both "Eligible" and "Not Eligible" are learned equally

## Expected Improvements

1. **Fair Feature Representation**: All features now contribute more evenly to predictions
2. **No Single-Feature Dominance**: Regularization prevents over-reliance on any one feature
3. **Better Predictions for OWN**: Users with "OWN" status will get fairer predictions based on all features
4. **More Balanced SHAP Values**: SHAP explanations will show more distributed feature importance

## Technical Details

### OneHot Encoding Process

For `person_home_ownership` with values: [RENT, MORTGAGE, OWN, OTHER]

**Before (LabelEncoder)**:
```
person_home_ownership: 0, 1, 2, 3
```

**After (OneHotEncoder with drop='first')**:
```
person_home_ownership_MORTGAGE: 0 or 1
person_home_ownership_OWN: 0 or 1
person_home_ownership_OTHER: 0 or 1
(RENT is the reference, dropped)
```

### Feature Count Change

**Before**: ~12 features
**After**: ~20+ features (due to one-hot expansion)

This gives the model more granular information and prevents ordinal bias.

## Migration Notes

- Old models trained with LabelEncoder will need to be retrained
- The preprocessor now saves `onehot_encoders` in addition to `label_encoders`
- Backward compatibility is maintained for loading old preprocessors

## Next Steps

1. **Retrain Models**: Run `python train_model.py` to train with new encoding
2. **Verify SHAP Values**: Check that feature importance is more distributed
3. **Test Predictions**: Verify that "OWN" status no longer always predicts "Not Eligible"
4. **Monitor Performance**: Ensure model accuracy is maintained or improved

## Files Modified

1. `app/ml_pipeline/preprocessing.py`: Changed to OneHotEncoder
2. `app/ml_pipeline/train_model.py`: Added regularization and class weights



