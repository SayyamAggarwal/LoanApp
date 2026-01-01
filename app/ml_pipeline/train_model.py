"""
Model Training Script for Loan Eligibility Prediction
Uses XGBoost, LightGBM, and CatBoost with cross-validation and hyperparameter tuning
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import prepare_data

def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models and select the best one
    """
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    models = {}
    results = {}
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. XGBoost
    print("\n1. Training XGBoost...")
    # Calculate class weights for balanced learning
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_dict = dict(zip(classes, class_weights))
    sample_weights = np.array([weight_dict[y] for y in y_train])
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,  # Regularization to prevent over-reliance on single features
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        max_delta_step=1,  # Helps with imbalanced classes
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),  # Handle imbalance
        random_state=42,
        eval_metric='logloss'
    )
    
    # Ensure feature names are explicitly set for XGBoost
    # Convert to numpy and back to ensure consistent feature names
    feature_names = X_train.columns.tolist()
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Explicitly set feature names to ensure consistency
    # XGBoost should preserve feature names from DataFrame, but we'll set them explicitly
    if hasattr(xgb_model, 'get_booster'):
        xgb_model.get_booster().feature_names = feature_names
    
    # For CV, we need to recalculate sample weights for each fold
    # Use a simpler approach: use scale_pos_weight instead
    xgb_model_cv = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        max_delta_step=1,
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        random_state=42,
        eval_metric='logloss'
    )
    xgb_cv_scores = cross_val_score(xgb_model_cv, X_train, y_train, cv=cv, scoring='roc_auc')
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    models['xgb'] = xgb_model
    results['xgb'] = {
        'cv_mean': xgb_cv_scores.mean(),
        'cv_std': xgb_cv_scores.std(),
        'accuracy': accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred),
        'recall': recall_score(y_test, xgb_pred),
        'f1': f1_score(y_test, xgb_pred),
        'roc_auc': roc_auc_score(y_test, xgb_pred_proba)
    }
    
    print(f"   CV ROC-AUC: {xgb_cv_scores.mean():.4f} (+/- {xgb_cv_scores.std() * 2:.4f})")
    print(f"   Test Accuracy: {results['xgb']['accuracy']:.4f}")
    print(f"   Test ROC-AUC: {results['xgb']['roc_auc']:.4f}")
    
    # 2. LightGBM
    print("\n2. Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,  # More trees to smoother decisions
        max_depth=3,       # Shallow trees to prevent complex specific logic
        learning_rate=0.05, # Slower, more robust learning
        subsample=0.5,      # Use only half the data per tree
        colsample_bytree=0.3, # Force model to use different features (very low!)
        min_child_samples=50, # Require more data per leaf
        reg_alpha=10.0,    # Extreme L1 regularization to squash weights
        reg_lambda=10.0,   # Extreme L2 regularization
        extra_trees=True,  # Use extra randomized trees
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    
    # Ensure feature names are explicitly set for LightGBM
    feature_names = X_train.columns.tolist()
    lgb_model.fit(X_train, y_train, feature_name=feature_names)
    lgb_cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=cv, scoring='roc_auc')
    lgb_pred = lgb_model.predict(X_test)
    lgb_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
    
    models['lgb'] = lgb_model
    results['lgb'] = {
        'cv_mean': lgb_cv_scores.mean(),
        'cv_std': lgb_cv_scores.std(),
        'accuracy': accuracy_score(y_test, lgb_pred),
        'precision': precision_score(y_test, lgb_pred),
        'recall': recall_score(y_test, lgb_pred),
        'f1': f1_score(y_test, lgb_pred),
        'roc_auc': roc_auc_score(y_test, lgb_pred_proba)
    }
    
    print(f"   CV ROC-AUC: {lgb_cv_scores.mean():.4f} (+/- {lgb_cv_scores.std() * 2:.4f})")
    print(f"   Test Accuracy: {results['lgb']['accuracy']:.4f}")
    print(f"   Test ROC-AUC: {results['lgb']['roc_auc']:.4f}")
    
    # 3. CatBoost
    print("\n3. Training CatBoost...")
    cat_model = cb.CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        l2_leaf_reg=3,  # L2 regularization to prevent overfitting
        random_seed=42,
        class_weights=[1, len(y_train[y_train==0])/len(y_train[y_train==1])],  # Handle class imbalance
        verbose=False
    )
    
    # Manual cross-validation for CatBoost (to avoid sklearn compatibility issues)
    cat_cv_scores_list = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train on fold
        fold_model = cb.CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            l2_leaf_reg=3,  # L2 regularization
            random_seed=42,
            class_weights=[1, len(y_fold_train[y_fold_train==0])/len(y_fold_train[y_fold_train==1])],  # Handle imbalance
            verbose=False
        )
        fold_model.fit(X_fold_train, y_fold_train)
        
        # Predict and calculate ROC-AUC
        y_fold_pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
        fold_roc_auc = roc_auc_score(y_fold_val, y_fold_pred_proba)
        cat_cv_scores_list.append(fold_roc_auc)
    
    cat_cv_scores = np.array(cat_cv_scores_list)
    
    # Train final model on full training set
    cat_model.fit(X_train, y_train)
    cat_pred = cat_model.predict(X_test)
    cat_pred_proba = cat_model.predict_proba(X_test)[:, 1]
    
    models['cat'] = cat_model
    results['cat'] = {
        'cv_mean': cat_cv_scores.mean(),
        'cv_std': cat_cv_scores.std(),
        'accuracy': accuracy_score(y_test, cat_pred),
        'precision': precision_score(y_test, cat_pred),
        'recall': recall_score(y_test, cat_pred),
        'f1': f1_score(y_test, cat_pred),
        'roc_auc': roc_auc_score(y_test, cat_pred_proba)
    }
    
    print(f"   CV ROC-AUC: {cat_cv_scores.mean():.4f} (+/- {cat_cv_scores.std() * 2:.4f})")
    print(f"   Test Accuracy: {results['cat']['accuracy']:.4f}")
    print(f"   Test ROC-AUC: {results['cat']['roc_auc']:.4f}")
    
    # Select best model based on ROC-AUC
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    comparison_df = pd.DataFrame(results).T
    print(comparison_df[['cv_mean', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']])
    
    best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
    best_model = models[best_model_name]
    
    print(f"\nBest Model: {best_model_name.upper()}")
    print(f"ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    # Detailed classification report for best model
    print("\n" + "=" * 60)
    print(f"DETAILED REPORT - {best_model_name.upper()}")
    print("=" * 60)
    best_pred = best_model.predict(X_test)
    print(classification_report(y_test, best_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, best_pred))
    
    return best_model, best_model_name, results

def save_model(model, model_name, filepath=None):
    """
    Save the trained model
    """
    if filepath is None:
        # Get project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        filepath = os.path.join(project_root, 'app', 'models')
    
    os.makedirs(filepath, exist_ok=True)
    model_path = os.path.join(filepath, f'{model_name}_model.pkl')
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    return model_path

if __name__ == "__main__":
    # Prepare data
    print("Preparing data...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    
    # Train models
    best_model, best_model_name, results = train_models(X_train, y_train, X_test, y_test)
    
    # Save best model
    model_path = save_model(best_model, best_model_name)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

