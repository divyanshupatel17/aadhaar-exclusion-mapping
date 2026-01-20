"""
Machine Learning Model Utilities
Train and evaluate exclusion risk prediction model
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                               roc_auc_score, accuracy_score, precision_score, 
                               recall_score, f1_score)
from typing import Tuple, Dict


def prepare_features(df: pd.DataFrame, 
                      feature_cols: list = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and target variable for modeling.
    
    Args:
        df: Master district DataFrame
        feature_cols: List of feature column names (if None, uses default set)
        
    Returns:
        Tuple of (X features, y target)
    """
    if feature_cols is None:
        feature_cols = [
            'total_enrollments',
            'age_0_5',
            'age_5_17',
            'age_18_greater',
            'child_enrollment_rate',
            'demo_update_count',
            'bio_update_count',
            'demo_update_intensity',
            'bio_update_intensity',
            'pincode_count'
        ]
    
    X = df[feature_cols].copy()
    y = df['is_high_risk'].copy()
    
    # Fill missing values
    X = X.fillna(X.median())
    
    return X, y


def train_exclusion_model(X: pd.DataFrame, 
                           y: pd.Series,
                           test_size: float = 0.2,
                           random_state: int = 42) -> Dict:
    """
    Train Gradient Boosting model for exclusion risk prediction.
    
    Args:
        X: Feature matrix
        y: Target variable (is_high_risk)
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing model, scaler, metrics, and split data
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=random_state,
        verbose=0
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    metrics['cv_mean'] = cv_scores.mean()
    metrics['cv_std'] = cv_scores.std()
    
    return {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'feature_cols': list(X.columns)
    }


def get_feature_importance(model, feature_cols: list) -> pd.DataFrame:
    """
    Extract feature importance from trained model.
    
    Args:
        model: Trained GradientBoostingClassifier
        feature_cols: List of feature names
        
    Returns:
        DataFrame with features and importance scores
    """
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df


def save_model(model, scaler, filepath_model: str, filepath_scaler: str):
    """
    Save trained model and scaler to disk.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        filepath_model: Path to save model (.pkl)
        filepath_scaler: Path to save scaler (.pkl)
    """
    joblib.dump(model, filepath_model)
    joblib.dump(scaler, filepath_scaler)
    print(f"Model saved: {filepath_model}")
    print(f"Scaler saved: {filepath_scaler}")


def load_model(filepath_model: str, filepath_scaler: str) -> Tuple:
    """
    Load trained model and scaler from disk.
    
    Args:
        filepath_model: Path to model file
        filepath_scaler: Path to scaler file
        
    Returns:
        Tuple of (model, scaler)
    """
    model = joblib.load(filepath_model)
    scaler = joblib.load(filepath_scaler)
    print(f"Model loaded: {filepath_model}")
    print(f"Scaler loaded: {filepath_scaler}")
    return model, scaler


def predict_all_districts(df: pd.DataFrame, 
                           model, 
                           scaler, 
                           feature_cols: list) -> pd.DataFrame:
    """
    Generate predictions for all districts in master DataFrame.
    
    Args:
        df: Master district DataFrame
        model: Trained model
        scaler: Fitted scaler
        feature_cols: List of feature names used in training
        
    Returns:
        DataFrame with prediction columns added
    """
    df = df.copy()
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    X_scaled = scaler.transform(X)
    
    df['predicted_risk_probability'] = model.predict_proba(X_scaled)[:, 1]
    df['predicted_high_risk'] = model.predict(X_scaled)
    
    return df
