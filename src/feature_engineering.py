"""
Feature Engineering Utilities
Create derived features for exclusion risk modeling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def add_enrolment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to enrolment DataFrame.
    
    Args:
        df: Enrolment DataFrame with age columns
        
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # Total enrollments
    df['total_enrollments'] = (
        df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    )
    
    # Age distribution ratios (avoid division by zero)
    df['child_0_5_ratio'] = df['age_0_5'] / (df['total_enrollments'] + 1)
    df['child_5_17_ratio'] = df['age_5_17'] / (df['total_enrollments'] + 1)
    df['adult_ratio'] = df['age_18_greater'] / (df['total_enrollments'] + 1)
    
    # Temporal features
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek
    
    return df


def aggregate_update_frequency(df: pd.DataFrame, 
                                 group_cols: list = ['state', 'district', 'pincode'],
                                 count_col: str = 'update_count') -> pd.DataFrame:
    """
    Aggregate update frequency by location.
    
    Args:
        df: DataFrame with location columns
        group_cols: Columns to group by
        count_col: Name for the count column
        
    Returns:
        Aggregated DataFrame with update counts
    """
    frequency = df.groupby(group_cols).size().reset_index(name=count_col)
    return frequency


def create_district_master(df_enrol: pd.DataFrame,
                            df_demo: pd.DataFrame,
                            df_bio: pd.DataFrame) -> pd.DataFrame:
    """
    Create master district-level dataset by aggregating all data sources.
    
    Args:
        df_enrol: Enrolment DataFrame (with features)
        df_demo: Demographic update DataFrame
        df_bio: Biometric update DataFrame
        
    Returns:
        Master district DataFrame with all metrics
    """
    # Aggregate enrolment by district
    enrol_district = df_enrol.groupby(['state', 'district']).agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum',
        'total_enrollments': 'sum',
        'pincode': 'nunique'
    }).reset_index()
    enrol_district.rename(columns={'pincode': 'pincode_count'}, inplace=True)
    
    # Aggregate demographic updates
    demo_frequency = aggregate_update_frequency(df_demo, count_col='demo_update_count')
    district_demo = demo_frequency.groupby(['state', 'district']).agg({
        'demo_update_count': 'sum'
    }).reset_index()
    
    # Aggregate biometric updates
    bio_frequency = aggregate_update_frequency(df_bio, count_col='bio_update_count')
    district_bio = bio_frequency.groupby(['state', 'district']).agg({
        'bio_update_count': 'sum'
    }).reset_index()
    
    # Merge all
    df_master = enrol_district.merge(district_demo, on=['state', 'district'], how='left')
    df_master = df_master.merge(district_bio, on=['state', 'district'], how='left')
    
    # Fill NaN with 0
    df_master['demo_update_count'] = df_master['demo_update_count'].fillna(0)
    df_master['bio_update_count'] = df_master['bio_update_count'].fillna(0)
    
    # Calculate intensities
    df_master['demo_update_intensity'] = (
        df_master['demo_update_count'] / (df_master['total_enrollments'] + 1)
    )
    df_master['bio_update_intensity'] = (
        df_master['bio_update_count'] / (df_master['total_enrollments'] + 1)
    )
    
    # Child enrollment rate
    df_master['child_enrollment_rate'] = (
        df_master['age_0_5'] / (df_master['total_enrollments'] + 1)
    )
    
    return df_master


def calculate_exclusion_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate composite exclusion risk score using multiple indicators.
    
    Args:
        df: Master district DataFrame
        
    Returns:
        DataFrame with risk score columns added
    """
    df = df.copy()
    scaler = MinMaxScaler()
    
    # Individual risk components
    df['enroll_risk'] = 1 - scaler.fit_transform(df[['total_enrollments']])
    df['child_risk'] = 1 - scaler.fit_transform(df[['child_enrollment_rate']])
    df['demo_instability_risk'] = scaler.fit_transform(df[['demo_update_intensity']])
    df['bio_failure_risk'] = scaler.fit_transform(df[['bio_update_intensity']])
    
    # Composite score (weighted)
    df['exclusion_risk_score'] = (
        0.35 * df['enroll_risk'] +
        0.25 * df['child_risk'] +
        0.20 * df['demo_instability_risk'] +
        0.20 * df['bio_failure_risk']
    )
    
    # Binary classification
    df['is_high_risk'] = (df['exclusion_risk_score'] > 0.50).astype(int)
    
    return df


def calculate_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate intervention priority score (0-100).
    
    Args:
        df: District DataFrame with risk features and ML predictions
        
    Returns:
        DataFrame with priority_score column
    """
    df = df.copy()
    
    # Calculate gaps
    df['child_gap'] = df['child_enrollment_rate'].max() - df['child_enrollment_rate']
    df['enrollment_gap'] = df['total_enrollments'].max() - df['total_enrollments']
    
    # Normalize gaps
    df['child_gap_norm'] = (
        (df['child_gap'] - df['child_gap'].min()) / 
        (df['child_gap'].max() - df['child_gap'].min())
    )
    df['enrollment_gap_norm'] = (
        (df['enrollment_gap'] - df['enrollment_gap'].min()) / 
        (df['enrollment_gap'].max() - df['enrollment_gap'].min())
    )
    
    # Priority score (assumes 'predicted_risk_probability' exists from ML model)
    if 'predicted_risk_probability' in df.columns:
        df['priority_score'] = (
            40 * df['predicted_risk_probability'] +
            30 * df['child_gap_norm'] +
            20 * df['demo_instability_risk'] +
            10 * df['bio_failure_risk']
        ) * 100
    else:
        # Fallback if no ML prediction available
        df['priority_score'] = (
            40 * df['exclusion_risk_score'] +
            30 * df['child_gap_norm'] +
            20 * df['demo_instability_risk'] +
            10 * df['bio_failure_risk']
        ) * 100
    
    df['priority_score'] = df['priority_score'].clip(0, 100)
    
    return df
