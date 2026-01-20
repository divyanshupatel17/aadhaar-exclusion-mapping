"""
Data Loading Utilities
Load and concatenate Aadhaar datasets (Enrolment, Demographic, Biometric)
"""

import pandas as pd
import glob
import os
from typing import List, Tuple


def load_enrolment_data(data_dir: str = '../dataset') -> pd.DataFrame:
    """
    Load and concatenate all enrolment CSV files.
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        Combined DataFrame with all enrolment records
    """
    files = glob.glob(os.path.join(data_dir, 'api_data_aadhar_enrolment',
                                    'api_data_aadhar_enrolment', '*.csv'))
    
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    
    return df


def load_demographic_data(data_dir: str = '../dataset') -> pd.DataFrame:
    """
    Load and concatenate all demographic update CSV files.
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        Combined DataFrame with all demographic update records
    """
    files = glob.glob(os.path.join(data_dir, 'api_data_aadhar_demographic',
                                    'api_data_aadhar_demographic', '*.csv'))
    
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    
    return df


def load_biometric_data(data_dir: str = '../dataset') -> pd.DataFrame:
    """
    Load and concatenate all biometric update CSV files.
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        Combined DataFrame with all biometric update records
    """
    files = glob.glob(os.path.join(data_dir, 'api_data_aadhar_biometric',
                                    'api_data_aadhar_biometric', '*.csv'))
    
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    
    return df


def load_all_datasets(data_dir: str = '../dataset') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three Aadhaar datasets.
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        Tuple of (enrolment_df, demographic_df, biometric_df)
    """
    print("Loading enrolment data...")
    df_enrol = load_enrolment_data(data_dir)
    
    print("Loading demographic data...")
    df_demo = load_demographic_data(data_dir)
    
    print("Loading biometric data...")
    df_bio = load_biometric_data(data_dir)
    
    print(f"Loaded {len(df_enrol):,} enrolment records")
    print(f"Loaded {len(df_demo):,} demographic records")
    print(f"Loaded {len(df_bio):,} biometric records")
    
    return df_enrol, df_demo, df_bio


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize text fields (state, district, pincode).
    
    Args:
        df: DataFrame with state/district/pincode columns
        
    Returns:
        DataFrame with cleaned text fields
    """
    df = df.copy()
    
    if 'state' in df.columns:
        df['state'] = df['state'].str.strip().str.title()
    
    if 'district' in df.columns:
        df['district'] = df['district'].str.strip().str.title()
    
    if 'pincode' in df.columns:
        df['pincode'] = df['pincode'].astype(str).str.strip()
    
    return df


def remove_missing_critical_fields(df: pd.DataFrame, 
                                    critical_cols: List[str] = ['date', 'state', 'district']) -> pd.DataFrame:
    """
    Remove rows with missing critical fields.
    
    Args:
        df: Input DataFrame
        critical_cols: List of column names that must not be null
        
    Returns:
        DataFrame with rows removed where critical fields are missing
    """
    initial_count = len(df)
    df = df.dropna(subset=critical_cols)
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        print(f"WARNING: Removed {removed_count:,} rows with missing critical fields")
    
    return df
