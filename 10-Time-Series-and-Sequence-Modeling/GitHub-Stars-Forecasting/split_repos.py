import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # Re-imported for clarity if this were a separate file

def create_supervised_pairs(sequence, lookback):
    """
    Creates lookback sequences (X) and next-step targets (y) for DL models.
    """
    X, y = [], []
    for i in range(lookback, len(sequence)):
        X.append(sequence[i-lookback:i])
        y.append(sequence[i])
    return np.array(X), np.array(y)

def split_repo_data(df_scaled, lookback, train_ratio=0.6, val_ratio=0.2):
    """
    Splits the repository data chronologically into train, val, and test sets.
    
    Args:
        df_scaled (pd.DataFrame): DataFrame with scaled target column.
        lookback (int): Length of input sequence (history).
        train_ratio, val_ratio (float): Ratios for splitting.

    Returns:
        dict: Dictionary containing X/y splits and scaler.
    """
    data = df_scaled['delta_stars_scaled'].values
    T = len(data)

    # 4. Temporal Integrity: Chronological splits
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))
    
    # Raw splits
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Create supervised sequences for DL models
    X_train_seq, y_train_seq = create_supervised_pairs(train_data, lookback)
    X_val_seq, y_val_seq = create_supervised_pairs(val_data, lookback)
    X_test_seq, y_test_seq = create_supervised_pairs(test_data, lookback)
    
    # Reshape for DL models (N_samples, lookback, 1)
    X_train_seq = X_train_seq.reshape(X_train_seq.shape[0], lookback, 1)
    X_val_seq = X_val_seq.reshape(X_val_seq.shape[0], lookback, 1)
    X_test_seq = X_test_seq.reshape(X_test_seq.shape[0], lookback, 1)

    return {
        'X_train': X_train_seq, 'y_train': y_train_seq,
        'X_val': X_val_seq, 'y_val': y_val_seq,
        'X_test': X_test_seq, 'y_test': y_test_seq,
        'df_repo': df_scaled,
        'train_end': train_end, # Full data for ARMA fitting and inverse transform checks
        'val_end': val_end  
    }