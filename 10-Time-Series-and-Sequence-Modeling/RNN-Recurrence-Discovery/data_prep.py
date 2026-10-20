import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import sys

# Define the suggested file names for data organization
DATA_FILE = '../data/Q2/recurrence_timeseries.csv' 

class RecurrenceData:
    """Handles loading, splitting, transforming, and normalizing time series data."""
    def __init__(self, p, data_file='../data/Q2/recurrence_timeseries.csv', train_split=0.7, val_split=0.15):
        self.p = p  # Store history length
        self.data_file = data_file
        self.train_split = train_split
        self.val_split = val_split
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.train_ratio = train_split
        self.val_ratio = val_split
        self.raw_data = None
        # self.series = self._load_series()
        # self.series_normalized = self._normalize_series()


    def load_and_split(self):
        """Loads data, creates chronological splits, and performs normalization (Q2.1)."""
        if not os.path.exists(DATA_FILE):
            print(f"INFO: Data file not found ({DATA_FILE}). Using dummy data (1000 points).")
            t = np.linspace(0, 100, 1000)
            data = 0.5 * np.sin(0.1 * t) + 0.1 * np.random.randn(1000)
            self.raw_data = data.reshape(-1, 1)
        else:
            try:
                df = pd.read_csv(DATA_FILE)
                self.raw_data = df.iloc[:, 0].values.astype(np.float32).reshape(-1, 1)
            except Exception as e:
                print(f"ERROR: Could not read data file {DATA_FILE}. {e}")
                sys.exit(1)

        T = len(self.raw_data)
        train_end = int(T * self.train_ratio)
        val_end = int(T * (self.train_ratio + self.val_ratio))
        
        train_data = self.raw_data[:train_end]
        val_data = self.raw_data[train_end:val_end]
        test_data = self.raw_data[val_end:]

        self.scaler = StandardScaler()
        train_data_norm = self.scaler.fit_transform(train_data)
        val_data_norm = self.scaler.transform(val_data)
        test_data_norm = self.scaler.transform(test_data)
        
        X_train, y_train = self._create_supervised_pairs(train_data_norm)
        X_val, y_val = self._create_supervised_pairs(val_data_norm)
        X_test, y_test = self._create_supervised_pairs(test_data_norm)

        # Reshape for RNN input: (N_samples, p, 1)
        X_train = X_train.reshape(X_train.shape[0], self.p, 1)
        X_val = X_val.reshape(X_val.shape[0], self.p, 1)
        X_test = X_test.reshape(X_test.shape[0], self.p, 1)
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _create_supervised_pairs(self, sequence):
        """Q2.1 Task 2: Form supervised pairs (h_k -> x_k)."""
        X, y = [], []
        for i in range(self.p, len(sequence)):
            # History is x_{k-1}, x_{k-2}, ..., x_{k-p}
            history = sequence[i-self.p:i][::-1] 
            X.append(history)
            y.append(sequence[i])
            
        return np.array(X).squeeze(), np.array(y).squeeze()
    
    def inverse_transform(self, data_normalized):
        """Q2.1 Task 3: Applies the inverse transform."""
        if self.scaler:
            return self.scaler.inverse_transform(data_normalized.reshape(-1, 1)).flatten()
        return data_normalized.flatten()
