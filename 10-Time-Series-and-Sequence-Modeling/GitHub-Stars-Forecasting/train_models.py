import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import os

from dl_models import RNNForecaster, CNNForecaster

def train_dl_model(model, data_splits, model_name, epochs=50, batch_size=32, lr=0.001, patience=10):
    """
    Q3.2.2: Trains a PyTorch DL model (RNN or CNN) with validation and early stopping.
    
    Args:
        model (nn.Module): The PyTorch model to train.
        data_splits (dict): Contains X/y_train/val/test splits (NumPy arrays).
        model_name (str): Identifier (e.g., 'RNN' or 'CNN') for saving weights.
        epochs (int): Maximum number of training iterations.
        batch_size (int): Size of batches for training.
        lr (float): Learning rate for the optimizer.
        patience (int): Number of epochs to wait for improvement before early stopping.
        
    Returns:
        tuple: (Trained model, Best validation loss achieved)
    """
    X_train, y_train = data_splits['X_train'], data_splits['y_train']
    X_val, y_val = data_splits['X_val'], data_splits['y_val']

    # Convert NumPy arrays to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Justification: Using MSE Loss as it penalizes large forecasting errors heavily, 
    # which is appropriate for modeling potentially volatile growth rates.
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights_path = f'output/best_weights_{model_name}.pth'

    print(f"\nStarting training for {model_name}...")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)

        # Validation Step
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_t)
            val_loss = criterion(val_predictions, y_val_t).item()
        
        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model state dict
            torch.save(model.state_dict(), best_weights_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping triggered for {model_name} at epoch {epoch+1}.")
                break
        
        # Optional: Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_train_loss = total_train_loss / len(X_train)
            print(f"Epoch {epoch+1}/{epochs} | Train MSE: {avg_train_loss:.6f} | Val MSE: {val_loss:.6f}")


    # Load the best weights found during training
    model.load_state_dict(torch.load(best_weights_path))
    
    return model, best_val_loss