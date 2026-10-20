import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

from model import LinearARPredictor, RNNPredictor
from data_prep import RecurrenceData

# --- Training Utilities ---
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr, patience):
    """Q2.2: Trains a model with early stopping."""
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights_path = 'output/best_model_weights.pth' # Temporary path for saving

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_t)
            val_loss = criterion(val_predictions, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_weights_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break

    model.load_state_dict(torch.load(best_weights_path))
    return model, best_val_loss

def tune_history_length(dp_class, model_class, p_candidates, epochs, batch_size, lr, patience):
    """Q2.2: Tunes history length p by evaluating validation MSE."""
    print(f"\n--- Tuning History Length (p) from {p_candidates} ---")
    
    results = {}
    best_val_mse = float('inf')
    best_p = -1

    for p in p_candidates:
        print(f"\n-- Testing p = {p} --")
        dp = dp_class(p=p)
        X_train, y_train, X_val, y_val, _, _ = dp.load_and_split()
        
        model = model_class(input_size=1, hidden_size=32, num_layers=1, output_size=1)
        _, val_mse = train_model(
            model, X_train, y_train, X_val, y_val, 
            epochs=epochs, batch_size=batch_size, lr=lr, patience=patience
        )
        results[p] = val_mse
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_p = p

    print(f"\n--- Tuning Complete. Best p = {best_p} with Val MSE: {best_val_mse:.6f} ---")
    
    # Return all results for plotting
    return best_p, dp_class(p=best_p), results
