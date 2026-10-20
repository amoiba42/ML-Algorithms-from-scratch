import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from model import RecurrenceModel, RNNPredictor, LinearARPredictor
from data_prep import RecurrenceData
# This file will primarily use the best trained data processor (dp) and model.
def add_username_to_plot(username="gracy.garg"):
    """
    Adds the username to the top-right corner of the current plot.
    """
    ax = plt.gca()
    # For 3D axes, use text2D to place text in axes (fractional) coordinates
    if getattr(ax, "name", "") == "3d" or ax.__class__.__name__ == "Axes3D":
        ax.text2D(0.95, 0.95, username, ha="right", va="top", transform=ax.transAxes,
                  fontsize=10, color="gray", alpha=0.7)
    else:
        plt.text(0.95, 0.95, username, ha="right", va="top",
                 transform=plt.gca().transAxes, fontsize=10, color="gray", alpha=0.7)
    plt.text(
        0.95, 0.95, username,
        ha='right', va='top',
        transform=plt.gca().transAxes,
        fontsize=10, color='gray', alpha=0.7
    )

def identify_analytical_recurrence(dp_best, X_train_norm, y_train_norm):
    """Q2.3: Identifies a closed-form recurrence F_theta using Linear Regression."""
    p = dp_best.p
    X_train_flat = X_train_norm.squeeze(-1) 
    y_train_flat = y_train_norm.flatten()  
    
    X_train_unnorm = np.zeros_like(X_train_flat)
    y_train_unnorm = dp_best.inverse_transform(y_train_flat.reshape(-1, 1)).flatten()  
    
    for i in range(p):
        X_train_unnorm[:, i] = dp_best.inverse_transform(X_train_flat[:, i].reshape(-1, 1)).flatten()  
    
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_unnorm, y_train_unnorm)  
    weights = lin_reg.coef_
    bias = lin_reg.intercept_  
    
    print(f"\n--- Analytical Recurrence F_theta Identified (p={p}) ---")
    print(f"Equation: x_k = {bias:.4f} + sum(w_i * x_{{k-i}})")  
    print(f"Weights (w_1 to w_{p}): {weights}")
    
    def analytical_predictor(history_vector_unnorm):
        """
        Predicts next value given history.
        Args:
            history_vector_unnorm: shape (N, p) for batch or (p,) for single
        Returns:
            Predictions as array (N,) or scalar
        """
        if history_vector_unnorm.ndim == 1:
            # Single prediction: (p,) -> scalar
            return np.dot(history_vector_unnorm, weights) + bias
        else:
            # Batch predictions: (N, p) -> (N,)
            return history_vector_unnorm @ weights + bias
    
    return lin_reg, analytical_predictor

def evaluate_single_step(model, X_test, y_test, dp_best):
    """Q2.4: Evaluates single-step prediction with the DL model."""
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    with torch.no_grad():
        y_pred_norm = model(X_test_t).numpy()
    
    y_test_unnorm = dp_best.inverse_transform(y_test_t.numpy()).astype(np.float64)
    y_pred_unnorm = dp_best.inverse_transform(y_pred_norm).astype(np.float64)

    mae = mean_absolute_error(y_test_unnorm, y_pred_unnorm)
    mse = mean_squared_error(y_test_unnorm, y_pred_unnorm)
    
    return mae, mse, y_test_unnorm, y_pred_unnorm

def autoregressive_generation(model_predictor, initial_history_unnorm, forecast_steps, dp_best, is_dl_model=True):
    """Q2.4: Generates a sequence autoregressively (multi-step forecast)."""
    p = dp_best.p
    current_history_unnorm = list(initial_history_unnorm) 
    generated_sequence_unnorm = []

    for _ in range(forecast_steps):
        history_to_predict = np.array(current_history_unnorm[-p:])[::-1] 

        if is_dl_model:
            history_norm = dp_best.scaler.transform(history_to_predict.reshape(-1, 1)).flatten()
            X_input_t = torch.tensor(history_norm, dtype=torch.float32).reshape(1, p, 1)
            
            with torch.no_grad():
                model_predictor.eval()
                pred_norm = model_predictor(X_input_t).item()
            
            pred_unnorm = dp_best.inverse_transform(np.array([[pred_norm]]))[0]

        else:
            pred_unnorm = model_predictor(history_to_predict)

        generated_sequence_unnorm.append(pred_unnorm)
        current_history_unnorm.append(pred_unnorm)
    
    return np.array(generated_sequence_unnorm)

def plot_predictions(y_true, y_pred_rnn, y_pred_ar, num_points=200):
    """Plots model predictions against the ground truth for a sample of test points."""
    plt.figure(figsize=(12, 8))
    
    # Use a subset of points for clarity
    indices = np.arange(len(y_true))
    if len(y_true) > num_points:
        indices = np.random.choice(indices, num_points, replace=False)
        indices.sort()

    plt.plot(y_true[indices], label='Ground Truth', color='black', linewidth=2, alpha=0.7)
    plt.plot(y_pred_rnn[indices], label='DL RNN Prediction', color='blue', linestyle='--', marker='o', markersize=4)
    plt.plot(y_pred_ar[indices], label='Analytical AR Prediction', color='red', linestyle=':', marker='x', markersize=4)
    
    plt.title(f'Model Predictions vs. Ground Truth (Sample of {len(indices)} points)')
    plt.xlabel('Time Step (Sampled)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    add_username_to_plot("gracy.garg")
    plt.savefig('output/model_predictions.png')
    plt.show()
    plt.close()
    

def plot_error_vs_horizon(dp_best, dl_model, anal_predictor, X_test, y_test_unnorm, max_horizon):
    """Q2.4: Plots variation in error with increasing forecasting length."""
    p = dp_best.p
    dl_errors = []
    anal_errors = []
    horizons = range(1, max_horizon + 1)
    
    initial_history = dp_best.inverse_transform(X_test[0].flatten())
    
    for h in horizons:
        y_true_h = y_test_unnorm[:h]
        
        dl_pred_h = autoregressive_generation(dl_model, initial_history, h, dp_best, is_dl_model=True)
        dl_errors.append(mean_absolute_error(y_true_h, dl_pred_h))

        anal_pred_h = autoregressive_generation(anal_predictor, initial_history, h, dp_best, is_dl_model=False)
        anal_errors.append(mean_absolute_error(y_true_h, anal_pred_h))
        
    plt.figure(figsize=(10, 6))
    plt.plot(horizons, dl_errors, label='DL RNN (MAE)', marker='o', linestyle='-', markersize=4)
    plt.plot(horizons, anal_errors, label='Analytical AR (MAE)', marker='x', linestyle='--', markersize=4)
    
    plt.title("Forecasting Error (MAE) vs. Forecast Horizon")
    plt.xlabel("Forecast Horizon (Steps)")
    plt.ylabel("Mean Absolute Error (MAE) on Test Set")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig('output/error_vs_horizon.png')
    add_username_to_plot('gracy.garg')
    plt.show()