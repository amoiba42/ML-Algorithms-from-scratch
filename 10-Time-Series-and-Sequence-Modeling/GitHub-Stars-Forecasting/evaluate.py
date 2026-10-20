import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
import os
from tabulate import tabulate
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
# --- Configuration ---
HORIZON_STEPS = [1, 3, 7, 14, 30] # Required extended horizons for multi-step forecast
OUTPUT_DIR = './output'

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
# ==============================================================================
# 1. Prediction and Forecasting Functions
# ==============================================================================

def predict_dl_model(model, X_input):
    """Generates single-step predictions for the Deep Learning models."""
    model.eval()
    X_t = torch.tensor(X_input, dtype=torch.float32)
    with torch.no_grad():
        predictions_scaled = model(X_t).numpy()
    return predictions_scaled

def multi_step_dl_forecast(model, initial_history, steps, lookback):
    """
    Performs autoregressive (multi-step) forecasting for DL models.
    The input and output are assumed to be SCALED (normalized).
    
    Args:
        model (nn.Module): The trained RNN or CNN model.
        initial_history (np.ndarray): The starting sequence (lookback, 1), scaled.
        steps (int): The number of steps to forecast.
        lookback (int): The sequence length (p).
        
    Returns:
        np.ndarray: Forecasted sequence, scaled.
    """
    history = initial_history.flatten().tolist() # Store as list for easy updating
    forecast_scaled = []
    
    for _ in range(steps):
        # 1. Prepare input for the current step (last 'lookback' points)
        current_input = np.array(history[-lookback:]).reshape(1, lookback, 1)
        X_t = torch.tensor(current_input, dtype=torch.float32)
        
        # 2. Predict next value (single step)
        with torch.no_grad():
            model.eval()
            prediction = model(X_t).item()
            
        # 3. Update forecast and history
        forecast_scaled.append(prediction)
        history.append(prediction) # Feed prediction back into history
        
    return np.array(forecast_scaled)

def multi_step_arma_forecast(model_fit, steps):
    """
    Performs multi-step forecasting using the fitted ARMA model.
    ARMA forecasting is intrinsically autoregressive.
    """
    # model_fit.forecast(steps) handles the autoregressive generation internally.
    forecast_scaled = model_fit.forecast(steps=steps)
    return forecast_scaled

# ==============================================================================
# 2. Evaluation Protocol (Metrics and Backtesting)
# ==============================================================================

def calculate_metrics(y_true_unscaled, y_pred_unscaled):
    """Computes MAE, RMSE, and MSE."""
    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    mse = mean_squared_error(y_true_unscaled, y_pred_unscaled)
    rmse = sqrt(mse)
    return mae, rmse

def evaluate_models(models, data_splits, scaler, repo_id, lookback, df_repo):
    """
    Q3.2.3: Runs single-step and multi-step evaluations across all models.
    
    Args:
        models (dict): Dictionary of trained models.
        data_splits (dict): Data splits for the specific repo.
        scaler (MinMaxScaler): Scaler object.
        repo_id (str): ID of the repository.
        lookback (int): Sequence length.
        df_repo (pd.DataFrame): Full repository data (used for ARMA fitting).
        
    Returns:
        tuple: (dict of single-step test metrics, dict of error-vs-horizon data)
    """
    X_test, y_test = data_splits['X_test'], data_splits['y_test']
    
    # --- 2.1 Single-step Prediction and Evaluation ---
    single_step_metrics = {}
    
    for name, model in models.items():
        if name == 'ARMA':
            # ARMA requires a special approach as it's not sequence-based (X_test)
            # We use the raw scaled test data for comparison.
            # For simplicity here, we assume the fitted ARMA model's in-sample prediction 
            # ability provides the single-step baseline. 
            # The true test is done via multi-step backtesting below.
            
            # Use the fitted model to generate a forecast equal to the test set length
            arma_forecast_scaled = multi_step_arma_forecast(model, len(y_test))
            y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_unscaled = scaler.inverse_transform(arma_forecast_scaled.reshape(-1, 1)).flatten()
            
        else: # DL Models (RNN, CNN)
            y_pred_scaled = predict_dl_model(model, X_test)
            
            # Inverse transform to original scale
            y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_unscaled = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        mae, rmse = calculate_metrics(y_test_unscaled, y_pred_unscaled)
        single_step_metrics[name] = {'MAE': mae, 'RMSE': rmse}


    # --- 2.2 Multi-step Autoregressive Forecasting and Error vs. Horizon ---
    
    # We use the first window of the test set history to start the forecast
    initial_history_scaled = X_test[0].flatten() 
    
    error_vs_horizon = {}
    
    for horizon in HORIZON_STEPS:
        # Get the ground truth segment for this horizon
        y_true_horizon_scaled = y_test[:horizon]
        y_true_horizon_unscaled = scaler.inverse_transform(y_true_horizon_scaled.reshape(-1, 1)).flatten()
        
        errors = {}
        for name, model in models.items():
            if name == 'ARMA':
                y_pred_horizon_scaled = multi_step_arma_forecast(model, horizon)
            else:
                y_pred_horizon_scaled = multi_step_dl_forecast(model, initial_history_scaled, horizon, lookback)
            
            # Inverse transform prediction
            y_pred_horizon_unscaled = scaler.inverse_transform(y_pred_horizon_scaled.reshape(-1, 1)).flatten()
            
            # Calculate metrics (MAE only for the plot)
            mae, rmse = calculate_metrics(y_true_horizon_unscaled, y_pred_horizon_unscaled)
            errors[name] = {
                'MAE': mae, 
                'RMSE': rmse,
                'predictions': y_pred_unscaled,  # ← Add this
                'actuals': y_test_unscaled
            }
        
        error_vs_horizon[horizon] = errors
        
    return single_step_metrics, error_vs_horizon

# ==============================================================================
# 3. Visualization and Reporting Functions
# ==============================================================================

def plot_error_vs_horizon_chart(error_data, repo_id):
    """
    Plots MAE vs forecast horizon for ARMA, RNN, CNN.
    error_data[h][model] must contain {"MAE": x, "RMSE": y}.
    """

    horizons = sorted(error_data.keys())

    first_h = horizons[0]
    model_names = list(error_data[first_h].keys())

    plot_data = {
        model: [error_data[h][model]["MAE"] for h in horizons]
        for model in model_names
    }

    plt.figure(figsize=(10, 6))

    colors = {'ARMA': 'red', 'RNN': 'blue', 'CNN': 'green'}

    for model, mae_values in plot_data.items():
        plt.plot(
            horizons,
            mae_values,
            label=f"{model} MAE",
            color=colors.get(model, 'black'),
            linestyle='--'
        )

    plt.title(f"Multi-step Forecasting Error (MAE) vs. Horizon\nRepository: {repo_id}")
    plt.xlabel("Forecast Horizon (Steps h)")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.legend()
    plt.grid(True, linestyle=':')

    add_username_to_plot("gracy.garg")
    safe_repo_id = repo_id.replace('/', '_')
    filename = f"{OUTPUT_DIR}/forecast_horizon_repo_{safe_repo_id}.png"

    plt.savefig(filename)
    plt.show()
    plt.close()

    print(f"Saved stability plot to {filename}")

def plot_single_step_forecast(trained_models, data_splits, scaler, repo_id, output_dir='./output'):
    """
    Plot single-step ahead forecasts vs ground truth on test set.
    """
    import os
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Single-Step Forecast Comparison\nRepository: {repo_id}', 
                 fontsize=16, fontweight='bold')
    
    # Get test data (ground truth)
    y_test_scaled = data_splits['y_test']
    y_test_original = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    
    # Limit to first N points for visibility
    n_display = min(100, len(y_test_original))
    time_steps = np.arange(n_display)
    
    print(f"\n{'='*60}")
    print(f"Single-Step Forecast Plot for {repo_id}")
    print(f"{'='*60}")
    print(f"Test set size: {len(y_test_original)}")
    print(f"Displaying first {n_display} predictions")
    print(f"Ground truth range: [{y_test_original[:n_display].min():.2f}, {y_test_original[:n_display].max():.2f}]")
    
    # Plot each model
    model_colors = {'ARMA': 'red', 'RNN': 'blue', 'CNN': 'green'}
    model_idx = 0
    
    for model_name, model in trained_models.items():
        if model_idx >= 3:  # Only 3 models
            break
            
        ax = axes.flatten()[model_idx]
        
        # Get predictions based on model type
        if model_name == 'ARMA' or isinstance(model, SARIMAXResults):
            try:
                forecast_scaled = model.forecast(steps=n_display)
                
                # Convert to numpy array if it's a pandas Series
                if hasattr(forecast_scaled, 'values'):
                    forecast_array = forecast_scaled.values
                else:
                    forecast_array = np.asarray(forecast_scaled)
                
                # Inverse transform
                predictions_original = scaler.inverse_transform(
                    forecast_array.reshape(-1, 1)
                ).flatten()
                
            except Exception as e:
                print(f"WARNING: ARMA forecast failed: {e}")
                print(f"  forecast_scaled type: {type(forecast_scaled)}")
                print(f"  forecast_scaled shape: {forecast_scaled.shape if hasattr(forecast_scaled, 'shape') else 'N/A'}")
                predictions_original = np.full(n_display, np.nan)
        
        else:
            # Deep Learning Models (RNN/CNN)
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(data_splits['X_test'][:n_display])
                pred_scaled = model(X_test_tensor).cpu().numpy().flatten()
                predictions_original = scaler.inverse_transform(
                    pred_scaled.reshape(-1, 1)
                ).flatten()
        
        # Calculate metrics for displayed window
        valid_mask = ~np.isnan(predictions_original)
        if valid_mask.sum() > 0:
            mae = np.mean(np.abs(predictions_original[valid_mask] - y_test_original[:n_display][valid_mask]))
            rmse = np.sqrt(np.mean((predictions_original[valid_mask] - y_test_original[:n_display][valid_mask])**2))
        else:
            mae, rmse = np.nan, np.nan
        
        print(f"\n{model_name}:")
        print(f"  Prediction range: [{np.nanmin(predictions_original):.2f}, {np.nanmax(predictions_original):.2f}]")
        print(f"  MAE: {mae:.3f} | RMSE: {rmse:.3f}")
        
        # Plot ground truth
        ax.plot(time_steps, y_test_original[:n_display], 
                color='black', linewidth=2, label='Ground Truth', alpha=0.7)
        
        # Plot predictions
        ax.plot(time_steps, predictions_original[:n_display], 
                color=model_colors[model_name], linewidth=1.5, 
                label=f'{model_name} Forecast', linestyle='--', alpha=0.8)
        
        # Styling
        ax.set_xlabel('Time Step (Days)', fontsize=11)
        ax.set_ylabel('Incremental Stars (Δy_t)', fontsize=11)
        ax.set_title(f'{model_name} Model\nMAE: {mae:.2f} | RMSE: {rmse:.2f}', 
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        model_idx += 1
    
    # Combined plot in 4th subplot
    ax_combined = axes.flatten()[3]
    ax_combined.plot(time_steps, y_test_original[:n_display], 
                     color='black', linewidth=2.5, label='Ground Truth', alpha=0.8)
    
    for model_name, model in trained_models.items():
        if model_name == 'ARMA' or isinstance(model, SARIMAXResults):
            try:
                forecast_scaled = model.forecast(steps=n_display)
                
                if hasattr(forecast_scaled, 'values'):
                    forecast_array = forecast_scaled.values
                else:
                    forecast_array = np.asarray(forecast_scaled)
                
                predictions_original = scaler.inverse_transform(
                    forecast_array.reshape(-1, 1)
                ).flatten()
            except:
                predictions_original = np.full(n_display, np.nan)
        else:
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(data_splits['X_test'][:n_display])
                pred_scaled = model(X_test_tensor).cpu().numpy().flatten()
                predictions_original = scaler.inverse_transform(
                    pred_scaled.reshape(-1, 1)
                ).flatten()
        
        ax_combined.plot(time_steps, predictions_original[:n_display], 
                        color=model_colors[model_name], linewidth=1.5,
                        label=f'{model_name}', linestyle='--', alpha=0.7)
    
    ax_combined.set_xlabel('Time Step (Days)', fontsize=11)
    ax_combined.set_ylabel('Incremental Stars (Δy_t)', fontsize=11)
    ax_combined.set_title('All Models Combined', fontsize=12, fontweight='bold')
    ax_combined.legend(loc='upper right', fontsize=10)
    ax_combined.grid(True, alpha=0.3, linestyle='--')
    
    # Save plot
    plt.tight_layout()
    output_path = f'{output_dir}/{repo_id.replace("/", "_")}_single_step_forecast.png'
    add_username_to_plot("gracy.garg")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✓ Saved single-step forecast plot to: {output_path}")
    print(f"{'='*60}\n")
    
    plt.close()


def plot_single_step_residuals(trained_models, data_splits, scaler, repo_id, output_dir='./output'):
    """
    Plot residuals (prediction errors) for single-step forecasts.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Single-Step Forecast Residuals\nRepository: {repo_id}', 
                 fontsize=16, fontweight='bold')
    
    # Get test data
    y_test_scaled = data_splits['y_test']
    y_test_original = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    n_display = min(100, len(y_test_original))
    
    model_colors = {'ARMA': 'red', 'RNN': 'blue', 'CNN': 'green'}
    
    for idx, (model_name, model) in enumerate(trained_models.items()):
        if idx >= 3:
            break
            
        ax = axes[idx]
        
        # Get predictions
        if model_name == 'ARMA' or isinstance(model, SARIMAXResults):
            try:
                forecast_scaled = model.forecast(steps=n_display)
                
                if hasattr(forecast_scaled, 'values'):
                    forecast_array = forecast_scaled.values
                else:
                    forecast_array = np.asarray(forecast_scaled)
                
                predictions_original = scaler.inverse_transform(
                    forecast_array.reshape(-1, 1)
                ).flatten()
            except Exception as e:
                print(f"WARNING: ARMA residual plot failed: {e}")
                predictions_original = np.full(n_display, np.nan)
        else:
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(data_splits['X_test'][:n_display])
                pred_scaled = model(X_test_tensor).cpu().numpy().flatten()
                predictions_original = scaler.inverse_transform(
                    pred_scaled.reshape(-1, 1)
                ).flatten()
        
        # Calculate residuals
        residuals = y_test_original[:n_display] - predictions_original
        
        # Plot residuals
        ax.scatter(range(n_display), residuals, 
                   color=model_colors[model_name], alpha=0.5, s=20)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='Zero Error')
        ax.axhline(y=np.nanmean(residuals), color='orange', 
                   linestyle=':', linewidth=2, label=f'Mean: {np.nanmean(residuals):.2f}')
        
        # Styling
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Residual (Actual - Predicted)', fontsize=11)
        ax.set_title(f'{model_name}\nStd: {np.nanstd(residuals):.2f}', 
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f'{output_dir}/{repo_id.replace("/", "_")}_residuals.png'
    add_username_to_plot("gracy.garg")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved residual plot to: {output_path}")
    plt.close()

def plot_representative_forecast(models, data_splits, scaler, repo_id, lookback):
    """
    Q3.2.3: Plots the true values vs. a single multi-step forecast for a visual comparison.
    """
    # Use the first 30 days of the test set for the representative forecast
    forecast_length = 30
    
    X_test, y_test = data_splits['X_test'], data_splits['y_test']
    y_true_unscaled = scaler.inverse_transform(y_test[:forecast_length].reshape(-1, 1)).flatten()
    
    # We use the initial history from the test set for the start
# Prefer last 'lookback' points from scaled training series (the true forecasting origin)
    y_train_scaled = np.asarray(data_splits.get('y_train'))
    if y_train_scaled is not None and len(y_train_scaled) >= lookback:
        initial_history_scaled = y_train_scaled[-lookback:].reshape(-1)
    else:
        # fallback (but warn)
        initial_history_scaled = data_splits['X_test'][0].flatten()
        print("WARNING: insufficient training data for representative forecast; using X_test[0] as fallback")

    
    # --- Define markers and colors for consistency ---
    colors = {'ARMA': 'red', 'RNN': 'blue', 'CNN': 'green'}
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_unscaled, label='Ground Truth (Test Data)', color='black', linewidth=2)
    
    for name, model in models.items():
        if name == 'ARMA':
            y_pred_scaled = multi_step_arma_forecast(model, forecast_length)
        else:
            y_pred_scaled = multi_step_dl_forecast(model, initial_history_scaled, forecast_length, lookback)
            
        y_pred_unscaled = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        plt.plot(
            y_pred_unscaled, 
            label=f'{name} Forecast', 
            linestyle='--',
            color=colors.get(name, 'gray')   # Use color from dict
        )
        
    plt.title(f"Representative 30-Day Incremental Star Forecast\nRepository: {repo_id}")
    plt.xlabel("Time Step (Days)")
    plt.ylabel("Incremental Star Count (Δyt)")
    plt.legend()
    plt.grid(True, linestyle=':')
    
    # Add username
    add_username_to_plot("gracy.garg")
    safe_repo_id = repo_id.replace('/', '_') # Replace '/' with '_'
    filename = f"{OUTPUT_DIR}/representative_forecast_repo_{safe_repo_id}.png"
    plt.savefig(filename)
    plt.show()
    plt.close()
    print(f"Saved representative forecast plot to {filename}")

def plot_single_metrics(single_metrics, repo_id, output_dir):
    """
    Creates and saves a bar chart summarizing MAE and RMSE for a single repository.

    Args:
        single_metrics (dict): A dictionary of metrics for one repo. 
                               Example: {'ARMA': {'MAE': x, 'RMSE': y}, ...}
        repo_id (str): The identifier for the repository.
        output_dir (str): Directory to save the plot.
    """
    MODEL_COLORS = {'ARMA': 'red', 'RNN': 'blue', 'CNN': 'green'}
    
    # Extract model names and their corresponding metrics
    model_names = list(single_metrics.keys())
    mae_values = [single_metrics[model]['MAE'] for model in model_names]
    rmse_values = [single_metrics[model]['RMSE'] for model in model_names]
    colors = [MODEL_COLORS.get(model, 'gray') for model in model_names]

    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Single-Step Performance Metrics for: {repo_id}', fontsize=16)

    # MAE Plot
    ax1.bar(model_names, mae_values, color=colors)
    ax1.set_title('Mean Absolute Error (MAE)')
    ax1.set_ylabel('MAE (Lower is Better)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # RMSE Plot
    ax2.bar(model_names, rmse_values, color=colors)
    ax2.set_title('Root Mean Squared Error (RMSE)')
    ax2.set_ylabel('RMSE (Lower is Better)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    safe_repo_id = repo_id.replace('/', '_')
    save_path = os.path.join(output_dir, f"single_metrics_{safe_repo_id}.png")
    add_username_to_plot("gracy.garg")
    plt.savefig(save_path)
    plt.show()
    plt.close()
    print(f"Saved single-repo metrics plot to {save_path}")


def generate_summary_table(all_results):
    """
    Prints a detailed, multi-horizon summary table using the tabulate library for neat formatting.
    Accepts metrics_multi_horizon in either:
      - {horizon: {model: {"MAE":..., "RMSE":...}}} OR
      - {model: {horizon: {"MAE":..., "RMSE":...}}}
    """
    # --- 1. Discover HORIZON STEPS ---
    HORIZON_STEPS = []
    # Simplified discovery by iterating through all_results
    for repo_id, data in all_results.items():
        mmh = data.get('metrics_multi_horizon', {})
        if not mmh:
            continue
        first_key = next(iter(mmh))
        first_val = mmh[first_key]
        
        # Determine the structure:
        if isinstance(first_val, dict) and first_val and isinstance(next(iter(first_val.values())), dict):
            # Structure is: horizon -> model -> metrics (Keys are horizons)
            HORIZON_STEPS = sorted(list(mmh.keys()))
        elif isinstance(first_val, dict):
            # Structure is: model -> horizon -> metrics (Keys of first_val are horizons)
            HORIZON_STEPS = sorted(list(first_val.keys()))
        break # Only need to check one repo

    # --- 2. Build normalized model->horizon->metrics mapping per repo ---
    data_rows = []
    for repo_id, data in all_results.items():
        val_data = data.get('validation_mse', {})
        single = data.get('metrics_single', {})
        mmh = data.get('metrics_multi_horizon', {})

        # Normalize mmh into model_multi_map: { model_name: { horizon: {"MAE":..,"RMSE":..} } }
        model_multi_map = {}
        if mmh:
            some_val = next(iter(mmh.values()))
            if some_val and isinstance(next(iter(some_val.values())), dict):
                # mmh is horizon -> model -> metrics
                for h, mdict in mmh.items():
                    for mname, metrics in mdict.items():
                        model_multi_map.setdefault(mname, {})[h] = metrics
            else:
                model_multi_map = mmh

        # For each model in single-step metrics produce a row
        for model_name, metrics in single.items():
            cfg = val_data.get(model_name, {})
            val_mse_val = cfg.get('Val_MSE')
            architecture = cfg.get('Architecture', 'N/A')
            
            # Use raw float values for better formatting control later
            row = {
                'Repo ID': repo_id,
                'Model': model_name,
                'Architecture': architecture,
                'Val MSE': val_mse_val if isinstance(val_mse_val, (int, float)) else float('nan'),
                'Single-Step MAE': metrics.get('MAE', float('nan')),
                'Single-Step RMSE': metrics.get('RMSE', float('nan')),
            }

            for h in HORIZON_STEPS:
                model_h_metrics = model_multi_map.get(model_name, {})
                h_metrics = model_h_metrics.get(h, {})
                row[f"MAE @ L={h}"] = h_metrics.get('MAE', float('nan'))
                row[f"RMSE @ L={h}"] = h_metrics.get('RMSE', float('nan'))

            data_rows.append(row)

    # --- 3. Create DataFrame and print in a stable column order ---
    if not data_rows:
        print("\nNo results to display (all_results is empty or missing required keys).")
        return

    df = pd.DataFrame(data_rows)

    fixed_cols = ['Repo ID', 'Model', 'Architecture', 'Val MSE',
                  'Single-Step MAE', 'Single-Step RMSE']
    multi_cols = []
    for h in HORIZON_STEPS:
        multi_cols.append(f"MAE @ L={h}")
        multi_cols.append(f"RMSE @ L={h}")

    final_cols = [c for c in fixed_cols + multi_cols if c in df.columns]

    # --- Use tabulate to print neat table ---
    print("\n" + "="*120)
    print("                 Q3.2.3 SUMMARY TABLE: FULL METRICS AND CONFIGURATION (MAE & RMSE)")
    print("="*120)
    
    # Prepare data for tabulate
    # headers=list(df[final_cols].columns) uses the column names as headers
    # tablefmt="fancy_grid" creates a highly readable, neat table with lines.
    # floatfmt=".6f" ensures all float numbers are formatted to 6 decimal places.
    table_data = tabulate(
        df[final_cols], 
        headers='keys', 
        tablefmt="fancy_grid",
        showindex=False,
        floatfmt=".6f"
    )
    print(table_data)
    print("="*120)

