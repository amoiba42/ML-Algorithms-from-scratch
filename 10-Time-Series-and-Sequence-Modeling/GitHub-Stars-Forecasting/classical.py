import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

def fit_arma_model(df_scaled, order, scaler, target_col='delta_stars_scaled',
                   enforce_stationarity=True, enforce_invertibility=True,
                   method='lbfgs'):
    """
    Fit ARIMA model with constraints to avoid convergence warnings.
    
    Args:
        df_scaled: Scaled dataframe
        order: (p, d, q) tuple for ARIMA
        scaler: Fitted scaler object
        target_col: Column name to model
        enforce_stationarity: Force stationary AR parameters
        enforce_invertibility: Force invertible MA parameters
        method: Optimization method ('lbfgs', 'powell', or 'bfgs')
    
    Returns:
        fitted_model, scaler
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    y = df_scaled[target_col].values
    
    # Build ARIMA model with constraints
    model = SARIMAX(
        y, 
        order=order,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        concentrate_scale=True  # Better numerical stability
    )
    
    # Fit with specified optimizer
    fitted = model.fit(
        disp=False,           # Suppress convergence output
        maxiter=500,          # More iterations for convergence
        method=method,        # Optimization algorithm
        low_memory=True       # Reduce memory usage
    )
    
    return fitted, scaler

def forecast_arma(model_fit, steps):
    """Generates multi-step forecasts using the fitted ARMA model."""
    # Forecast returns both predictions and confidence intervals
    forecast_result = model_fit.forecast(steps=steps)
    
    # statsmodels often returns a Series or array. We need the predictions only.
    if isinstance(forecast_result, pd.Series):
        predictions_scaled = forecast_result.values
    else:
        predictions_scaled = forecast_result
        
    return predictions_scaled