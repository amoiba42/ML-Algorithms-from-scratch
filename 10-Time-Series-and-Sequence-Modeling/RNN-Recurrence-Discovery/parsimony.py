import numpy as np

def count_parameters(model):
    """Counts the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_stability(lin_reg_model):
    """
    Analyzes the stability of the linear recurrence relation by checking the
    magnitudes of the eigenvalues of its companion matrix.
    A system is stable if all eigenvalues have a magnitude < 1.
    """
    weights = lin_reg_model.coef_.flatten()
    p = len(weights)
    
    # Create the companion matrix
    companion_matrix = np.zeros((p, p))
    companion_matrix[0, :] = weights
    if p > 1:
        companion_matrix[1:, :-1] = np.eye(p - 1)
        
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(companion_matrix)
    eigenvalue_magnitudes = np.abs(eigenvalues)
    
    # Check for stability
    is_stable = np.all(eigenvalue_magnitudes < 1)
    
    return is_stable, eigenvalue_magnitudes

def generate_parsimony_comparison(
    rnn_model, 
    lin_reg_model, 
    p, 
    rnn_mae, 
    anal_mae, 
    rnn_val_mse, 
    anal_val_mse
):
    """Q2.5: Generates a report comparing the DL and Analytical models."""
    
    # 1. Count parameters for parsimony comparison
    rnn_params = count_parameters(rnn_model)
    # Linear model has 'p' weights and 1 bias term
    ar_params = p + 1
    
    # 2. Analyze stability of the analytical model
    is_stable, eig_mags = analyze_stability(lin_reg_model)
    stability_verdict = "Stable" if is_stable else "Unstable"

    # 3. Print the formatted report
    print("\n" + "="*50)
    print("--- Q2.5: Parsimony and Stability Report ---")
    print("="*50)
    
    # --- Table Header ---
    header = f"{'Metric':<25} | {'DL RNN':<20} | {'Analytical AR':<20}"
    print(header)
    print("-" * len(header))
    
    # --- Performance ---
    print(f"{'Test MAE':<25} | {rnn_mae:<20.6f} | {anal_mae:<20.6f}")
    print(f"{'Validation MSE':<25} | {rnn_val_mse:<20.6f} | {anal_val_mse:<20.6f}")
    
    # --- Parsimony ---
    print(f"{'Number of Parameters':<25} | {rnn_params:<20} | {ar_params:<20}")
    
    # --- Stability ---
    print(f"{'System Stability':<25} | {'(Not Applicable)':<20} | {stability_verdict:<20}")
    print(f"{'Max Eigenvalue Mag.':<25} | {'-':<20} | {np.max(eig_mags):<20.4f}")
    
    print("="*50)
    
    # --- Conclusion ---
    print("\nDiscussion:")
    print("The table above compares the complex, black-box RNN model with the simple, interpretable Analytical AR model.")
    if anal_mae < rnn_mae * 1.2: # If analytical is close to RNN performance
        print("The Analytical AR model provides comparable performance with significantly fewer parameters, making it the more parsimonious choice.")
    else:
        print("The DL RNN model achieves better predictive performance, but at the cost of much higher complexity and a lack of interpretability.")
    
    if not is_stable:
        print("Crucially, the identified Analytical AR system is UNSTABLE, meaning its predictions will diverge over long horizons.")
    else:
        print("The identified Analytical AR system is STABLE, meaning its autoregressive forecasts will not explode over time.")
