import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from evaluate import add_username_to_plot

DATA_DIR = '../data/Q3'
STARS_FILE = f'{DATA_DIR}/stars_data.csv'
METADATA_FILE= f'{DATA_DIR}/repo_metadata.json' 
REPO_ID_COLUMN_CANDIDATES = ['repo_id', 'repository_id', 'id']

def load_and_clean_data(repo_ids, min_stagnant_days=250):
    """
    Loads star data, cleans timestamps, handles missing values, and
    TRUNCATES the series after a long period of stagnation.
    
    Args:
        repo_ids (list): List of M repository IDs to process.
        min_stagnant_days (int): Minimum consecutive days of zero growth 
                                 required to consider the period 'stagnant' (e.g., 250 days).
        
    Returns:
        dict: Dictionary where keys are repo IDs and values are cleaned DataFrames.
    """
    if not os.path.exists(STARS_FILE):
        print(f"Error: Data file not found at {STARS_FILE}. Cannot proceed.")
        return {} 

    df_raw = pd.read_csv(STARS_FILE)
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])

    # --- Initial Column Check ---
    repo_id_col = None
    for col in df_raw.columns:
        if col.lower() in REPO_ID_COLUMN_CANDIDATES:
            repo_id_col = col
            break
    if repo_id_col is None:
        print("\nFATAL ERROR: Could not find the repository ID column...")
        return {}
    
    cleaned_repos = {}
    
    for repo_id in repo_ids:
        df_repo = df_raw[df_raw[repo_id_col] == repo_id].copy()
        df_repo = df_repo.sort_values('timestamp')
        
        # --- 1. Resampling and Imputation (Ensures Daily Frequency) ---
        df_repo = df_repo.set_index('timestamp')
        df_repo = df_repo.resample('D').ffill() 
        df_repo['stars'] = df_repo['stars'].cummax() 

        # --- 2. Feature Engineering: Cumulative to Incremental (Delta) ---
        df_repo['delta_stars'] = df_repo['stars'].diff().fillna(0)
        
        # --- 3. DYNAMIC TRUNCATION (Crucial Step) ---
        # Identify consecutive days where incremental star count is zero.
        # This prevents training/testing on years of stagnant data.
        
        # Calculate a rolling count of consecutive zeros in delta_stars
        is_zero = (df_repo['delta_stars'] == 0)
        zero_groups = is_zero.ne(is_zero.shift()).cumsum()
        df_repo['zero_count'] = df_repo.groupby(zero_groups)['delta_stars'].transform(lambda x: (x == 0).sum())
        stagnant_df = df_repo[(df_repo['delta_stars'] == 0) & 
                      (df_repo['zero_count'] >= min_stagnant_days)]

        stagnant_start_index = stagnant_df.index.min() # Find the very first day of the valid stagnant period

        if pd.notna(stagnant_start_index):
            # Find the row index (position) of the first day of stagnation.
            stagnant_loc = df_repo.index.get_loc(stagnant_start_index)
            
            # The truncate point is the timestamp *just before* the stagnation began.
            # This keeps the last dynamic, non-zero point.
            truncate_point = df_repo.index[stagnant_loc - 1]
            
            df_repo = df_repo.loc[:truncate_point].copy()
            print(f"Repo {repo_id}: Data truncated at {truncate_point.strftime('%Y-%m-%d')} due to {min_stagnant_days} days of stagnation.")
            
        else:
            print(f"Repo {repo_id}: No extended stagnation period found (Data kept).")

        df_repo = df_repo.drop(columns=['zero_count'], errors='ignore')
        # --- 4. Final Features ---
        df_repo['day_of_week'] = df_repo.index.dayofweek
        df_repo['day_of_year'] = df_repo.index.dayofyear
        
        cleaned_repos[repo_id] = df_repo
        print(f"Repo {repo_id}: Final length: {len(df_repo)} days.")
        
    return cleaned_repos

def normalize_data(df, target_column='delta_stars'):
    """
    Scales the target data using StandardScaler.
    
    Args:
        df (pd.DataFrame): DataFrame containing the target and features.
        target_column (str): Column to be scaled.
        
    Returns:
        tuple: (pd.DataFrame with scaled data, fitted scaler object)
    """
    # 3. Scaling: Use StandardScaler for the target variable
    scaler = StandardScaler()
    
    # Scale the target data
    scaled_data = scaler.fit_transform(df[[target_column]])
    df_scaled = df.copy()
    df_scaled[f'{target_column}_scaled'] = scaled_data.flatten()
    
    return df_scaled, scaler

def plot_star_progression(df_repo, repo_id, output_dir):
    """
    Visualizes the progression of GitHub stars in different time domains for a repository.

    Args:
        df_repo (pd.DataFrame): The cleaned DataFrame for a single repository, with a DatetimeIndex.
        repo_id (str): The identifier of the repository (e.g., 'facebook/react').
        output_dir (str): The directory to save the plot.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    repo_name_safe = repo_id.replace('/', '_')
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot 1: Cumulative Stars (Overall Growth)
    ax1.plot(df_repo.index, df_repo['stars'], label='Cumulative Stars', color='navy')
    ax1.set_ylabel('Total Stars')
    ax1.set_title('Cumulative Star Count Over Time')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Plot 2: Daily Incremental Stars (Daily Volatility)
    ax2.plot(df_repo.index, df_repo['delta_stars'], label='Daily New Stars', color='crimson', alpha=0.8)
    ax2.set_ylabel('New Stars per Day')
    ax2.set_title('Daily Incremental Stars (Δy_t)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # Plot 3: Weekly Aggregated Stars (Smoothed Trend)
    weekly_stars = df_repo['delta_stars'].resample('W').sum()
    ax3.plot(weekly_stars.index, weekly_stars.values, label='Weekly New Stars', color='forestgreen')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('New Stars per Week')
    ax3.set_title('Weekly Aggregated New Stars')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend()
    
    fig.suptitle(f'Star Progression for {repo_id}', fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    save_path = os.path.join(output_dir, f"{repo_name_safe}_star_progression.png")
    plt.savefig(save_path)
    add_username_to_plot("gracy.garg")
    plt.show()
    plt.close()
    print(f"  -> Saved star progression plot to {save_path}")
