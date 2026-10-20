# 10 · Time Series and Sequence Modeling

| Project | Notebook | Problem statement |
|---|---|---|
| Learn a hidden discrete-time recurrence with linear AR, MLP, CNN, RNN and LSTM predictors, identify it analytically, and compare parsimony and stability | [Recurrence_Discovery.ipynb](RNN-Recurrence-Discovery/Recurrence_Discovery.ipynb) | [Problem_Statement.pdf](RNN-Recurrence-Discovery/Problem_Statement.pdf) |
| Forecast cumulative GitHub stars with ARMA, RNN/GRU and 1D CNN models | [GitHub_Stars_Forecasting.ipynb](GitHub-Stars-Forecasting/GitHub_Stars_Forecasting.ipynb) | [Problem_Statement.pdf](GitHub-Stars-Forecasting/Problem_Statement.pdf) |

## Layout
- `data/Q2/recurrence_timeseries.csv` and `data/Q3/{stars_data.csv, repo_metadata.json}`. The code reads them through `../data/Q2` and `../data/Q3`, so keep these folder names.
- `RNN-Recurrence-Discovery/`: `data_prep.py`, `model.py`, `train.py`, `analyze.py`, `parsimony.py`
- `GitHub-Stars-Forecasting/`: `prep_stars.py`, `split_repos.py`, `classical.py`, `dl_models.py`, `train_models.py`, `evaluate.py`
- Each project has an `output/` folder with plots and the best model weights.

Run each notebook from its own folder so that the relative paths resolve.
