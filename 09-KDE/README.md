# 09 · KDE

| Project | Notebook | Problem statement |
|---|---|---|
| Custom KDE class (uniform, Gaussian, triangular kernels) for background modelling and foreground detection, with a grid search over bandwidth and threshold | [KDE_Foreground_Detection.ipynb](KDE-Foreground-Detection/KDE_Foreground_Detection.ipynb) | [Problem_Statement.pdf](KDE-Foreground-Detection/Problem_Statement.pdf) |

## Layout
- `data/Q1/`: background (`back.jpg`) and test (`Full.jpg`) images. The notebook reads them through `DATA_DIR = '../data/Q1'`.
- `KDE-Foreground-Detection/output/`: foreground masks for each kernel, bandwidth and threshold, plus `results/grid_search_summary.csv`
