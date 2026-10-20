# Statistical Methods in AI (Monsoon 2025)

Coursework for *Statistical Methods in AI* at IIIT Hyderabad, Monsoon 2025. Instructors: Prof. Ravi Kiran Sarvadevabhatla and Prof. Saikiran Bulusu.

The work is grouped by topic, roughly from classical methods to deep learning. Each section has a README with its notebooks and a `Problem_Statement.pdf` for every project.

| # | Section | Projects |
|---|---|---|
| 01 | [Data Analysis and Sampling](01-Data-Analysis-and-Sampling) | Synthetic dataset generation, EDA, simple vs stratified sampling, balanced cohorts |
| 02 | [KNN](02-KNN) | KNN gender classification; Euclidean, Manhattan and cosine distance; feature ablation |
| 03 | [Linear Regression and Regularization](03-Linear-Regression-and-Regularization) | Polynomial regression, Lasso and Ridge, feature importance |
| 04 | [Clustering and PCA](04-Clustering-and-PCA) | K-Means, GMM (EM), image segmentation, PCA, PCA + KNN, data transforms |
| 05 | [Neural Networks from Scratch](05-Neural-Networks-from-Scratch) | NumPy MLP framework, border-map classification, Fourier feature mappings |
| 06 | [Autoencoders and VAE](06-Autoencoders-and-VAE) | MLP autoencoder and anomaly detection, β-VAE on Fashion-MNIST |
| 07 | [CNN](07-CNN) | Multi-task CNN on Fashion-MNIST, CIFAR-10 image colourization |
| 08 | [Decision Trees and Random Forests](08-Decision-Trees-and-Random-Forests) | Regression tree from scratch, tree and forest tuning for review sentiment |
| 09 | [KDE](09-KDE) | Custom KDE, foreground detection |
| 10 | [Time Series and Sequence Modeling](10-Time-Series-and-Sequence-Modeling) | Recurrence discovery, GitHub stars forecasting (ARMA, RNN, CNN) |

## Repository layout

```
.
├── 01-Data-Analysis-and-Sampling/
├── 02-KNN/
├── 03-Linear-Regression-and-Regularization/
├── 04-Clustering-and-PCA/
├── 05-Neural-Networks-from-Scratch/
│   ├── MLP-Border-Classification/
│   └── Fourier-Feature-Image-Reconstruction/
├── 06-Autoencoders-and-VAE/
│   ├── Autoencoder-Anomaly-Detection/
│   ├── VAE/
│   └── data/Q4/
├── 07-CNN/
│   ├── Multitask-CNN-Fashion-MNIST/
│   └── CNN-Image-Colourization/{NIC1,NIC3}/
├── 08-Decision-Trees-and-Random-Forests/
├── 09-KDE/
│   ├── KDE-Foreground-Detection/
│   └── data/Q1/
└── 10-Time-Series-and-Sequence-Modeling/
    ├── RNN-Recurrence-Discovery/
    ├── GitHub-Stars-Forecasting/
    └── data/{Q2,Q3}/
```

## Notes
- **Not included:** model checkpoints, local `wandb/` logs, and datasets that the notebooks download (MNIST, Fashion-MNIST, CIFAR-10). The notebooks rebuild or download them.
- **Username seed:** plots carry a username tag, and random seeds are derived from the username.

## Required input data (not in repo)

LFW is too large for the repo, so add it by hand before you run [Autoencoder_Anomaly_Detection.ipynb](06-Autoencoders-and-VAE/Autoencoder-Anomaly-Detection/Autoencoder_Anomaly_Detection.ipynb):

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?resource=download).
2. Take the inner `lfw-deepfunneled/` folder and rename it to `LFW_Dataset/`.
3. Put it at `06-Autoencoders-and-VAE/Autoencoder-Anomaly-Detection/LFW_Dataset/<Person_Name>/<image>.jpg`.

`.gitignore` excludes `LFW_Dataset/`. All other data is included or downloads on first run.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn scipy opencv-python pillow \
            torch torchvision torchmetrics statsmodels tabulate wandb kagglehub tqdm jupyter
```
