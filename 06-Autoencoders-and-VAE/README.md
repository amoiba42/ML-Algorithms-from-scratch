# 06 · Autoencoders and VAE

| Project | Notebook | Problem statement |
|---|---|---|
| MLP autoencoder on MNIST, then anomaly detection on Labeled Faces in the Wild, with a bottleneck-size study | [Autoencoder_Anomaly_Detection.ipynb](Autoencoder-Anomaly-Detection/Autoencoder_Anomaly_Detection.ipynb) | [Problem_Statement.pdf](Autoencoder-Anomaly-Detection/Problem_Statement.pdf) |
| VAE on Fashion-MNIST: β sweep, latent-space evolution, sampling, frozen latent parameters | [VAE_Fashion_MNIST.ipynb](VAE/VAE_Fashion_MNIST.ipynb) | [Problem_Statement.pdf](VAE/Problem_Statement.pdf) |

The autoencoder uses the MLP framework from [section 05](../05-Neural-Networks-from-Scratch).

## Layout
- `VAE/output/`: loss curves per β, reconstructions, generated samples, latent-evolution GIFs
- `data/Q4/`: reference latent-evolution video. The notebook reads it through `DATA_DIR = '../data/Q4'`.

## Not included
- `Autoencoder-Anomaly-Detection/LFW_Dataset/`: Labeled Faces in the Wild images, one folder per person. Download it from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?resource=download) and rename the inner `lfw-deepfunneled/` folder to `LFW_Dataset/`.
- `VAE/checkpoints/` (about 100 MB). The notebook writes it again during training.
- Fashion-MNIST. torchvision downloads it on first run.
