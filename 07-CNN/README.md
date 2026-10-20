# 07 · CNN

| Project | Notebook | Problem statement |
|---|---|---|
| Multi-task CNN on Fashion-MNIST: classification head plus regression head, λ₁/λ₂ trade-off, feature-map visualisation | [Multitask_CNN.ipynb](Multitask-CNN-Fashion-MNIST/Multitask_CNN.ipynb) (script version: [Multitask_CNN.py](Multitask-CNN-Fashion-MNIST/Multitask_CNN.py)) | [Problem_Statement.pdf](Multitask-CNN-Fashion-MNIST/Problem_Statement.pdf) |
| CIFAR-10 image colourization as per-pixel classification over colour centroids, 1 input channel. Includes analytical weight, activation and connection counts | [Colourization_NIC1.ipynb](CNN-Image-Colourization/NIC1/Colourization_NIC1.ipynb) | [Problem_Statement.pdf](CNN-Image-Colourization/Problem_Statement.pdf) |
| Same colourization model with 3 input channels | [Colourization_NIC3.ipynb](CNN-Image-Colourization/NIC3/Colourization_NIC3.ipynb) | [Problem_Statement.pdf](CNN-Image-Colourization/Problem_Statement.pdf) |

Weights & Biases logs every experiment. The dashboard links are at the end of each notebook.

## Layout
- `Multitask-CNN-Fashion-MNIST/output.csv`: run results
- `CNN-Image-Colourization/NIC*/data/color_centroids.npy`: colour centroids
- `CNN-Image-Colourization/NIC*/output/`: sweep plots, sample colourizations, `output.csv`

## Not included
- `model_checkpoints/` (about 130 MB in total). The multi-task notebook loads three `BEST_*.pth` files from `Multitask-CNN-Fashion-MNIST/model_checkpoints/` for test evaluation. Run training first to create them.
- Local `wandb/` logs, Fashion-MNIST and CIFAR-10. The datasets download into `./data` on first run.
