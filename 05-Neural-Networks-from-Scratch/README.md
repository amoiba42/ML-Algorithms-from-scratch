# 05 · Neural Networks from Scratch

| Project | Notebook | Problem statement |
|---|---|---|
| NumPy MLP framework (layers, activations, backprop, optimizers). Belgium–Netherlands border map (Baarle-Nassau) as pixel classification, XOR and gradient checks, architecture and hyperparameter studies | [MLP_Border_Map.ipynb](MLP-Border-Classification/MLP_Border_Map.ipynb) | [Problem_Statement.pdf](MLP-Border-Classification/Problem_Statement.pdf) |
| Coordinate MLP for image reconstruction with polynomial and Fourier feature mappings, including blurred inputs | [Fourier_Feature_Image_Reconstruction.ipynb](Fourier-Feature-Image-Reconstruction/Fourier_Feature_Image_Reconstruction.ipynb) | [Problem_Statement.pdf](Fourier-Feature-Image-Reconstruction/Problem_Statement.pdf) |

## Layout
- `MLP-Border-Classification/data/border.png`: 50×50 pixelated border map
- `Fourier-Feature-Image-Reconstruction/data/smiley.png`: 256×256 grayscale target
- `Fourier-Feature-Image-Reconstruction/data/cat.jpg`: RGB target, resized to 256×256 by the notebook
- `Fourier-Feature-Image-Reconstruction/data/blurred/blur_0.png` … `blur_9.png`: cat image with Gaussian blur σ = 0 to 9
