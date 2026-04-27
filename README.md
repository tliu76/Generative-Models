# Generative Models: VAE, GAN, and Diffusion

A PyTorch implementation of three core generative model families — Variational Autoencoders (VAE), Generative Adversarial Networks (GAN), and Diffusion Models — trained on MNIST and FashionMNIST.


## Overview

| Model | Dataset | Key Idea |
|---|---|---|
| VAE | MNIST | Encode to latent distribution, decode via reparameterization |
| GAN | FashionMNIST | Generator vs. Discriminator adversarial training |
| Diffusion (Simple) | Single MNIST image | Learn to predict added noise with an MLP |
| Diffusion (DDPM + UNet) | FashionMNIST | Time-conditioned UNet denoiser with 500 timesteps |

## Project Structure

```
Generative-Models/
├── models/
│   ├── VAE.py               # BasicEncoder, VAE (reparameterization trick)
│   ├── GAN.py               # BasicDiscriminator, BasicLeakyGenerator
│   ├── UNet.py              # Time-conditioned UNet for DDPM
│   ├── noise_predictor.py   # SimpleNoisePredictor (MLP) for simple diffusion
│   └── decoder.py           # BasicDecoder (shared by VAE and GAN generator)
├── configs/
│   ├── config_vae.yaml      # VAE hyperparameters
│   ├── config_gan.yaml      # GAN hyperparameters
│   └── config_diffusion.yaml # Diffusion hyperparameters
├── utils/
│   ├── trainer.py           # Base Trainer class (data loading, optimizer setup)
│   └── data_utils.py        # Dataset helpers
├── simple_diffusion.py      # Single-image diffusion trainer/demo
├── trainer_vae.py           # VAE training loop
├── trainer_gan.py           # GAN training loop
├── trainer_diffusion.py     # DDPM training loop
└── config.py                # Config dataclass helper
```

## Models

### 1. Variational Autoencoder (VAE)

**Architecture:** `BasicEncoder` → reparameterization → `BasicDecoder`

- **Encoder**: Linear(784 → 512) → ReLU → two parallel heads for `μ` and `log σ²`
- **Reparameterization**: `z = μ + σ · ε`, where `ε ~ N(0, I)`
- **Decoder**: Linear(latent → 512) → ReLU → Linear(512 → 784) → Sigmoid

**Loss**: Reconstruction loss (L1 or L2) + KL divergence weighted by `β`

| Hyperparameter | Value |
|---|---|
| Hidden dim | 512 |
| Latent dim | 32 |
| Recon loss | L2 |
| β (KL weight) | 0.1 |
| Batch size | 128 |
| Epochs | 100 |
| LR | 3e-4 |
| Optimizer | AdamW |

### 2. Generative Adversarial Network (GAN)

**Architecture:** `BasicDiscriminator` vs. `BasicLeakyGenerator`

- **Discriminator**: Linear(784 → 256) → LeakyReLU → Linear(256 → 1) → Sigmoid
- **Generator**: Linear(latent → 256) → LeakyReLU → Linear(256 → 784) → Sigmoid

| Hyperparameter | Value |
|---|---|
| Hidden dim | 256 |
| Latent dim | 64 |
| Activation | LeakyReLU |
| Batch size | 128 |
| Epochs | 120 |
| LR | 1e-4 |
| Optimizer | AdamW |

### 3. Simple Diffusion (Single-Image MLP)

A lightweight demonstration on a single MNIST image:
- Adds fixed random noise to an image
- Trains a 3-layer MLP (`SimpleNoisePredictor`: 784 → 1024 → 784) to predict the noise
- Reverse step: `denoised = noisy_image − predicted_noise`

### 4. DDPM with UNet

Full denoising diffusion probabilistic model with a time-conditioned UNet:

**UNet Architecture:**
- Input conv → 2× downsampling (residual blocks + strided conv) → bottleneck → 2× upsampling with skip connections → output conv
- Time embedding: Linear(1 → 128) → SiLU → Linear → SiLU, projected and added at each scale
- Residual blocks use GroupNorm + SiLU activations

**Noise schedule:**

| Parameter | Value |
|---|---|
| Timesteps | 500 |
| β start | 0.0001 |
| β end | 0.02 |

| Training Hyperparameter | Value |
|---|---|
| Batch size | 64 |
| Epochs | 15 |
| LR | 1e-4 |
| Optimizer | AdamW (weight decay 0.01) |

## Requirements

```
torch
torchvision
pyyaml
matplotlib
```

## Usage

Train a model by running the corresponding trainer with a config file:

```bash
# VAE on MNIST
python trainer_vae.py --config configs/config_vae.yaml

# GAN on FashionMNIST
python trainer_gan.py --config configs/config_gan.yaml

# Diffusion on FashionMNIST
python trainer_diffusion.py --config configs/config_diffusion.yaml

# Simple single-image diffusion demo
python simple_diffusion.py
```

Outputs (generated images, loss curves) are saved to `./outputs/<model_name>/`.

## References

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Kingma & Welling, 2013
- [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — Goodfellow et al., 2014
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho et al., 2020
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger et al., 2015
