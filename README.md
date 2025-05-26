# VAE and Diffusion Models for FashionMNIST

Implementation of Variational Autoencoder (VAE) and Diffusion models for FashionMNIST dataset generation.

## Requirements
```bash
pip install torch torchvision matplotlib numpy
```

## Usage

Train VAE model:
```bash
python main.py -vae
```

Train Diffusion model:
```bash
python main.py -diffusion
```

Train both models:
```bash
python main.py -vae -diffusion
```

### Optional Arguments
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 128)
- `--lr_vae`: VAE learning rate (default: 1e-3)
- `--lr_diffusion`: Diffusion learning rate (default: 1e-4)

Example:
```bash
python main.py -vae --epochs 100 --batch_size 64 --lr_vae 5e-4
```

## Output
- Model checkpoints: `vae.pt`, `diffusion.pt`
- Generated samples: `vae_samples.png`, `diffusion_samples.png`
- Training loss plots: `vae_training_loss.png`, `diffusion_training_loss.png` 