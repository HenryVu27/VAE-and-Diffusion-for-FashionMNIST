import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from vae import VAE
from diffusion import DiffusionModel, NoiseEstimatingNet, VarianceScheduler
import argparse

def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader

def train_vae(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        
        recons_loss, prior_loss = model.get_loss(data, labels)
        loss = recons_loss + prior_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    return total_loss / len(train_loader)

def train_diffusion(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        
        loss = model(data, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    return total_loss / len(train_loader)

def plot_samples(model, device, num_samples=10, model_type='vae'):
    model.eval()
    with torch.no_grad():
        if model_type == 'vae':
            # Generate samples for each class
            samples = []
            for i in range(10):
                labels = torch.full((num_samples,), i, device=device)
                sample = model.generate_sample(num_samples, labels, device)
                samples.append(sample)
            samples = torch.cat(samples, dim=0)
        else:  # diffusion
            # Generate samples for each class
            samples = []
            for i in range(10):
                labels = torch.full((num_samples,), i, device=device)
                sample = model.generate_sample(num_samples, labels, device)
                samples.append(sample)
            samples = torch.cat(samples, dim=0)
        
        # Plot samples
        fig, axes = plt.subplots(10, num_samples, figsize=(num_samples, 10))
        for i in range(10):
            for j in range(num_samples):
                idx = i * num_samples + j
                axes[i, j].imshow(samples[idx].cpu().squeeze(), cmap='gray')
                axes[i, j].axis('off')
        plt.tight_layout()
        plt.savefig(f'{model_type}_samples.png')
        plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate VAE or Diffusion models on FashionMNIST')
    parser.add_argument('-vae', action='store_true', help='Train and evaluate VAE model')
    parser.add_argument('-diffusion', action='store_true', help='Train and evaluate Diffusion model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
    parser.add_argument('--lr_vae', type=float, default=1e-3, help='Learning rate for VAE (default: 1e-3)')
    parser.add_argument('--lr_diffusion', type=float, default=1e-4, help='Learning rate for Diffusion (default: 1e-4)')
    args = parser.parse_args()

    # Check if at least one model is selected
    if not (args.vae or args.diffusion):
        parser.error("Please select at least one model to train (-vae or -diffusion)")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size)
    
    # Train VAE if selected
    if args.vae:
        print("\nInitializing VAE...")
        vae = VAE(hidden_dim=400, latent_dim=20, class_emb_dim=10, num_classes=10).to(device)
        vae_optimizer = optim.Adam(vae.parameters(), lr=args.lr_vae)
        
        print("Training VAE...")
        vae_losses = []
        for epoch in range(1, args.epochs + 1):
            loss = train_vae(vae, train_loader, vae_optimizer, device, epoch)
            vae_losses.append(loss)
            if epoch % 5 == 0:
                plot_samples(vae, device, model_type='vae')
                torch.save(vae.state_dict(), 'vae.pt')
        
        # Plot VAE training loss
        plt.figure(figsize=(10, 5))
        plt.plot(vae_losses, label='VAE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('vae_training_loss.png')
        plt.close()
    
    # Train Diffusion if selected
    if args.diffusion:
        print("\nInitializing Diffusion Model...")
        noise_net = NoiseEstimatingNet(n_steps=1000, time_emb_dim=100, class_emb_dim=100, num_classes=10).to(device)
        var_scheduler = VarianceScheduler(beta_start=0.0001, beta_end=0.02, num_steps=1000)
        diffusion = DiffusionModel(noise_net, var_scheduler).to(device)
        diffusion_optimizer = optim.Adam(diffusion.parameters(), lr=args.lr_diffusion)
        
        print("Training Diffusion Model...")
        diffusion_losses = []
        for epoch in range(1, args.epochs + 1):
            loss = train_diffusion(diffusion, train_loader, diffusion_optimizer, device, epoch)
            diffusion_losses.append(loss)
            if epoch % 5 == 0:
                plot_samples(diffusion, device, model_type='diffusion')
                torch.save(diffusion.state_dict(), 'diffusion.pt')
        
        # Plot Diffusion training loss
        plt.figure(figsize=(10, 5))
        plt.plot(diffusion_losses, label='Diffusion Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('diffusion_training_loss.png')
        plt.close()

if __name__ == '__main__':
    main() 