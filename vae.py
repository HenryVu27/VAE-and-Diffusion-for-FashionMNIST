import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
  def __init__(self, hidden_dim, latent_dim, class_emb_dim, num_classes=10):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.latent_dim = latent_dim
    self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())

    self.mu_net = nn.Linear(hidden_dim, latent_dim) 
    self.logvar_net = nn.Linear(hidden_dim, latent_dim)
    self.class_embedding = nn.Embedding(num_classes, class_emb_dim) 

    self.decoder = nn.Sequential(
            nn.Linear(latent_dim + class_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28*28),
            nn.Sigmoid()) 

  def forward(self, x: torch.Tensor, y: torch.Tensor):
    """
    Args:
        x (torch.Tensor): image [B, 1, 28, 28]
        y (torch.Tensor): labels [B]
        
    Returns:
        reconstructed: image [B, 1, 28, 28]
        mu: [B, latent_dim]
        logvar: [B, latent_dim]
    """
    
    encoded = self.encoder(x)
    mu = self.mu_net(encoded)
    logvar = self.logvar_net(encoded)
    z = self.reparameterize(mu, logvar)
        
    class_emb = self.class_embedding(y)
    z = torch.cat((z, class_emb), dim=1)
    reconstructed = self.decoder(z)
    reconstructed = reconstructed.view(-1, 1, 28, 28)  
    return reconstructed, mu, logvar

  def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
    """
    applies the reparameterization trick
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    new_sample = mu+std*eps

    return new_sample

  def kl_loss(self, mu, logvar):
    """
    calculates the KL divergence between a normal distribution with mean "mu" and
    log-variance "logvar" and the standard normal distribution (mean=0, var=1)
    """
    kl_div = 0.5*torch.sum(mu.pow(2)+logvar.exp()-1-logvar, dim=1)
    
    return kl_div

  def get_loss(self, x: torch.Tensor, y: torch.Tensor):
    """
    given the image x, and the label y calculates the prior loss and reconstruction loss
    """
    reconstructed, mu, logvar = self.forward(x, y)
    
    # reconstruction loss
    recons_loss = F.binary_cross_entropy(reconstructed.view(-1, 784), x.view(-1, 784), reduction='mean')  # prior matching loss
    prior_loss = self.kl_loss(mu, logvar).mean()
    return recons_loss, prior_loss

  @torch.no_grad()
  def generate_sample(self, num_images: int, y, device):
    """
    generates num_images samples by passing noise to the model's decoder
    if y is not None (e.g., y = torch.tensor([1, 2, 3]).to(device)) the model
    generates samples according to the specified labels
    
    Returns:
        samples: [num_images, 1, 28, 28]
    """
    
    # sample from noise, find the class embedding and use both in the decoder to generate new samples
    z = torch.randn(num_images, self.latent_dim).to(device)
    if y is not None:
        y = y.to(device)
        class_embeddings = self.class_embedding(y)
        z = torch.cat((z, class_embeddings), dim=1)
    samples = self.decoder(z).view(num_images, 1, 28, 28)
    return samples


def load_vae_and_generate():
    device = torch.device('cuda')
    num_classes = 10
    hidden_dim = 400
    latent_dim = 20
    class_emb_dim = 10
    vae = VAE(hidden_dim=hidden_dim, latent_dim=latent_dim, class_emb_dim=class_emb_dim, num_classes=num_classes)
    
    vae.load_state_dict(torch.load('vae.pt'))
    vae = vae.to(device)
    
    desired_labels = []
    for i in range(10):
        for _ in range(5):
            desired_labels.append(i)

    desired_labels = torch.tensor(desired_labels).to(device)
    generated_samples = vae.generate_sample(50, desired_labels, device)
    
    return generated_samples