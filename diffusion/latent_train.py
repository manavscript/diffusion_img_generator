import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from latent_diffusion import VAE
from model import UNet
from diffusion import NoiseScheduler, compute_loss
import utils
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Train a Latent Diffusion Model")
    parser.add_argument("--dataset", type=str, default="celeba", choices=["cifar10", "stl10", "celeba"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()

def train(args):
    train_set, val_set, _ = utils.get_celeba() if args.dataset == "celeba" else utils.get_cifar10()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    vae = VAE().cuda()
    unet = UNet(img_channels=256).cuda()
    optimizer_vae = optim.AdamW(vae.parameters(), lr=args.lr)
    optimizer_unet = optim.AdamW(unet.parameters(), lr=args.lr)
    noise_scheduler = NoiseScheduler(1e-4, 0.02, 1000, "cuda")

    for epoch in range(args.epochs):
        vae.train()
        unet.train()
        for x_0, _ in tqdm(train_loader):
            x_0 = x_0.cuda()

            # Train VAE (latent encoding)
            optimizer_vae.zero_grad()
            x_recon, mu, log_var = vae(x_0)
            recon_loss = torch.nn.functional.mse_loss(x_recon, x_0)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss_vae = recon_loss + 0.001 * kl_loss  # Balance reconstruction vs. regularization
            loss_vae.backward()
            optimizer_vae.step()

            # Train UNet in Latent Space
            optimizer_unet.zero_grad()
            z = vae.encode(x_0)  # Get latent representation
            loss_unet = compute_loss(unet, z, noise_scheduler)
            loss_unet.backward()
            optimizer_unet.step()

        # Save models
        torch.save(vae.state_dict(), f"vae_epoch_{epoch+1}.pth")
        torch.save(unet.state_dict(), f"unet_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    args = get_args()
    train(args)
