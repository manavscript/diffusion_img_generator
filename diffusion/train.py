import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import utils
from torch.utils.data import DataLoader
from diffusion import NoiseScheduler, compute_loss
from model import UNet
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Train a DDPM Model")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "stl10"],
                        help="Dataset to use: cifar10 or stl10")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--val_batch_size", type=int, default=128, help="Validation batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--betas", type=float, nargs=2, default=[1e-4, 0.02], help="Noise schedule (beta_1, beta_end)")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume training from checkpoint")

    return parser.parse_args()

def train(args):
    if args.dataset == "cifar10":
        train_set, val_set, _ = utils.get_cifar10()
    elif args.dataset == "stl10":
        train_set, val_set, _ = utils.get_stl10()
    else:
        raise ValueError("Invalid dataset choice.")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device(args.device)
    model = UNet().to(device)
    noise_scheduler = NoiseScheduler(args.betas[0], args.betas[1], args.timesteps, device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed training from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for x_0, _ in progress_bar:
            x_0 = x_0.to(device)
            optimizer.zero_grad()

            loss = compute_loss(model, x_0, noise_scheduler)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

        os.makedirs(args.save_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    print("Training Complete!")

if __name__ == "__main__":
    args = get_args()
    train(args)
