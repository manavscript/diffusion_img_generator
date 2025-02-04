import torch
from latent_diffusion import VAE
from model import UNet
from diffusion import generate_images

vae = VAE().cuda()
vae.load_state_dict(torch.load("vae_epoch_10.pth"))
vae.eval()

unet = UNet(img_channels=256).cuda()
unet.load_state_dict(torch.load("unet_epoch_10.pth"))
unet.eval()

def generate_latent_samples(num_samples=5, cfg_scale=3.0):
    z = torch.randn((num_samples, 256)).cuda()  # Latent noise
    samples = generate_images(unet, num_samples, None, cfg_scale)
    images = [vae.decode(sample).detach().cpu().numpy() for sample in samples]
    return images
