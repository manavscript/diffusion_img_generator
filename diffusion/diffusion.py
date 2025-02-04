import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoiseScheduler():
    def __init__(self, beta_1, beta_end, timesteps, device):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_1, beta_end, timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def forward_noising(self, x_0):
        """
        Adds noise to x_0 at a random timestep t.
        Returns: (noisy image x_t, timestep t, actual noise ε)
        """
        t = torch.randint(1, self.timesteps+1, (1,), device=self.device).item()
        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(self.alphas_bar[t-1]) * x_0 + torch.sqrt(1 - self.alphas_bar[t-1]) * noise
        return x_t, t, noise


def compute_loss(model, x_0, noise_scheduler):
    loss_fn = torch.nn.functional.mse_loss

    x_t, t, noise = noise_scheduler.forward_noising(x_0)

    predicted_noise = model(x_t, t)

    # ✅ Ensure predicted_noise matches noise in shape
    if predicted_noise.shape != noise.shape:
        print(f"⚠️ Shape Mismatch: predicted_noise {predicted_noise.shape}, noise {noise.shape}")
        predicted_noise = F.interpolate(predicted_noise, size=noise.shape[2:], mode='bilinear', align_corners=False)

    return loss_fn(predicted_noise, noise)



@torch.no_grad()
def sample_images(model, noise_scheduler, img_shape, num_samples=16, classifier_free_guidance=False, guidance_scale=3.0):
    """
    Generates images from pure noise using reverse diffusion.
    classifier_free_guidance: If True, enables guidance with an unconditional model.
    """
    device = next(model.parameters()).device
    x_t = torch.randn((num_samples, *img_shape), device=device)  # Start from Gaussian noise

    for t in reversed(range(noise_scheduler.timesteps)):
        t_tensor = torch.full((num_samples,), t, device=device).long()
        predicted_noise = model(x_t, t_tensor)

        # Optional: Apply classifier-free guidance
        if classifier_free_guidance:
            with torch.no_grad():
                unconditional_pred = model(x_t, t_tensor * 0)  # Model with t=0 (no conditions)
            predicted_noise = unconditional_pred + guidance_scale * (predicted_noise - unconditional_pred)

        alpha_t = noise_scheduler.alphas[t]
        alpha_bar_t = noise_scheduler.alphas_bar[t]
        beta_t = noise_scheduler.betas[t]

        # Compute the denoised image estimate
        mu_theta = (1 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise)

        # Add stochasticity
        if t > 0:
            z = torch.randn_like(x_t, device=device)
            sigma_t = torch.sqrt(beta_t)
            x_t = mu_theta + sigma_t * z
        else:
            x_t = mu_theta  # Final clean image

    return x_t  # Final generated images
