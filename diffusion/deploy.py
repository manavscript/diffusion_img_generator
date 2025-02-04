import torch
import gradio as gr
from model import UNet
from diffusion import generate_images

model = UNet().cuda()
model.load_state_dict(torch.load("model_epoch_10.pth"))
model.eval()

def generate_sample(label):
    img = generate_images(model, 1, num_classes=10, cfg_scale=3.0)[0]
    return img.permute(1, 2, 0).cpu().numpy()

gr.Interface(fn=generate_sample, inputs="dropdown", outputs="image").launch()
