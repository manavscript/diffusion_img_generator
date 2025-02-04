# 🎨 Latent Diffusion Model (LDM) for Image Generation

![Latent Diffusion](https://upload.wikimedia.org/wikipedia/commons/9/94/Stable_Diffusion_Logo.png)  

## **🚀 Overview**
This project implements a **Latent Diffusion Model (LDM)** for **high-quality image generation**.  
The model **compresses images into a latent space** using a **Variational Autoencoder (VAE)**, and then **trains a diffusion model** on the latent representations.  
This makes **training more efficient** than standard pixel-based diffusion models like DDPM.  

---

## **✨ Features**
✅ **Latent Diffusion Training** (More Efficient than Pixel-Space DDPMs)  
✅ **Classifier-Free Guidance (CFG)** for Controllable Image Generation  
✅ **Multi-Dataset Support** (CIFAR-10, STL-10, CelebA)  
✅ **Trained UNet Model for Diffusion in Latent Space**  
✅ **Interactive Gradio Deployment**  

---

## **📦 Installation**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/manavscript/diffusion_img_generator
cd diffusion
```

### **2️⃣ Install Dependencies**
```bash
pip install torch torchvision gradio tqdm numpy matplotlib
```

---

<!-- ## **🛠️ Usage**
### **1️⃣ Train the Model**
Train the **Latent Autoencoder (VAE) and UNet Diffusion Model**:
```bash
python latent_train.py --dataset celeba --epochs 20 --batch_size 64 --lr 2e-4
```
- The **VAE** encodes images into a **compressed latent space**.
- The **UNet Diffusion Model** learns to generate images in latent space.

---

### **2️⃣ Generate Samples**
Generate images using the trained **Latent Diffusion Model**:
```bash
python latent_sample.py --num_samples 5
```
This will **decode generated latent representations** into full-resolution images.

---

### **3️⃣ Deploy with Gradio**
Launch a **web-based demo** using **Gradio**:
```bash
python deploy.py
```
- Opens an interactive **GUI** for real-time image generation.
- Hosted **locally** or can be deployed on **Hugging Face Spaces**.

--- -->

<!-- ## **🎨 Sample Results**
| Input Noise | Generated Image |
|------------|---------------|
| ![Noise](https://via.placeholder.com/128) | ![Generated Image](https://via.placeholder.com/128) |

--- -->

## **🔮 Future Improvements**
🚀 **Train on Larger Datasets (FFHQ, ImageNet)**  
🚀 **Improve UNet Architecture for Higher-Resolution Images**  
🚀 **Deploy the Model as an API (FastAPI or Flask)**  
🚀 **Fine-Tune Classifier-Free Guidance for Style Transfer**  

---

## **🙏 Acknowledgments**
This project is inspired by:  
- **Stable Diffusion** ([CompVis](https://github.com/CompVis/stable-diffusion))  
- **DDPM Paper** ([Ho et al., 2020](https://arxiv.org/abs/2006.11239))  
- **Latent Diffusion Models** ([Rombach et al., 2022](https://arxiv.org/abs/2112.10752))  

---

