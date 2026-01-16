import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def show_images(x):
    x = x * 0.5 + 0.5
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

def make_grid(images, size=64):
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im

def make_transform(img_size):
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    def _transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": torch.stack(images)}
    return _transform

def train_model(model, noise_scheduler, train_loader, optimizer, epochs, device):
    losses = []
    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            clean_images = batch["images"].to(device)
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)
            losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
        if (epoch + 1) % 5 == 0:
            loss_last_epoch = sum(losses[-len(train_loader) :]) / len(train_loader)
            print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")  
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(losses)
    axs[1].plot(np.log(losses))
    plt.show()

def sample_images(model, noise_scheduler, device):
    sample = torch.randn(8, 3, 32, 32).to(device)
    for i, t in enumerate(noise_scheduler.timesteps):
        with torch.no_grad():
            residual = model(sample, t).sample
        sample = noise_scheduler.step(residual, t, sample).prev_sample
    return sample

def corrupt(x, amount):
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount

def visualize_noise(x):
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].set_title("Input Data")
    axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap="Greys")
    amount = torch.linspace(0, 1, x.shape[0])
    noised_x = corrupt(x, amount)
    axs[1].set_title('Corrupted Data')
    axs[1].imshow(torchvision.utils.make_grid(noised_x)[0], cmap='Greys')