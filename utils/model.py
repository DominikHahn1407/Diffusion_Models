import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn

class BasicUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])
        self.act = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)
    
    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))
            if i < 2:
                h.append(x) # skip connection
                x = self.downscale(x)
        for i, l in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)
                x += h.pop()
            x = self.act(l(x))
        return x

def corrupt(x, amount):
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount

def train_model(model, train_loader, optim, epochs, device):
    loss_fn = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            noise_amount = torch.rand(x.shape[0]).to(device)
            noisy_x = corrupt(x, noise_amount)
            pred = model(noisy_x)
            loss = loss_fn(pred, x)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        avg_loss = sum(losses[-len(train_loader):])/len(train_loader)
        print(f'Finished Epoch {epoch+1} --- Average Loss: {avg_loss:04f}')
    plt.plot(losses)
    plt.ylim(0, 0.1)

def make_predictions(model, train_loader, device):
    x, y = next(iter(train_loader))
    x = x[:8]
    amount = torch.linspace(0, 1, x.shape[0])
    noised_x = corrupt(x, amount)
    with torch.no_grad():
        preds = model(noised_x.to(device)).detach().cpu()
    fig, axs = plt.subplots(3, 1, figsize=(12, 7))
    axs[0].set_title('Input data')
    axs[0].imshow(torchvision.utils.make_grid(x)[0].clip(0, 1), cmap='Greys')
    axs[1].set_title('Corrupted data')
    axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].clip(0, 1), cmap='Greys')
    axs[2].set_title('Network Predictions')
    axs[2].imshow(torchvision.utils.make_grid(preds)[0].clip(0, 1), cmap='Greys')

def sample_images(model, device, n_steps=5):
    x = torch.rand(8, 1, 28, 28).to(device)
    step_history = [x.detach().cpu()]
    pred_output_history = []
    for i in range(n_steps):
        with torch.no_grad():
            pred = model(x)
        pred_output_history.append(pred.detach().cpu())
        mix_factor = 1 / (n_steps - i)
        x = x * (1 - mix_factor) + pred * mix_factor
        step_history.append(x.detach().cpu())
    fig, axs = plt.subplots(n_steps, 2, figsize=(9, 4), sharex=True)
    axs[0,0].set_title('X (Model Input)')
    axs[0,1].set_title('Model Prediction')
    for i in range(n_steps):
        axs[i, 0].imshow(torchvision.utils.make_grid(step_history[i])[0].clip(0, 1), cmap='Greys')
        axs[i, 1].imshow(torchvision.utils.make_grid(pred_output_history[i])[0].clip(0, 1), cmap='Greys')