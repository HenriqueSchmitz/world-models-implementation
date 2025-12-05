import torch

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(torch.nn.Module):
    def __init__(self, size):
        super(UnFlatten, self).__init__()
        self.size = size

    def forward(self, input):
        return input.view(input.size(0), self.size, 1, 1)
    
class ConvVAE(torch.nn.Module):
    def __init__(self, image_channels, h_dim, z_dim, device="cpu"):
        super(ConvVAE, self).__init__()
        self.device = device
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            Flatten()
        ).to(self.device)

        self.fc1 = torch.nn.Linear(h_dim, z_dim).to(self.device)
        self.fc2 = torch.nn.Linear(h_dim, z_dim).to(self.device)
        self.fc3 = torch.nn.Linear(z_dim, h_dim).to(self.device)

        self.decoder = torch.nn.Sequential(
            UnFlatten(size=h_dim),
            torch.nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            torch.nn.Sigmoid(),
        ).to(self.device)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(self.device)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
    
    def to(self, device):
        super().to(device)
        self.device = device
        return self