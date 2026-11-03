"""Plain autoencoder used as the dissimilarity (OOD) score model.

The per-sample reconstruction error of the autoencoder is used as a
"dissimilarity score": large error = the sample is unlike the data the AE was trained on.
"""
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data


class Encoder(nn.Module):
    """Encodes an input into the latent space (sigmoid MLP)."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, device="cpu"):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [100, 50]
        if latent_dim is None:
            latent_dim = 20

        layers = []
        architecture = [input_dim] + hidden_dims + [latent_dim]
        for in_dim, out_dim in zip(architecture[:-1], architecture[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Sigmoid())
        self.encoder = nn.Sequential(*layers).to(device)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_tensor)


class Decoder(nn.Module):
    """Decodes a latent representation back to the input space (sigmoid MLP)."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, device="cpu"):
        super().__init__()
        architecture = [latent_dim] + hidden_dims + [input_dim]
        layers = []
        for in_dim, out_dim in zip(architecture[:-1], architecture[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers).to(device)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.decoder(input_tensor)


class AEModule(nn.Module):
    """Encoder + decoder. `forward` returns the per-sample reconstruction MSE."""

    def __init__(self, input_size: int, hidden_dims: List[int], latent_dim: int, device="cpu"):
        super().__init__()
        self.z_dim = latent_dim
        self.encoder = Encoder(input_size, hidden_dims, latent_dim, device)
        self.decoder = Decoder(input_size, hidden_dims, latent_dim, device)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor.float()
        z = self.encoder(input_tensor)
        reconstruction = self.decoder(z)
        mse = nn.MSELoss(reduction="none")
        return mse(reconstruction, input_tensor).mean(dim=1)

    def train(self):
        self.encoder.encoder.train()
        self.decoder.decoder.train()

    def eval(self):
        self.encoder.encoder.eval()
        self.decoder.decoder.eval()


class AE:
    """Autoencoder wrapper that handles training and scoring.

    Parameters
    ----------
    hidden_sizes : sizes of the hidden layers (default [100, 50]).
    input_size   : dimensionality of the input.
    latent_dim   : size of the latent space (default 20).
    device       : e.g. 'cpu' or 'cuda'.
    lr           : Adam learning rate.
    """

    def __init__(self, hidden_sizes: List[int], input_size: int, latent_dim: int,
                 device="cpu", lr: float = 1e-3):
        if hidden_sizes is None:
            hidden_sizes = [100, 50]
        if latent_dim is None:
            latent_dim = 20

        self.model = AEModule(input_size=input_size, hidden_dims=hidden_sizes,
                              latent_dim=latent_dim, device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

    def train(self, X_train: torch.Tensor, batch_size: int = 32, n_epochs: int = 5):
        """Train on a float tensor `X_train`."""
        print("Start training ...")
        train_data = torch.utils.data.DataLoader(X_train.float(), batch_size=batch_size)

        for epoch in range(n_epochs):
            self.model.train()
            total, i = 0.0, 0
            for i, batch in enumerate(train_data):
                batch = batch.to(self.device)
                reconstruction_error = self.model(batch).mean()
                total += reconstruction_error.item()

                self.optimizer.zero_grad()
                reconstruction_error.backward()
                self.optimizer.step()

            print("Epoch:", epoch, "Average reconstruction error:", total / (i + 1))

    @torch.no_grad()
    def dissimilarity_score(self, X) -> np.ndarray:
        """Per-sample reconstruction error (higher = more dissimilar from the training data)."""
        self.model.eval()
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        return self.model(X).cpu().numpy()
