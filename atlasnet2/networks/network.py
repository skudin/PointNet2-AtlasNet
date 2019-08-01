import torch
import torch.optim as optim

from atlasnet2.networks.autoencoder import Autoencoder


class Network:
    def __init__(self, encoder_type: str = "pointnet", learning_rate: float = 0.001):
        self._network = Autoencoder(encoder_type)
        self._optimizer = optim.Adam(self._network.parameters(), lr=learning_rate)

    def forward(self, x: torch.Tensor):
        return self._network(x)

    def backward(self, loss):
        loss.backward()
        self._optimizer.step()

    def set_train_mode(self):
        self._network.train()

    def set_test_mode(self):
        self._network.eval()
