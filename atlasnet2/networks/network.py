import torch
import torch.optim as optim

from atlasnet2.networks.autoencoder import Autoencoder


class Network:
    def __init__(self, encoder_type: str = "pointnet", learning_rate=0.001):
        self._network = Autoencoder(encoder_type)
        self._optimizer = optim.Adam(self._network.parameters(), lr=learning_rate)

    def backward(self, loss):
        pass

    def set_train_mode(self):
        pass

    def set_test_mode(self):
        pass
