import torch
import torch.optim as optim

from atlasnet2.networks.autoencoder import Autoencoder


class Network:
    def __init__(self, encoder_type: str = "pointnet", num_points: int = 2500, num_primitives: int = 1,
                 bottleneck_size=1024, learning_rate: float = 0.001):
        self._network = Autoencoder(encoder_type=encoder_type, num_points=num_points, num_primitives=num_primitives,
                                    bottleneck_size=bottleneck_size)
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
