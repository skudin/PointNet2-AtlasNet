import torch
import torch.optim as optim

from atlasnet2.networks.autoencoder import Autoencoder
import atlasnet2.libs.helpers as h


class Network:
    def __init__(self, encoder_type: str = "pointnet", num_points: int = 2500, num_primitives: int = 1,
                 bottleneck_size=1024, learning_rate: float = 0.001):
        self._network = Autoencoder(encoder_type=encoder_type, num_points=num_points, num_primitives=num_primitives,
                                    bottleneck_size=bottleneck_size)
        self._network.apply(h.weights_init)
        self._network.cuda()

        self._optimizer = optim.Adam(self._network.parameters(), lr=learning_rate)

    def forward(self, tensor: torch.Tensor):
        self._optimizer.zero_grad()

        tensor = tensor.transpose(2, 1).contiguous()
        tensor = tensor.cuda()

        return self._network(tensor)

    def backward(self, loss):
        loss.backward()
        self._optimizer.step()

    def set_train_mode(self):
        self._network.train()

    def set_test_mode(self):
        self._network.eval()
