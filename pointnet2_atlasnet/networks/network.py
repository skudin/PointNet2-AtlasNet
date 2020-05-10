import logging
from typing import Optional

import torch
import torch.optim as optim

from pointnet2_atlasnet.networks.autoencoder import Autoencoder
from pointnet2_atlasnet.networks.atlasnet import SVR
import pointnet2_atlasnet.libs.helpers as h


logger = logging.getLogger(__name__)


class Network:
    def __init__(self, svr: bool = False, encoder_type: str = "pointnet", pretrained_ae: Optional[str] = None,
                 num_points: int = 2500, num_primitives: int = 1, bottleneck_size: int = 1024,
                 learning_rate: float = 0.001):
        self._svr = svr
        self._pretrained_ae = pretrained_ae

        if self._svr:
            self._network = SVR(num_points=num_points, num_primitives=num_primitives, bottleneck_size=bottleneck_size)
        else:
            self._network = Autoencoder(encoder_type=encoder_type, num_points=num_points, num_primitives=num_primitives,
                                        bottleneck_size=bottleneck_size)

        self._network.apply(h.weights_init)

        if self._svr and self._pretrained_ae is not None:
            ae = Autoencoder(encoder_type=encoder_type, num_points=num_points, num_primitives=num_primitives,
                             bottleneck_size=bottleneck_size)
            ae.load_state_dict(torch.load(self._pretrained_ae))

            self._network.decoder = ae.decoder

            del ae
            pass

        self._network.cuda()

        if self._svr and self._pretrained_ae is not None:
            params = self._network.encoder.parameters()
        else:
            params = self._network.parameters()

        self._optimizer = optim.Adam(params, lr=learning_rate)

        logger.info("Network architecture:")
        logger.info(str(self._network))

    def forward(self, tensor: torch.Tensor):
        self._optimizer.zero_grad()

        if not self._svr:
            tensor = tensor.transpose(2, 1).contiguous()
        tensor = tensor.cuda()

        return self._network(tensor)

    def inference(self, tensor: torch.Tensor, num_points: Optional[int] = None):
        if not self._svr:
            tensor = tensor.transpose(2, 1).contiguous()
        tensor = tensor.cuda()

        return self._network.inference(tensor, num_points)

    def backward(self, loss):
        loss.backward()
        self._optimizer.step()

    def set_train_mode(self):
        self._network.train()

    def set_test_mode(self):
        self._network.eval()

    def reset_optimizer(self, learning_rate):
        if self._svr and self._pretrained_ae is not None:
            params = self._network.encoder.parameters()
        else:
            params = self._network.parameters()

        self._optimizer = optim.Adam(params, lr=learning_rate)

    def save_snapshot(self, path: str):
        torch.save(self._network.state_dict(), path)

    def load_snapshot(self, snapshot: str):
        self._network.load_state_dict(torch.load(snapshot))
        logger.info("Snapshot %s loaded!" % snapshot)
