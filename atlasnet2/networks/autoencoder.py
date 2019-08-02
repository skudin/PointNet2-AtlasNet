from typing import Union

import torch
import torch.nn as nn

import atlasnet2.networks.atlasnet as atlasnet
import atlasnet2.networks.pointnet2 as pointnet2


class Autoencoder(nn.Module):
    def __init__(self, encoder_type: str = "pointnet", num_points: int = 2500, num_primitives: int = 1,
                 bottleneck_size: int = 1024):
        super().__init__()

        if encoder_type == "pointnet":
            self._encoder = atlasnet.Encoder(num_points=num_points, bottleneck_size=bottleneck_size)
        else:
            self._decoder = pointnet2.Encoder()

        self._decoder = atlasnet.Decoder(num_points=num_points, num_primitives=num_primitives,
                                         bottleneck_size=bottleneck_size)

    def forward(self, x: torch.Tensor):
        x = self._encoder.forward(x)

        return self._decoder.forward(x)

    def inference(self, x: torch.Tensor, num_point: Union[int, None] = None):
        with torch.no_grad():
            x = self._encoder.forward(x)

            return self._decoder.inference(x, num_point)
