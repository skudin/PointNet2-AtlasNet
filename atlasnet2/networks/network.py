import torch
import torch.nn as nn

import atlasnet2.networks.atlasnet as atlasnet
import atlasnet2.networks.pointnet2 as pointnet2


class Network(nn.Module):
    def __init__(self, encoder_type: str = "pointnet"):
        super().__init__()

        if encoder_type == "pointnet":
            self._encoder = atlasnet.Encoder()
        else:
            self._decoder = pointnet2.Encoder()

        self._decoder = atlasnet.Decoder()

    def forward(self, x):
        x = self._encoder.forward(x)

        return self._decoder.forward(x)

    def inference(self, x, num_point=None):
        with torch.no_grad():
            x = self._encoder.forward(x)
    
            return self._decoder.inference(x, num_point)

    def backward(self, loss):
        pass

    def set_train_mode(self):
        pass

    def set_test_mode(self):
        pass
