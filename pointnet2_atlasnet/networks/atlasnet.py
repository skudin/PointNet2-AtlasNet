import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class STN3d(nn.Module):
    def __init__(self, num_points=2500):
        super().__init__()

        self._num_points = num_points
        self._conv_1 = torch.nn.Conv1d(3, 64, 1)
        self._conv_2 = torch.nn.Conv1d(64, 128, 1)
        self._conv_3 = torch.nn.Conv1d(128, 1024, 1)

        self._fc_1 = nn.Linear(1024, 512)
        self._fc_2 = nn.Linear(512, 256)
        self._fc_3 = nn.Linear(256, 9)
        self._relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size()[0]

        x = F.relu(self._conv_1(x))
        x = F.relu(self._conv_2(x))
        x = F.relu(self._conv_3(x))

        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self._fc_1(x))
        x = F.relu(self._fc_2(x))
        x = self._fc_3(x)

        iden = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).view(1, 9).repeat(batch_size,
                                                                                                            1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x + iden
        x = x.view(-1, 3, 3)

        return x


class PointNetFeat(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, trans=False):
        super().__init__()

        self._num_points = num_points
        self._global_feat = global_feat
        self._trans = trans

        self._stn = STN3d(num_points=num_points)
        self._conv_1 = nn.Conv1d(3, 64, 1)
        self._conv_2 = nn.Conv1d(64, 128, 1)
        self._conv_3 = nn.Conv1d(128, 1024, 1)

        self._bn_1 = nn.BatchNorm1d(64)
        self._bn_2 = nn.BatchNorm1d(128)
        self._bn_3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        if self._trans:
            trans = self._stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)

        x = F.relu(self._bn_1(self._conv_1(x)))
        point_feat = x

        x = F.relu(self._bn_2(self._conv_2(x)))
        x = self._bn_3(self._conv_3(x))

        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)

        if self._trans:
            if self._global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self._num_points)
                return torch.cat([x, point_feat], 1), trans
        else:
            return x


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        super().__init__()

        self._bottleneck_size = bottleneck_size

        self._conv_1 = nn.Conv1d(self._bottleneck_size, self._bottleneck_size, 1)
        self._conv_2 = nn.Conv1d(self._bottleneck_size, self._bottleneck_size // 2, 1)
        self._conv_3 = nn.Conv1d(self._bottleneck_size // 2, self._bottleneck_size // 4, 1)
        self._conv_4 = nn.Conv1d(self._bottleneck_size // 4, 3, 1)

        self._bn_1 = nn.BatchNorm1d(self._bottleneck_size)
        self._bn_2 = nn.BatchNorm1d(self._bottleneck_size // 2)
        self._bn_3 = nn.BatchNorm1d(self._bottleneck_size // 4)

        self._th = nn.Tanh()

    def forward(self, x):
        x = F.relu(self._bn_1(self._conv_1(x)))
        x = F.relu(self._bn_2(self._conv_2(x)))
        x = F.relu(self._bn_3(self._conv_3(x)))
        x = self._th(self._conv_4(x))

        return x


class Encoder(nn.Module):
    def __init__(self, num_points=2048, bottleneck_size=1024):
        super().__init__()

        self._encoder = nn.Sequential(
            PointNetFeat(num_points, global_feat=True, trans=False),
            nn.Linear(1024, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self._encoder(x)


class Decoder(nn.Module):
    def __init__(self, num_points, num_primitives, bottleneck_size):
        super().__init__()

        self._num_points = num_points
        self._num_primitives = num_primitives
        self._bottleneck_size = bottleneck_size

        self._decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=2 + self._bottleneck_size) for _ in range(0, self._num_primitives)])

    def forward(self, x):
        outs = []
        for i in range(0, self._num_primitives):
            num_points_in_primitive = self._num_points // self._num_primitives
            outs.append(self._get_surface(x, i, num_points_in_primitive))

        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def inference(self, x, num_points=None, grid=None):
        if num_points is None or num_points < 1:
            num_points = self._num_points

        outs = []
        for i in range(0, self._num_primitives):
            if grid is None:
                num_points_in_primitive = num_points // self._num_primitives
                outs.append(self._get_surface(x, i, num_points_in_primitive=num_points_in_primitive))
            else:
                outs.append(self._get_surface(x, i, grid=grid))

        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def _get_surface(self, x: torch.Tensor, num_primitive, num_points_in_primitive=None, grid=None):
        if grid is None:
            batch_size = x.size(0)

            rand_grid = torch.cuda.FloatTensor(batch_size, 2, num_points_in_primitive)
            rand_grid.data.uniform_(0, 1)
        else:
            rand_grid = torch.cuda.FloatTensor(grid[num_primitive])
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()

        y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat((rand_grid, y), 1).contiguous()

        return self._decoder[num_primitive](y)


class SVR(nn.Module):
    def __init__(self, num_points: int = 2500, num_primitives: int = 1, bottleneck_size: int = 1024,
                 pretrained_encoder: bool = False):
        super().__init__()

        self._encoder = models.resnet18(pretrained=pretrained_encoder, num_classes=1024)
        self._decoder = Decoder(num_points=num_points, num_primitives=num_primitives,
                                bottleneck_size=bottleneck_size)

    def forward(self, x):
        x = x[:, : 3, :, :].contiguous()

        x = self._encoder(x)

        return self._decoder(x)

    def inference(self, x, num_points=None, grid=None):
        x = self._encoder(x)

        return self._decoder.inference(x, num_points, grid)

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @decoder.setter
    def decoder(self, value):
        self._decoder = value
