import logging

import torch
from torch.utils.data import DataLoader

from atlasnet2.datasets.shapenet_dataset import ShapeNetDataset
from atlasnet2.networks.network import Network
from atlasnet2.libs.helpers import AverageValueMeter

import dist_chamfer


logger = logging.getLogger(__name__)


class NetworkWrapper:
    def __init__(self, mode: str, dataset_path: str, num_epochs: int, num_points: int, batch_size: int,
                 num_workers: int):
        self._mode = mode
        self._dataset_path = dataset_path
        self._num_epochs = num_epochs
        self._num_points = num_points
        self._batch_size = batch_size
        self._num_workers = num_workers

        self._train_data_loader = self._get_data_loader("train")
        self._test_data_loader = self._get_data_loader("test")

        self._network = Network()

        self._loss_func = dist_chamfer.chamferDist()

        self._train_loss = AverageValueMeter()
        self._test_loss = AverageValueMeter()

        self._train_curve = []
        self._test_curve = []

    def train(self):
        for epoch in range(self._num_epochs):
            self._train_epoch(epoch)
            self._test_epoch(epoch)
            self._save_snapshot(epoch)
            self._print_epoch_stat(epoch)

    def test(self):
        pass

    def _get_data_loader(self, dataset_part: str = "test"):
        logger.info("\nInitializing data loader. Mode: %s, dataset part: %s.\n" % (self._mode, dataset_part))

        if self._mode == "train":
            if dataset_part == "train":
                return DataLoader(
                    dataset=ShapeNetDataset(dataset_path=self._dataset_path, mode="train", num_points=self._num_points),
                    batch_size=self._batch_size,
                    shuffle=True,
                    num_workers=self._num_workers
                )
            else:
                return DataLoader(
                    dataset=ShapeNetDataset(dataset_path=self._dataset_path, mode="test", num_points=self._num_points),
                    batch_size=self._batch_size,
                    shuffle=False,
                    num_workers=self._num_workers
                )
        else:
            return DataLoader(
                dataset=ShapeNetDataset(dataset_path=self._dataset_path, mode="test", num_points=self._num_points),
                batch_size=1,
                shuffle=False,
                num_workers=1
            )

    def _train_epoch(self, epoch):
        self._train_loss.reset()
        self._network.set_train_mode()

        for batch_num, batch_data in enumerate(self._train_data_loader, 1):
            reconstructed_point_clouds = self._network.forward(batch_data)

            dist_1, dist_2 = self._loss_func(batch_data, reconstructed_point_clouds)
            loss = torch.mean(dist_1) + torch.mean(dist_2)
            self._network.backward(loss)

            loss_value = loss.item()
            self._train_loss.update(loss_value)

        self._train_curve.append(self._train_loss.avg)

    def _test_epoch(self, epoch):
        self._test_loss.reset()
        self._network.set_test_mode()

        with torch.no_grad():
            for batch_num, batch_data in enumerate(self._test_data_loader, 1):
                reconstructed_point_clouds = self._network.forward(batch_data)

                dist_1, dist_2 = self._loss_func(batch_data, reconstructed_point_clouds)
                loss = torch.mean(dist_1) + torch.mean(dist_2)

                loss_value = loss.item()
                self._test_loss.update(loss_value)

            self._test_curve.append(self._test_loss.avg)

    def _save_snapshot(self, epoch):
        pass

    def _print_epoch_stat(self, epoch):
        pass
