import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from atlasnet2.datasets.shapenet_dataset import ShapeNetDataset
from atlasnet2.networks.network import Network
from atlasnet2.libs.helpers import AverageValueMeter

import dist_chamfer
import atlasnet2.configuration as conf
from atlasnet2.libs.visdom_wrapper import VisdomWrapper


logger = logging.getLogger(__name__)


class NetworkWrapper:
    def __init__(self, mode: str, vis: VisdomWrapper, dataset_path: str, num_epochs: int, batch_size: int,
                 num_workers: int, encoder_type: str, num_points: int, num_primitives: int, bottleneck_size: int,
                 learning_rate: float):
        self._mode = mode
        self._vis = vis
        self._dataset_path = dataset_path
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._num_points = num_points

        self._train_data_loader = self._get_data_loader("train")
        self._test_data_loader = self._get_data_loader("test")

        self._network = Network(encoder_type=encoder_type, num_points=self._num_points, num_primitives=num_primitives,
                                bottleneck_size=bottleneck_size, learning_rate=learning_rate)

        self._loss_func = dist_chamfer.chamferDist()

        self._train_loss = AverageValueMeter()
        self._test_loss = AverageValueMeter()

    def train(self):
        logger.info("Training started!")

        for epoch in range(self._num_epochs):
            self._train_epoch(epoch)
            self._test_epoch(epoch)
            self._show_graphs()
            # self._save_snapshot(epoch)
            # self._print_epoch_stat(epoch)

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

        for batch_num, point_clouds in enumerate(self._train_data_loader, 1):
            reconstructed_point_clouds = self._network.forward(point_clouds)

            dist_1, dist_2 = self._loss_func(point_clouds.cuda(), reconstructed_point_clouds)
            loss = torch.mean(dist_1) + torch.mean(dist_2)
            self._network.backward(loss)

            loss_value = loss.item()
            self._train_loss.update(loss_value)

            if batch_num % conf.VISDOM_UPDATE_FREQUENCY:
                self._vis.show_point_cloud("REAL TRAIN", point_clouds)
                self._vis.show_point_cloud("FAKE TRAIN", reconstructed_point_clouds)

            logger.info(
                "[%d: %d/%d] train chamfer loss: %f " % (epoch, batch_num, len(self._train_data_loader), loss_value))

        self._vis.append_point_to_curve("Chamfer loss", "train", epoch, self._train_loss.avg)
        self._vis.append_point_to_curve("Chamfer log loss", "train", epoch, np.log(self._train_loss.avg))

    def _test_epoch(self, epoch):
        self._test_loss.reset()
        self._network.set_test_mode()

        with torch.no_grad():
            for batch_num, point_clouds in enumerate(self._test_data_loader, 1):
                reconstructed_point_clouds = self._network.forward(point_clouds)

                dist_1, dist_2 = self._loss_func(point_clouds.cuda(), reconstructed_point_clouds)
                loss = torch.mean(dist_1) + torch.mean(dist_2)

                loss_value = loss.item()
                self._test_loss.update(loss_value)

                if batch_num % conf.VISDOM_UPDATE_FREQUENCY:
                    self._vis.show_point_cloud("REAL TEST", point_clouds)
                    self._vis.show_point_cloud("FAKE TEST", reconstructed_point_clouds)

                logger.info(
                    "[%d: %d/%d] test chamfer loss: %f " % (epoch, batch_num, len(self._test_data_loader), loss_value))

            self._vis.append_point_to_curve("Chamfer loss", "test", epoch, self._test_loss.avg)
            self._vis.append_point_to_curve("Chamfer log loss", "test", epoch, np.log(self._test_loss.avg))

    def _show_graphs(self):
        self._vis.show_graph("Chamfer loss")
        self._vis.show_graph("Chamfer log loss")

    def _save_snapshot(self, epoch):
        pass

    def _print_epoch_stat(self, epoch):
        pass
