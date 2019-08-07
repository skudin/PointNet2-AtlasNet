import logging
import os
import time
import copy

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
    def __init__(self, mode: str, vis: VisdomWrapper, dataset_path: str, snapshots_path: str, num_epochs: int,
                 batch_size: int, num_workers: int, encoder_type: str, num_points: int, num_primitives: int,
                 bottleneck_size: int, learning_rate: float, epoch_num_reset_optimizer: int,
                 multiplier_learning_rate: float):
        self._mode = mode
        self._vis = vis
        self._dataset_path = dataset_path
        self._snapshots_path = snapshots_path
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._num_points = num_points
        self._learning_rate = learning_rate
        self._epoch_num_reset_optimizer = epoch_num_reset_optimizer
        self._multiplier_learning_rate = multiplier_learning_rate

        self._train_data_loader = self._get_data_loader("train")
        self._test_data_loader = self._get_data_loader("test")
        self._categories = self._get_categories()

        self._network = Network(encoder_type=encoder_type, num_points=self._num_points, num_primitives=num_primitives,
                                bottleneck_size=bottleneck_size, learning_rate=learning_rate)

        self._loss_func = dist_chamfer.chamferDist()

        self._train_loss = AverageValueMeter()
        self._test_loss = AverageValueMeter()
        self._per_cat_test_loss = {key: AverageValueMeter() for key in self._categories}
        self._best_loss = 1e6
        self._best_per_cat_test_loss = None

        self._total_learning_time = 0.0
        self._epoch_learning_time = 0.0

    def train(self):
        logger.info("Training started!")

        for epoch in range(self._num_epochs):
            if epoch == self._epoch_num_reset_optimizer:
                new_learning_rate = self._learning_rate * self._multiplier_learning_rate
                logger.info("Reset optimizer! New learning rate is %.16f." % new_learning_rate)
                self._network.reset_optimizer(new_learning_rate)

            start_time = time.time()

            self._train_epoch(epoch)
            self._test_epoch(epoch)
            self._show_graphs()
            self._save_snapshot(epoch)

            self._epoch_learning_time = time.time() - start_time
            self._total_learning_time += self._epoch_learning_time

            self._print_epoch_stat(epoch)

        self._print_summary_stat()

    def test(self):
        pass

    def _get_data_loader(self, dataset_part: str = "test"):
        logger.info("\nInitializing data loader. Mode: %s, dataset part: %s.\n" % (self._mode, dataset_part))

        if self._mode == "train" and dataset_part == "train":
            return DataLoader(
                dataset=ShapeNetDataset(dataset_path=self._dataset_path, mode="train", num_points=self._num_points),
                batch_size=self._batch_size,
                shuffle=True,
                num_workers=self._num_workers
            )

        return DataLoader(
            dataset=ShapeNetDataset(dataset_path=self._dataset_path, mode="test", num_points=self._num_points),
            batch_size=1,
            shuffle=False,
            num_workers=1
        )

    def _get_categories(self):
        result = []

        with open(os.path.join(self._dataset_path, "synsetoffset2category.txt"), "r") as fp:
            for line in fp:
                tokens = line.strip().split()
                result.append(tokens[0])

        logger.info("Categories: %s" % str(result))

        return result

    def _train_epoch(self, epoch):
        self._train_loss.reset()
        self._network.set_train_mode()

        for batch_num, batch_data in enumerate(self._train_data_loader, 1):
            point_clouds, _ = batch_data

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
        self._reset_per_cat_test_loss()
        self._network.set_test_mode()

        with torch.no_grad():
            for batch_num, batch_data in enumerate(self._test_data_loader, 1):
                point_cloud, category = batch_data

                reconstructed_point_cloud = self._network.forward(point_cloud)

                dist_1, dist_2 = self._loss_func(point_cloud.cuda(), reconstructed_point_cloud)
                loss = torch.mean(dist_1) + torch.mean(dist_2)

                loss_value = loss.item()
                self._test_loss.update(loss_value)
                self._per_cat_test_loss[category[0]].update(loss_value)

                if batch_num % conf.VISDOM_UPDATE_FREQUENCY:
                    self._vis.show_point_cloud("REAL TEST", point_cloud)
                    self._vis.show_point_cloud("FAKE TEST", reconstructed_point_cloud)

                logger.info(
                    "[%d: %d/%d] test chamfer loss: %f " % (epoch, batch_num, len(self._test_data_loader), loss_value))

            self._vis.append_point_to_curve("Chamfer loss", "test", epoch, self._test_loss.avg)
            self._vis.append_point_to_curve("Chamfer log loss", "test", epoch, np.log(self._test_loss.avg))

    def _reset_per_cat_test_loss(self):
        for key in self._per_cat_test_loss:
            self._per_cat_test_loss[key].reset()

    def _show_graphs(self):
        self._vis.show_graph("Chamfer loss")
        self._vis.show_graph("Chamfer log loss")

    def _save_snapshot(self, epoch):
        logger.info("Saving network snapshot to latest.pth...")
        self._network.save_snapshot(os.path.join(self._snapshots_path, "latest.pth"))
        logger.info("Snapshot of epoch %d saved." % epoch)

        if self._best_loss > self._test_loss.avg:
            self._best_loss = self._test_loss.avg
            self._best_per_cat_test_loss = copy.deepcopy(self._per_cat_test_loss)
            logger.info("Current network snapshot is best. Saving it to best.pth...")
            self._network.save_snapshot(os.path.join(self._snapshots_path, "best.pth"))
            logger.info("Snapshot saved.")

    def _print_epoch_stat(self, epoch):
        logger.info("\nEpoch %d stat:" % epoch)
        logger.info("\tTrain loss: %f" % self._train_loss.avg)
        logger.info("\tTest loss: %f" % self._test_loss.avg)
        logger.info("\tPer cat test loss: " + ", ".join(
            ["%s: %.16f" % (key, self._per_cat_test_loss[key].avg) for key in self._per_cat_test_loss]))
        logger.info("\tEpoch learning time: %f sec.\n" % self._epoch_learning_time)

    def _print_summary_stat(self):
        logger.info("\nSummary stat:")
        logger.info("\tBest score: %.16f" % self._best_loss)
        logger.info("\tPer cat best score: " + ", ".join(
            ["%s: %.16f" % (key, self._best_per_cat_test_loss[key].avg) for key in self._best_per_cat_test_loss]))
        logger.info("\tTotal learning time: %f minutes" % (self._total_learning_time / 60.0))
