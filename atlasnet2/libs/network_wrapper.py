import logging
import os
import time
import copy
import json
from typing import Optional
from collections import namedtuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import atlasnet2.libs.helpers as h
from atlasnet2.datasets.dataset import Dataset
from atlasnet2.networks.network import Network
from atlasnet2.libs.helpers import AverageValueMeter
from atlasnet2.libs.ply import write_ply

import dist_chamfer
import atlasnet2.configuration as conf
from atlasnet2.libs.visdom_wrapper import VisdomWrapper


logger = logging.getLogger(__name__)


class NetworkWrapper:
    def __init__(self, svr: bool, mode: str, dataset_path: str, snapshots_path: str,
                 vis: Optional[VisdomWrapper] = None, num_epochs: int = 1,
                 batch_size: int = 1, num_workers: int = 1, encoder_type: str = "pointnet",
                 pretrained_ae: Optional[str] = None,
                 num_points: int = 2500,
                 num_primitives: int = 1, bottleneck_size: int = 1024, learning_rate: float = 0.001,
                 epoch_num_reset_optimizer: int = 1000, multiplier_learning_rate: float = 0.1,
                 result_path: Optional[str] = None, snapshot: Optional[str] = None,
                 num_points_gen: Optional[int] = None, scaling_fn: Optional[str] = None):
        self._svr = svr
        self._mode = mode
        self._vis = vis
        self._dataset_path = dataset_path
        self._snapshots_path = snapshots_path
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._num_points = num_points
        self._num_primitives = num_primitives
        self._learning_rate = learning_rate
        self._epoch_num_reset_optimizer = epoch_num_reset_optimizer
        self._multiplier_learning_rate = multiplier_learning_rate
        self._result_path = result_path
        self._snapshot = snapshot

        self._num_points_gen = num_points
        if num_points_gen is not None:
            self._num_points_gen = num_points_gen

        self._train_data_loader = self._get_data_loader("train")
        self._test_data_loader = self._get_data_loader("test")
        if self._svr:
            self._test_view_data_loader = self._get_data_loader("train", gen_view=True)

        self._categories = self._get_categories()

        if self._svr:
            self._network = Network(svr=True, encoder_type=encoder_type, pretrained_ae=pretrained_ae,
                                    num_points=self._num_points,
                                    num_primitives=self._num_primitives, bottleneck_size=bottleneck_size,
                                    learning_rate=learning_rate)
        else:
            self._network = Network(svr=False, encoder_type=encoder_type, num_points=self._num_points,
                                    num_primitives=self._num_primitives, bottleneck_size=bottleneck_size,
                                    learning_rate=learning_rate)

        self._loss_func = dist_chamfer.chamferDist()

        self._train_loss = AverageValueMeter()
        self._test_loss = AverageValueMeter()
        if self._svr:
            self._test_view_loss = AverageValueMeter()

        self._per_cat_test_loss = {key: AverageValueMeter() for key in self._categories}

        self._best_loss = 1e6
        self._best_per_cat_test_loss = None

        self._total_learning_time = 0.0
        self._epoch_learning_time = 0.0

        if self._mode == "test":
            self._generate_regular_grid()
            self._scaling_coeffs = self._read_scaling_coeffs(scaling_fn)

    def train(self):
        logger.info("Training started!")

        reset_index = 0

        for epoch in range(self._num_epochs):
            if epoch == self._epoch_num_reset_optimizer[reset_index]:
                self._learning_rate *= self._multiplier_learning_rate[reset_index]
                reset_index = reset_index + 1 if reset_index < len(self._epoch_num_reset_optimizer) - 1 else reset_index
                logger.info("Reset optimizer! New learning rate is %.16f." % self._learning_rate)
                self._network.reset_optimizer(self._learning_rate)

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
        snapshot = os.path.join(self._snapshots_path, self._snapshot)
        self._network.load_snapshot(snapshot)

        self._test_loss.reset()
        self._reset_per_cat_test_loss()
        self._network.set_test_mode()

        with torch.no_grad():
            for batch_num, batch_data in enumerate(self._test_data_loader, 1):
                point_cloud, category, name = batch_data
                category = category[0]
                name = name[0]

                reconstructed_point_cloud = self._scale_point_cloud(
                    self._network.inference(point_cloud, self._num_points_gen))
                point_cloud = self._scale_point_cloud(point_cloud)

                dist_1, dist_2 = self._loss_func(point_cloud.cuda(), reconstructed_point_cloud)
                loss = torch.mean(dist_1) + torch.mean(dist_2)

                loss_value = loss.item()
                self._test_loss.update(loss_value)

                self._per_cat_test_loss[category].update(loss_value)

                self._write_3d_data(name, point_cloud, reconstructed_point_cloud)

                logger.info(
                    "[%d/%d] test chamfer loss: %f " % (batch_num, len(self._test_data_loader), loss_value))

        self._print_test_stat()

    def _scale_point_cloud(self, point_cloud):
        if self._scaling_coeffs is not None:
            point_cloud[:, :, 0] = (point_cloud[:, :, 0] - self._scaling_coeffs.b_x) / self._scaling_coeffs.k_x
            point_cloud[:, :, 1] = (point_cloud[:, :, 1] - self._scaling_coeffs.b_y) / self._scaling_coeffs.k_y
            point_cloud[:, :, 2] = (point_cloud[:, :, 2] - self._scaling_coeffs.b_z) / self._scaling_coeffs.k_z

        return point_cloud

    def _write_3d_data(self, name, point_cloud, reconstructed_point_cloud):
        write_ply(filename=os.path.join(self._result_path, "%s_input_point_cloud.ply" % name),
                  points=pd.DataFrame(point_cloud.data.squeeze().numpy()), as_text=True)

        write_ply(filename=os.path.join(self._result_path,
                                        "%s_output_point_cloud_%d_points.ply" % (name, self._num_points_gen)),
                  points=pd.DataFrame(reconstructed_point_cloud.cpu().data.squeeze().numpy()), as_text=True)

    def _generate_regular_grid(self):
        logger.info("Generation of regular grid...")
        grain = int(np.sqrt(self._num_points_gen / self._num_primitives)) - 1.0
        logger.info("Grain: %f" % grain)

        self._faces = []
        vertices = []
        vertex_colors = []
        colors = h.get_colors(self._num_primitives)

        for i in range(0, int(grain + 1)):
            for j in range(0, int(grain + 1)):
                vertices.append([i / grain, j / grain])

        for prim in range(0, self._num_primitives):
            for i in range(0, int(grain + 1)):
                for j in range(0, int(grain + 1)):
                    vertex_colors.append(colors[prim])

            for i in range(1, int(grain + 1)):
                for j in range(0, (int(grain + 1) - 1)):
                    self._faces.append([(grain + 1) * (grain + 1) * prim + j + (grain + 1) * i,
                                        (grain + 1) * (grain + 1) * prim + j + (grain + 1) * i + 1,
                                        (grain + 1) * (grain + 1) * prim + j + (grain + 1) * (i - 1)])

            for i in range(0, (int((grain + 1)) - 1)):
                for j in range(1, int((grain + 1))):
                    self._faces.append([(grain + 1) * (grain + 1) * prim + j + (grain + 1) * i,
                                        (grain + 1) * (grain + 1) * prim + j + (grain + 1) * i - 1,
                                        (grain + 1) * (grain + 1) * prim + j + (grain + 1) * (i + 1)])

        self._grid = [vertices for i in range(0, self._num_primitives)]
        self._grid_pytorch = torch.Tensor(int(self._num_primitives * (grain + 1) * (grain + 1)), 2)
        for i in range(self._num_primitives):
            for j in range(int((grain + 1) * (grain + 1))):
                self._grid_pytorch[int(j + (grain + 1) * (grain + 1) * i), 0] = vertices[j][0]
                self._grid_pytorch[int(j + (grain + 1) * (grain + 1) * i), 1] = vertices[j][1]

        logger.info("Number vertices: %d" % (len(vertices) * self._num_primitives))
        logger.info("Regular grid generated.")

    def _get_data_loader(self, dataset_part: str = "test", gen_view: bool = False):
        logger.info("\nInitializing data loader. Mode: %s, dataset part: %s.\n" % (self._mode, dataset_part))

        if self._mode == "train":
            if dataset_part == "train":
                return DataLoader(
                    dataset=Dataset(svr=self._svr, dataset_path=self._dataset_path, mode="train",
                                    num_points=self._num_points, gen_view=gen_view),
                    batch_size=self._batch_size,
                    shuffle=True,
                    num_workers=self._num_workers
                )
            else:
                return DataLoader(
                    dataset=Dataset(svr=self._svr, dataset_path=self._dataset_path, mode="test",
                                    num_points=self._num_points, fixed_render_num=0),
                    batch_size=self._batch_size,
                    shuffle=False,
                    num_workers=self._num_workers
                )
        else:
            if dataset_part == "test":
                return DataLoader(
                    dataset=Dataset(svr=self._svr, dataset_path=self._dataset_path, run_type="inference", mode="test",
                                    num_points=self._num_points),
                    batch_size=1,
                    shuffle=False,
                    num_workers=1
                )
            else:
                return None

    def _get_categories(self):
        if self._mode == "train":
            categories_filename = os.path.join(self._dataset_path, "categories.json")
        else:
            categories_filename = os.path.join(self._snapshots_path, "..", "categories.json")

        with open(categories_filename, "r") as fp:
            categories = json.load(fp)

        logger.info("Categories: %s" % str(categories))

        return categories

    def _train_epoch(self, epoch):
        self._train_loss.reset()
        self._network.set_train_mode()

        for batch_num, batch_data in enumerate(self._train_data_loader, 1):
            if self._svr:
                image, point_clouds, *_ = batch_data
                network_input = image
            else:
                point_clouds, *_ = batch_data
                network_input = point_clouds

            reconstructed_point_clouds = self._network.forward(network_input)

            dist_1, dist_2 = self._loss_func(point_clouds.cuda(), reconstructed_point_clouds)
            loss = torch.mean(dist_1) + torch.mean(dist_2)
            self._network.backward(loss)

            loss_value = loss.item()
            self._train_loss.update(loss_value)

            if batch_num % conf.VISDOM_UPDATE_FREQUENCY:
                if self._svr:
                    self._vis.show_image("INPUT IMAGE TRAIN", image)

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
            if self._svr:
                self._test_view_loss.reset()

                for batch_num, batch_data in enumerate(self._test_view_data_loader, 1):
                    image, point_cloud, *_ = batch_data

                    reconstructed_point_cloud = self._network.forward(image)

                    dist_1, dist_2 = self._loss_func(point_cloud.cuda(), reconstructed_point_cloud)
                    loss = torch.mean(dist_1) + torch.mean(dist_2)

                    loss_value = loss.item()
                    self._test_view_loss.update(loss_value)

                    logger.info(
                        "[%d: %d/%d] test view chamfer loss: %f " % (
                            epoch, batch_num, len(self._test_view_data_loader), loss_value))

                self._vis.append_point_to_curve("Chamfer loss", "test view", epoch, self._test_view_loss.avg)
                self._vis.append_point_to_curve("Chamfer log loss", "test view", epoch,
                                                np.log(self._test_view_loss.avg))

            for batch_num, batch_data in enumerate(self._test_data_loader, 1):
                if self._svr:
                    image, point_cloud, category, *_ = batch_data
                    network_input = image
                else:
                    point_cloud, category, *_ = batch_data
                    network_input = point_cloud

                reconstructed_point_cloud = self._network.forward(network_input)

                dist_1, dist_2 = self._loss_func(point_cloud.cuda(), reconstructed_point_cloud)
                loss = torch.mean(dist_1) + torch.mean(dist_2)

                loss_value = loss.item()
                self._test_loss.update(loss_value)

                for index in range(dist_1.shape[0]):
                    item_loss_value = (torch.mean(dist_1[index]) + torch.mean(dist_2[index])).item()
                    self._per_cat_test_loss[category[index]].update(item_loss_value)

                if batch_num % conf.VISDOM_UPDATE_FREQUENCY:
                    if self._svr:
                        self._vis.show_image("INPUT IMAGE TEST", image)

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
        if self._svr:
            logger.info("\tTest view loss: %f" % self._test_view_loss.avg)
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

    def _print_test_stat(self):
        logger.info("\nSummary test stat:")
        logger.info("\tAvg loss: %.16f" % self._test_loss.avg)
        logger.info("\tPer cat avg loss: " + ", ".join(
            ["%s: %.16f" % (key, self._per_cat_test_loss[key].avg) for key in self._per_cat_test_loss]))

    @staticmethod
    def _read_scaling_coeffs(filename):
        Coeffs = namedtuple("Coeffs", ("k_x", "b_x", "k_y", "b_y", "k_z", "b_z"))

        if filename is not None:
            with open(filename, "r") as fp:
                data = json.load(fp)
                return Coeffs(k_x=data["k_x"], b_x=data["b_x"], k_y=data["k_y"], b_y=data["b_y"], k_z=data["k_z"],
                              b_z=data["b_z"])

        return None
