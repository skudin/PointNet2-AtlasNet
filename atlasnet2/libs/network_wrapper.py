import torch

from atlasnet2.networks.network import Network
from atlasnet2.libs.helpers import AverageValueMeter


class NetworkWrapper:
    def __init__(self, mode, dataset_path, num_epochs):
        self._num_epochs = num_epochs

        self._train_data_loader = self._get_data_loader("train", dataset_path, mode)
        self._test_data_loader = self._get_data_loader("test", dataset_path, mode)

        self._network = Network()

        self._loss_func = None

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
        pass

    def test(self):
        pass

    def _get_data_loader(self, dataset_part, dataset_path, mode):
        return None

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
        pass

    def _save_snapshot(self, epoch):
        pass

    def _print_epoch_stat(self, epoch):
        pass
