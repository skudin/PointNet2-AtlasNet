from atlasnet2.networks.network import Network
from atlasnet2.libs.helpers import AverageValueMeter


class NetworkWrapper:
    def __init__(self, mode, dataset_path):
        self._train_data_loader = self._get_data_loader("train", dataset_path, mode)
        self._test_data_loader = self._get_data_loader("test", dataset_path, mode)

        self._network = Network()

        self._loss = None

        self._train_loss = AverageValueMeter()
        self._test_loss = AverageValueMeter()

    def train(self):
        pass

    def test(self):
        pass

    def _get_data_loader(self, dataset_part, dataset_path, mode):
        return None

    def _init_network(self):
        pass

    def _init_loss(self):
        pass
