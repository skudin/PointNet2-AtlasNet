from atlasnet2.networks.network import Network


class NetworkWrapper:
    def __init__(self):
        self._train_data_loader = None
        self._test_data_loader = None
        
        self._network = Network()

    def train(self):
        self._train_data_loader = self._get_data_loader("train")
        self._test_data_loader = self._get_data_loader("test")

    def test(self):
        pass

    def _get_data_loader(self, dataset_part):
        pass

    def _init_network(self):
        pass

    def _init_loss(self):
        pass
