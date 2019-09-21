import logging


logger = logging.getLogger(__name__)


class SVRWrapper():
    def __init__(self, mode: str, dataset_path: str, snapshots_path: str, num_epochs: int, batch_size: int,
                 num_workers: int):
        self._mode = mode
        self._dataset_path = dataset_path
        self._snapshots_path = snapshots_path
        self._num_epochs = num_epochs
        self._batch_size = batch_size

    def train(self):
        pass
