import os

import torch.utils.data as data

import atlasnet2.configuration as conf


class ShapeNetDataset(data.Dataset):
    def __init__(self, dataset_path: str = os.path.join(conf.BASE_PATH, "data", "shapenet")):
        self._dataset_path = dataset_path

        self._categories = self._get_categories()

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def _get_categories(self):
        result = dict()

        with open(os.path.join(self._dataset_path, "synsetoffset2category.txt"), "r") as fp:
            for line in fp:
                tokens = line.strip().split()
                result[tokens[0]] = int(tokens[1])

        return result
