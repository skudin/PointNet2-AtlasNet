import logging
import os
import json
from collections import namedtuple

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

import atlasnet2.configuration as conf


logger = logging.getLogger(__name__)


class ShapeNetDataset(data.Dataset):
    def __init__(self, dataset_path: str = os.path.join(conf.BASE_PATH, "data", "shapenet", "dataset"),
                 mode: str = "train", num_points: int = 2500, include_normals: bool = False, svr: bool = False):
        self._dataset_path = dataset_path
        self._mode = mode
        self._num_points = num_points
        self._include_normals = include_normals
        self._svr = svr

        self._items = list()

        self._indexing()
        logger.info("Indexing is finished.")

        self._init_transforms()
        logger.info("All transformations are initialized.")

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        item = self._items[index]

        raw_point_cloud = np.load(item.point_cloud, allow_pickle=False)

        if not self._include_normals:
            raw_point_cloud = raw_point_cloud[:, 0: 3]

        point_cloud = raw_point_cloud[np.random.choice(raw_point_cloud.shape[0], self._num_points, replace=False), :]
        np.random.shuffle(point_cloud)

        point_cloud = torch.from_numpy(point_cloud)

        return point_cloud, item.category, item.name

    def _indexing(self):
        dataset_part_path = os.path.join(self._dataset_path, self._mode)

        Item = namedtuple("Item", "name category point_cloud")

        items_list = sorted(os.listdir(dataset_part_path))

        for item_name in items_list:
            with open(os.path.join(dataset_part_path, item_name, "meta.json"), "r") as fp:
                category = json.load(fp)["category"]

            self._items.append(Item(item_name, category, os.path.join(dataset_part_path, item_name, "point_cloud.npy")))

    def _init_transforms(self):
        self._transforms = transforms.Compose([
            transforms.Resize(size=224, interpolation=2),
            transforms.ToTensor()
        ])

        # RandomResizedCrop or RandomCrop
        self._data_augmentation = transforms.Compose([
            transforms.RandomCrop(127),
            transforms.RandomHorizontalFlip(),
        ])

        self._validating_transform = transforms.Compose([
            transforms.CenterCrop(127),
        ])


if __name__ == "__main__":
    dataset = ShapeNetDataset()
    tmp = dataset[0]
    pass
