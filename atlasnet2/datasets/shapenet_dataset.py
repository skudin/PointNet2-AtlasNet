import logging
import os
import json
import random
from typing import Optional
from collections import namedtuple

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

import atlasnet2.configuration as conf


logger = logging.getLogger(__name__)


class ShapeNetDataset(data.Dataset):
    def __init__(self, svr: bool = False,
                 dataset_path: str = os.path.join(conf.BASE_PATH, "data", "shapenet_tiny", "dataset"),
                 mode: str = "train", num_points: int = 2500, include_normals: bool = False, gen_view: bool = False,
                 fixed_render_num: Optional[int] = None):
        self._svr = svr
        self._dataset_path = dataset_path
        self._mode = mode
        self._num_points = num_points
        self._include_normals = include_normals
        self._gen_vew = gen_view
        self._fixed_render_num = fixed_render_num

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

        if self._svr:
            if self._gen_vew:
                render_path = item.renders[0]
            elif self._fixed_render_num is not None:
                render_path = item.renders[self._fixed_render_num]
            else:
                render_path = item.renders[random.randint(1, len(item.renders) - 1)]

            image = Image.open(os.path.join(render_path))

            if self._mode == "train":
                image = self._data_augmentation(image) # random crop
            else:
                image = self._validating_transform(image) # center crop

            image = self._finish_transforms(image) # scale and to tensor
            image = image[: 3]

            return image, point_cloud, item.category, item.name
        else:
            return point_cloud, item.category, item.name

    def _indexing(self):
        dataset_part_path = os.path.join(self._dataset_path, self._mode)

        Item = namedtuple("Item", "name category point_cloud renders")

        items_list = sorted(os.listdir(dataset_part_path))

        for item_name in items_list:
            case_path = os.path.join(dataset_part_path, item_name)

            with open(os.path.join(case_path, "meta.json"), "r") as fp:
                category = json.load(fp)["category"]

            renders_path = os.path.join(case_path, "renders")
            renders = sorted([os.path.join(renders_path, file_obj) for file_obj in os.listdir(renders_path) if
                              os.path.isfile(os.path.join(renders_path, file_obj)) and file_obj.endswith(".png")])

            self._items.append(
                Item(item_name, category, os.path.join(dataset_part_path, item_name, "point_cloud.npy"), renders))

    def _init_transforms(self):
        self._finish_transforms = transforms.Compose([
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
    dataset = ShapeNetDataset(svr=True)
    tmp = dataset[0]
    pass
