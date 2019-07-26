import logging
import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from plyfile import PlyData
import numpy as np

import atlasnet2.configuration as conf


logger = logging.getLogger(__name__)


class ShapeNetDataset(data.Dataset):
    def __init__(self, dataset_path: str = os.path.join(conf.BASE_PATH, "data", "shapenet"), mode: str = "train",
                 num_points: int = 2500, include_normals: bool = False):
        self._dataset_path = dataset_path
        self._mode = mode
        self._num_points = num_points
        self._include_normals = include_normals

        self._img_path = os.path.join(self._dataset_path, "ShapeNet", "ShapeNetRendering")
        self._point_clouds_path = os.path.join(self._dataset_path, "customShapeNet")

        self._meta = dict()
        self._items = list()

        self._categories = self._get_categories()
        logger.info("Categories: %s" % str(self._categories))

        self._indexing()
        logger.info("Indexing is finished.")

        self._init_transforms()
        logger.info("All transformations are initialized.")

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        item = self._items[index]
        with open(item["point_cloud"], "rb") as fp:
            ply_data = PlyData.read(fp)

        raw_point_cloud = np.vstack([ply_data["vertex"][key].T for key, _ in ply_data["vertex"].data.dtype.descr]).T

        if not self._include_normals:
            raw_point_cloud = raw_point_cloud[:, 0: 3]

        point_cloud = raw_point_cloud[np.random.choice(raw_point_cloud.shape[0], self._num_points, replace=False), :]
        np.random.shuffle(point_cloud)

        point_cloud = torch.from_numpy(point_cloud)

        return point_cloud

    def _get_categories(self):
        result = dict()

        with open(os.path.join(self._dataset_path, "synsetoffset2category.txt"), "r") as fp:
            for line in fp:
                tokens = line.strip().split()
                result[tokens[0]] = tokens[1]

        return result

    def _indexing(self):
        empty = []
        for item in self._categories:
            img_folder = os.path.join(self._img_path, self._categories[item])
            img_folder_index = sorted(os.listdir(img_folder))

            ply_folder = os.path.join(self._point_clouds_path, self._categories[item], "ply")
            # noinspection PyBroadException
            try:
                ply_folder_index = sorted(os.listdir(ply_folder))
            except:
                ply_folder_index = []

            index = [value for value in img_folder_index if value + ".points.ply" in ply_folder_index]
            logger.info(
                "Category: %s, amount of files: %d, %f %%" % (item, len(index), len(index) / len(img_folder_index)))

            if self._mode == "train":
                index = index[: int(len(index) * 0.8)]
            else:
                index = index[int(len(index) * 0.8):]

            if len(index) != 0:
                self._meta[item] = list()
                for filename in index:
                    self._meta[item].append({
                        "rendering_path": os.path.join(img_folder, filename, "rendering"),
                        "point_cloud": os.path.join(ply_folder, filename + ".points.ply"),
                        "category": item,
                        "item": filename
                    })
            else:
                empty.append(item)

        for item in empty:
            del self._categories[item]

        for item in self._categories:
            for filenames in self._meta[item]:
                self._items.append(filenames)

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
