import logging
import os

import torch.utils.data as data

import atlasnet2.configuration as conf


logger = logging.getLogger(__name__)


class ShapeNetDataset(data.Dataset):
    def __init__(self, dataset_path: str = os.path.join(conf.BASE_PATH, "data", "shapenet"), mode: str = "train"):
        self._dataset_path = dataset_path
        self._mode = mode

        self._img_path = os.path.join(self._dataset_path, "ShapeNet", "ShapeNetRendering")
        self._point_clouds_path = os.path.join(self._dataset_path, "customShapeNet")

        self._meta = dict()
        self._datapath = list()

        self._categories = self._get_categories()
        logger.info("Categories: %s" % str(self._categories))

        self._indexing()
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

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
                index = index[int(len(index)) * 0.8:]

            if len(index) != 0:
                self._meta[item] = list()
                for filename in index:
                    self._meta[item].append((os.path.join(img_folder, filename, "rendering"),
                                             os.path.join(ply_folder, filename + ".points.ply"), item, filename))
            else:
                empty.append(item)

        for item in empty:
            del self._categories[item]

        for item in self._categories:
            for filenames in self._meta[item]:
                self._datapath.append(filenames)
