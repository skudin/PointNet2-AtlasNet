import os
import shutil
import logging
import sys
import random
from typing import Union

import numpy as np
import torch

import atlasnet2.configuration as conf


def get_path_to_experiments_folder():
    return os.path.join(conf.BASE_PATH, "experiments")


def create_folders_for_experiment(experiment_name):
    experiment_path = os.path.join(get_path_to_experiments_folder(), experiment_name)
    create_folder_with_dialog(experiment_path)

    snapshots_path = os.path.join(experiment_path, "snapshots")
    os.makedirs(snapshots_path)

    return experiment_path, snapshots_path


def create_folder_with_dialog(path):
    if os.path.exists(path):
        print("%s exists. Remove it? (y/n)" % path)

        answer = str(input()).strip()
        if answer != "y":
            print("Shut down.")
            exit(0)

        shutil.rmtree(path)

    os.makedirs(path)


def set_logging(name: str, logging_level: int, logging_to_stdout: bool = False,
                log_filename: Union[str, None] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)

    if logging_to_stdout:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    if log_filename is not None:
        logger.info("Set additional logging to file %s." % log_filename)
        logger.addHandler(logging.FileHandler(log_filename))

    return logger


# Initialise weights of network.
def weights_init(m):
    class_name = m.__class__.__name__

    if class_name.find("Conv") != -1:
        if hasattr(m, "weight"):
            m.weight.data.normal_(0.0, 0.02)
    elif class_name.find("BatchNorm") != -1:
        if hasattr(m, "weight"):
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, "bias"):
            m.bias.data.fill_(0)


class AverageValueMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0.0
        self._count = 0
        self._avg = 0.0

    def update(self, value: float):
        self._sum += value
        self._count += 1
        self._avg = self._sum / self._count

    @property
    def avg(self):
        return self._avg


def get_colors(num_colors: int):
    colors = []
    for i in range(0, num_colors):
        colors.append(generate_new_color(colors, pastel_factor=0.9))

    for i in range(0, num_colors):
        for j in range(0, 3):
            colors[i][j] = int(colors[i][j] * 256)
        colors[i].append(255)

    return colors


def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None

    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)

        if not existing_colors:
            return color

        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color

    return best_color


def get_random_color(pastel_factor=0.5):
    return [(x + pastel_factor) / (1.0 + pastel_factor) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]]


def color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])


def set_random_seed():
    random.seed(conf.RANDOM_SEED)
    np.random.seed(conf.RANDOM_SEED)
    torch.manual_seed(conf.RANDOM_SEED)
