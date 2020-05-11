import os
import shutil
import logging
import sys
import random
from typing import Union

import numpy as np
import torch

import pointnet2_atlasnet.configuration as conf
from pointnet2_atlasnet.libs.visdom_wrapper import VisdomWrapper


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


def set_random_seed():
    random.seed(conf.RANDOM_SEED)
    np.random.seed(conf.RANDOM_SEED)
    torch.manual_seed(conf.RANDOM_SEED)


def init_train(settings):
    experiment_path, snapshots_path = create_folders_for_experiment(settings["experiment"])

    logger = set_logging(name="", logging_level=logging.INFO, logging_to_stdout=True,
                         log_filename=os.path.join(experiment_path, "training.log"))

    logger.info("Saving startup settings to the experiment folder.")
    settings.save_settings(experiment_path)
    logger.info("Done!")

    vis = VisdomWrapper(server=settings["visdom_server"], port=settings["visdom_port"], env=settings["visdom_env"],
                        output_filename=os.path.join(experiment_path, "graphs.json"))

    set_random_seed()
    logger.info("Random seed %d." % conf.RANDOM_SEED)

    return experiment_path, snapshots_path, logger, vis

