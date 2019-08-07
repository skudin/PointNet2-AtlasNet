import logging
import os
import random

import numpy as np
import torch

import atlasnet2.configuration as conf
import atlasnet2.libs.helpers as h
from atlasnet2.libs.settings import Settings
from atlasnet2.libs.visdom_wrapper import VisdomWrapper
from atlasnet2.libs.network_wrapper import NetworkWrapper


def set_random_seed():
    random.seed(conf.RANDOM_SEED)
    np.random.seed(conf.RANDOM_SEED)
    torch.manual_seed(conf.RANDOM_SEED)


def main():
    settings = Settings("train")

    experiment_path, snapshots_path = h.create_folders_for_experiment(settings["experiment"])

    logger = h.set_logging(name="", logging_level=logging.DEBUG, logging_to_stdout=True,
                           log_filename=os.path.join(experiment_path, "training.log"))

    logger.info("Saving startup settings to the experiment folder.")
    settings.save_settings(experiment_path)
    logger.info("Done!")

    vis = VisdomWrapper(server=settings["visdom_server"], port=settings["visdom_port"], env=settings["visdom_env"])

    set_random_seed()
    logger.info("Random seed %d." % conf.RANDOM_SEED)

    network = NetworkWrapper(mode="train", vis=vis, dataset_path=settings["dataset"], snapshots_path=snapshots_path,
                             num_epochs=settings["num_epochs"], batch_size=settings["batch_size"],
                             num_workers=settings["num_workers"], encoder_type=settings["encoder_type"],
                             num_points=settings["num_points"], num_primitives=settings["num_primitives"],
                             bottleneck_size=settings["bottleneck_size"], learning_rate=settings["learning_rate"])
    network.train()


if __name__ == "__main__":
    main()
