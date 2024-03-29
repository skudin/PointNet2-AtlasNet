import random

# For fix SIGSEGV.
import open3d
import numpy as np
import torch

import pointnet2_atlasnet.configuration as conf
import pointnet2_atlasnet.libs.helpers as h
from pointnet2_atlasnet.libs.settings import Settings
from pointnet2_atlasnet.libs.network_wrapper import NetworkWrapper


def set_random_seed():
    random.seed(conf.RANDOM_SEED)
    np.random.seed(conf.RANDOM_SEED)
    torch.manual_seed(conf.RANDOM_SEED)


def main():
    settings = Settings("train")

    experiment_path, snapshots_path, logger, vis = h.init_train(settings)

    network = NetworkWrapper(svr=False, mode="train", vis=vis, dataset_path=settings["dataset"],
                             snapshots_path=snapshots_path, num_epochs=settings["num_epochs"],
                             batch_size=settings["batch_size"], num_workers=settings["num_workers"],
                             encoder_type=settings["encoder_type"], num_points=settings["num_points"],
                             num_primitives=settings["num_primitives"], bottleneck_size=settings["bottleneck_size"],
                             learning_rate=settings["learning_rate"],
                             epoch_num_reset_optimizer=settings["epoch_num_reset_optimizer"],
                             multiplier_learning_rate=settings["multiplier_learning_rate"])
    network.train()


if __name__ == "__main__":
    main()
