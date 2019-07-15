import logging
import os

import atlasnet2.libs.helpers as h
from atlasnet2.libs.settings import Settings
from atlasnet2.libs.visdom_wrapper import VisdomWrapper
from atlasnet2.libs.network_wrapper import NetworkWrapper


def main():
    settings = Settings("train")

    common_params = settings.get_common_params()
    training_params = settings.get_training_params()

    experiment_path, snapshots_path = h.create_folders_for_experiment(common_params["experiment_name"])

    logger = h.set_logging("", logging_level=logging.DEBUG, logging_to_stdout=True,
                           log_filename=os.path.join(experiment_path, "training.log"))

    logger.info("Saving startup settings to the experiment folder.")
    settings.save_settings(experiment_path)
    logger.info("Done!")

    vis = VisdomWrapper()
    #
    # network = NetworkWrapper()
    # network.train()


if __name__ == "__main__":
    main()
