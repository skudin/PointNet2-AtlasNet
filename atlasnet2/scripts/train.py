import logging
import os

import atlasnet2.libs.helpers as h
from atlasnet2.libs.settings import Settings
from atlasnet2.libs.visdom_wrapper import VisdomWrapper
from atlasnet2.libs.network_wrapper import NetworkWrapper


def main():
    settings = Settings("train")

    experiment_path, snapshots_path = h.create_folders_for_experiment(settings["experiment"])

    logger = h.set_logging(name="", logging_level=logging.DEBUG, logging_to_stdout=True,
                           log_filename=os.path.join(experiment_path, "training.log"))

    logger.info("Saving startup settings to the experiment folder.")
    settings.save_settings(experiment_path)
    logger.info("Done!")

    vis = VisdomWrapper(server=settings["visdom_server"], port=settings["visdom_port"], env=settings["visdom_env"])

    network = NetworkWrapper(mode="train")
    network.train()


if __name__ == "__main__":
    main()
