import os
import logging

import atlasnet2.libs.helpers as h
from atlasnet2.libs.settings import Settings
from atlasnet2.libs.network_wrapper import NetworkWrapper


def main():
    settings = Settings("test")

    result_path = os.path.join(settings.experiment_folder, "result", "%d" % settings.snapshot)
    snapshots_path = os.path.join(settings.experiment_folder, "snapshots")

    h.create_folder_with_dialog(result_path)

    logger = h.set_logging("", logging_level=logging.DEBUG, logging_to_stdout=True,
                           log_filename=os.path.join(result_path, "testing.log"))

    network = NetworkWrapper(mode="test", dataset_path=settings["dataset"], snapshots_path=snapshots_path,
                             num_epochs=settings["num_epochs"], batch_size=settings["batch_size"],
                             num_workers=settings["num_workers"], encoder_type=settings["encoder_type"],
                             num_points=settings["num_points"], num_primitives=settings["num_primitives"],
                             bottleneck_size=settings["bottleneck_size"], learning_rate=settings["learning_rate"],
                             epoch_num_reset_optimizer=settings["epoch_num_reset_optimizer"],
                             multiplier_learning_rate=settings["multiplier_learning_rate"], result_path=result_path)
    network.test()


if __name__ == "__main__":
    main()
