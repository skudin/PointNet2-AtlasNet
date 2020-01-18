import os
import logging

import atlasnet2.libs.helpers as h
from atlasnet2.libs.settings import Settings
from atlasnet2.libs.network_wrapper import NetworkWrapper


def main():
    settings = Settings("ae", "test")

    snapshots_path = os.path.join(settings.experiment_folder, "snapshots")

    h.create_folder_with_dialog(settings.output)

    h.set_logging("", logging_level=logging.DEBUG, logging_to_stdout=True,
                  log_filename=os.path.join(settings.output, "testing.log"))

    network = NetworkWrapper(svr=False, mode="test", dataset_path=settings.input, snapshots_path=snapshots_path,
                             num_epochs=settings["num_epochs"], batch_size=settings["batch_size"],
                             num_workers=settings["num_workers"], encoder_type=settings["encoder_type"],
                             num_points=settings["num_points"], num_primitives=settings["num_primitives"],
                             bottleneck_size=settings["bottleneck_size"], learning_rate=settings["learning_rate"],
                             epoch_num_reset_optimizer=settings["epoch_num_reset_optimizer"],
                             multiplier_learning_rate=settings["multiplier_learning_rate"], result_path=settings.output,
                             snapshot=settings.snapshot + ".pth", num_points_gen=settings.num_points_gen)
    network.test()


if __name__ == "__main__":
    main()
