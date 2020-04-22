import os
import logging

# For fix SIGSEGV.
import open3d

import atlasnet2.libs.helpers as h
from atlasnet2.libs.settings import Settings
from atlasnet2.libs.network_wrapper import NetworkWrapper


def main():
    settings = Settings("test")

    snapshots_path = os.path.join(settings.experiment_folder, "snapshots")

    h.create_folder_with_dialog(settings.output)

    h.set_logging("", logging_level=logging.DEBUG, logging_to_stdout=True,
                  log_filename=os.path.join(settings.output, "testing.log"))

    network = NetworkWrapper(svr=True, mode="test", dataset_path=settings.input, snapshots_path=snapshots_path,
                             num_workers=1, encoder_type=settings["encoder_type"],
                             pretrained_ae=settings["pretrained_ae"],
                             num_points=settings["num_points"], num_primitives=settings["num_primitives"],
                             bottleneck_size=settings["bottleneck_size"], result_path=settings.output,
                             snapshot=settings.snapshot + ".pth", num_points_gen=settings.num_points_gen,
                             scaling_fn=settings.scaling_fn)
    network.test()


if __name__ == "__main__":
    main()
