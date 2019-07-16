import os
import logging

import atlasnet2.libs.helpers as h
from atlasnet2.libs.settings import Settings
from atlasnet2.libs.network_wrapper import NetworkWrapper


def main():
    settings = Settings("test")

    result_path = os.path.join(settings.experiment_folder, "result", "%d" % settings.snapshot_num)
    h.create_folder_with_dialog(result_path)

    logger = h.set_logging("", logging_level=logging.DEBUG, logging_to_stdout=True,
                           log_filename=os.path.join(result_path, "testing.log"))

    network = NetworkWrapper()
    network.test()


if __name__ == "__main__":
    main()
