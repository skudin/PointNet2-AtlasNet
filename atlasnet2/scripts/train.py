import argparse
import json
import logging
import os
import shutil

import atlasnet2.libs.helpers as h
from atlasnet2.libs.visdom_wrapper import VisdomWrapper
from atlasnet2.libs.network_wrapper import NetworkWrapper


class Settings:
    def __init__(self):
        self._common_params_filename = None
        self._training_params_filename = None

        self._parser = argparse.ArgumentParser()
        self._parser.add_argument("-c", "--common_parameters", required=True, type=str,
                                  help="file with common parameters")
        self._parser.add_argument("-t", "--training_parameters", required=True, type=str,
                                  help="file with training parameters")

    def parse_command_prompt(self):
        args = self._parser.parse_args()

        self._common_params_filename = args.common_parameters
        self._training_params_filename = args.training_parameters

    def get_common_params(self):
        return self._parse_params_file(self._common_params_filename)

    def get_training_params(self):
        return self._parse_params_file(self._training_params_filename)

    def save_settings(self, path):
        shutil.copy(self._common_params_filename, path)
        shutil.copy(self._training_params_filename, path)

    @staticmethod
    def _parse_params_file(filename):
        with open(filename, "r") as fp:
            return json.load(fp)


def main():
    settings = Settings()
    settings.parse_command_prompt()

    common_params = settings.get_common_params()
    training_params = settings.get_training_params()

    experiment_path, snapshots_path = h.create_folders_for_experiments(common_params["experiment_name"])

    logger = h.set_logging("", logging_level=logging.DEBUG, logging_to_stdout=True,
                           log_filename=os.path.join(experiment_path, "training.log"))

    logger.info("Saving startup settings to the experiment folder.")
    settings.save_settings(experiment_path)
    logger.info("Done!")

    vis = VisdomWrapper()

    network = NetworkWrapper()
    network.train()


if __name__ == "__main__":
    main()
