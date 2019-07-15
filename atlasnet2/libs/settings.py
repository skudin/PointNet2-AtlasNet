import argparse
import json
import shutil


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
