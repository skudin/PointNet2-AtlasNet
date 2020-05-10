import argparse
import json
import shutil
import os

import pointnet2_atlasnet.libs.helpers as h


class Settings:
    def __init__(self, mode):
        self._common_params_filename = None
        self._training_params_filename = None
        self.snapshot = None
        self.experiment_folder = None
        self.num_points_gen = None
        self.input = None
        self.output = None
        self.scaling = None

        self._read_command_prompt_parser(mode)
        self._init_values()

    def _read_command_prompt_parser(self, mode):
        parser = argparse.ArgumentParser()

        if mode == "train":
            parser.add_argument("-c", "--common_parameters", required=True, type=str,
                                      help="file with common parameters")
            parser.add_argument("-t", "--training_parameters", required=True, type=str,
                                      help="file with training parameters")
        else:
            parser.add_argument("-e", "--experiment", required=True, type=str, help="experiment name")
            parser.add_argument("-s", "--snapshot", required=True, choices=("latest", "best"), help="snapshot name")
            parser.add_argument("-i", "--input", required=True, type=str, help="input data for inference")
            parser.add_argument("-o", "--output", required=True, type=str, help="output folder")
            parser.add_argument("-n", "--num_points_gen", type=int, help="number of points to generate")
            parser.add_argument("--scaling_fn", type=str, help="filename with information about scaling")

        args = parser.parse_args()

        if mode == "train":
            self._common_params_filename = args.common_parameters
            self._training_params_filename = args.training_parameters
        else:
            self.experiment_folder = os.path.join(h.get_path_to_experiments_folder(), args.experiment)
            self._common_params_filename = os.path.join(self.experiment_folder, "common.json")
            self._training_params_filename = os.path.join(self.experiment_folder, "training.json")
            self.snapshot = args.snapshot
            self.input = args.input
            self.output = args.output
            self.scaling_fn = args.scaling_fn
            self.num_points_gen = args.num_points_gen

    def _init_values(self):
        common_params = self._parse_params_file(self._common_params_filename)
        training_params = self._parse_params_file(self._training_params_filename)

        self._settings = {**common_params, **training_params}

        if self._settings["visdom_env"] is None:
            self._settings["visdom_env"] = self._settings["experiment"]

        if isinstance(self._settings["epoch_num_reset_optimizer"], int):
            self._settings["epoch_num_reset_optimizer"] = (self._settings["epoch_num_reset_optimizer"], )
            self._settings["multiplier_learning_rate"] = (self._settings["multiplier_learning_rate"], )

    def __getitem__(self, key: str):
        return self._settings[key]

    def save_settings(self, path):
        shutil.copy(self._common_params_filename, os.path.join(path, "common.json"))
        shutil.copy(self._training_params_filename, os.path.join(path, "training.json"))
        shutil.copy(os.path.join(self._settings["dataset"], "categories.json"), os.path.join(path, "categories.json"))

    @staticmethod
    def _parse_params_file(filename):
        with open(filename, "r") as fp:
            return json.load(fp)
