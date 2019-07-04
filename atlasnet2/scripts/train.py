import argparse
import json


class ParamsReader:
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

    @staticmethod
    def _parse_params_file(filename):
        with open(filename, "r") as fp:
            return json.load(fp)


class Trainer:
    pass


def main():
    params_reader = ParamsReader()
    params_reader.parse_command_prompt()

    common_params = params_reader.get_common_params()
    training_params = params_reader.get_training_params()
    pass


if __name__ == "__main__":
    main()
