import argparse

import dist_chamfer


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to input data")
    parser.add_argument("--input_type", required=True, choices=("dataset", "inference_output"))
    parser.add_argument("--output", required=True, help="output filename")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    print("Hello world!")


if __name__ == "__main__":
    main()
