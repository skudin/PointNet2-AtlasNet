import argparse
import os
import os.path as osp
import json

import matplotlib.pyplot as plt
import seaborn as sns


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+", help="input files with distances")
    parser.add_argument("--output", required=True, help="output filename")

    return parser.parse_args()


def create_output_dir(filename):
    output_dir = osp.split(filename)[0]
    os.makedirs(output_dir, exist_ok=True)


def read_data(paths):
    data = list()
    for path in paths:
        with open(path, "r") as fp:
            data.append(json.load(fp))

    return data


def plot(output, data):
    fig, ax = plt.subplots(figsize=(10, 10))

    for item in data:
        sns.kdeplot(item, shade=True, ax=ax)

    plt.title("Test title", fontsize=22)
    plt.legend()

    fig.savefig(output)


def main():
    args = parse_command_prompt()

    create_output_dir(args.output)
    data = read_data(args.input)
    plot(args.output, data)

    print("Hello world!")


if __name__ == main():
    main()
