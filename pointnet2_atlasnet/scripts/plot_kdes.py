import argparse
import os
import os.path as osp
import json

import matplotlib.pyplot as plt
import seaborn as sns


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+", help="input files with distances")
    parser.add_argument("--labels", required=True, nargs="+", help="labels")
    parser.add_argument("--title", required=True, help="title")
    parser.add_argument("--output", required=True, help="output filename")

    return parser.parse_args()


def create_output_dir(filename):
    output_dir = osp.split(filename)[0]
    os.makedirs(output_dir, exist_ok=True)


def read_data(paths, labels):
    data = list()
    for num, path in enumerate(paths):
        with open(path, "r") as fp:
            data.append(dict(distances=json.load(fp), label=labels[num]))

    return data


def plot(output, data, title):
    fig, ax = plt.subplots(figsize=(15, 10))

    for item in data:
        sns.kdeplot(item["distances"], label=item["label"], ax=ax)

    plt.xlim(0.0, 0.2)
    plt.title(title, fontsize=32)
    plt.legend(fontsize=18)
    plt.tick_params(labelsize=18)
    plt.grid()

    fig.savefig(output)
    fig.savefig(output + ".png")


def main():
    args = parse_command_prompt()

    create_output_dir(args.output)
    data = read_data(args.input, args.labels)
    plot(args.output, data, args.title)

    print("Done.")


if __name__ == main():
    main()
