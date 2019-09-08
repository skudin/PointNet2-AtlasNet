import argparse
import os
import json

import numpy as np


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, help="path to npy dataset")
    parser.add_argument("-o", "--output", required=True, type=str, help="path to writing result")

    args = parser.parse_args()

    return args.input, args.output


def get_statistics(dataset_path):
    train_stats = get_statistics_for_dataset_part(dataset_path, "train")
    test_stats = get_statistics_for_dataset_part(dataset_path, "test")

    return train_stats, test_stats


def get_statistics_for_dataset_part(dataset_path, part):
    stats = init_stats()
    part_path = os.path.join(dataset_path, part)

    for file_obj in os.listdir(part_path):
        file_obj_path = os.path.join(part_path, file_obj)
        if not os.path.isdir(file_obj_path):
            continue

        point_cloud = np.load(os.path.join(file_obj_path, "point_cloud.npy"))

        stats["x"]["min"] = min(stats["x"]["min"], float(np.amin(point_cloud[:, 0])))
        stats["x"]["max"] = max(stats["x"]["max"], float(np.amax(point_cloud[:, 0])))

        stats["y"]["min"] = min(stats["y"]["min"], float(np.amin(point_cloud[:, 1])))
        stats["y"]["max"] = max(stats["y"]["max"], float(np.amax(point_cloud[:, 1])))

        stats["z"]["min"] = min(stats["z"]["min"], float(np.amin(point_cloud[:, 2])))
        stats["z"]["max"] = max(stats["z"]["max"], float(np.amax(point_cloud[:, 2])))

        print("%s processed." % file_obj)

    return stats


def init_stats():
    return {
        "x": {
            "min": 1e10,
            "max": -1e10
        },
        "y": {
            "min": 1e10,
            "max": -1e10
        },
        "z": {
            "min": 1e10,
            "max": -1e10
        }
    }


def save_stats(output_path, train_stats, test_stats):
    stats = {
        "train": train_stats,
        "test": test_stats
    }

    with open(os.path.join(output_path, "dataset_stat.json"), "w") as fp:
        json.dump(stats, fp=fp, indent=2)


def main():
    dataset_path, output_path = parse_command_prompt()
    train_stats, test_stats = get_statistics(dataset_path)
    save_stats(output_path, train_stats, test_stats)

    print("Done.")


if __name__ == "__main__":
    main()
