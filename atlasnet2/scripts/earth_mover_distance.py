import argparse
import os
import os.path as osp
import time

import open3d as o3d
import numpy as np
import torch
from emd import earth_mover_distance


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to inference results")
    parser.add_argument("--output", required=True, help="output filename")

    return parser.parse_args()


def get_items(path):
    items = list()

    for file_obj in os.listdir(path):
        file_obj_path = osp.join(path, file_obj)

        if not osp.isdir(file_obj_path):
            continue

        items.append({
            "name": file_obj,
            "input": osp.join(file_obj_path, "input_point_cloud.ply"),
            "output": osp.join(file_obj_path, "output_point_cloud.ply")
        })

    return items


def get_avg_emd(path):
    items = get_items(path)
    avg_emd = 0.0

    for num, item in enumerate(items, 1):
        start_calculation_time = time.time()

        input_point_cloud = torch.from_numpy(
            np.asarray(o3d.io.read_point_cloud(item["input"]).points)[np.newaxis, ...]).cuda()
        output_point_cloud = torch.from_numpy(
            np.asarray(o3d.io.read_point_cloud(item["output"]).points)[np.newaxis, ...]).cuda()

        metric_value = earth_mover_distance(input_point_cloud, output_point_cloud, transpose=False).item()

        avg_emd += metric_value / len(items)

        calculation_time = time.time() - start_calculation_time
        print("Item %s is ready (%d/%d). EMD: %f. Calculation metric time: %f s" % (
            item["name"], num, len(items), metric_value, calculation_time))

    return avg_emd


def main():
    args = parse_command_prompt()

    avg_emd = get_avg_emd(args.input)

    print("Avg EMD: %f" % avg_emd)

    print("Done.")


if __name__ == "__main__":
    main()
