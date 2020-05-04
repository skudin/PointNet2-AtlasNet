import argparse
import os
import os.path as osp
import time
import json

import open3d as o3d
import numpy as np
import torch
from emd import earth_mover_distance


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="path to inference results or file with list of paths inferences results")
    parser.add_argument("--output", required=True, help="output filename or output dir")

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
    total_time = 0.0

    for num, item in enumerate(items, 1):
        start_calculation_time = time.time()

        input_point_cloud = torch.from_numpy(
            np.asarray(o3d.io.read_point_cloud(item["input"]).points)[np.newaxis, ...]).cuda()
        output_point_cloud = torch.from_numpy(
            np.asarray(o3d.io.read_point_cloud(item["output"]).points)[np.newaxis, ...]).cuda()

        metric_value = earth_mover_distance(input_point_cloud, output_point_cloud, transpose=False).item()

        avg_emd += metric_value / len(items)

        calculation_time = time.time() - start_calculation_time
        total_time += calculation_time
        print("Item %s is ready (%d/%d). EMD: %f. Calculation metric time: %f s" % (
            item["name"], num, len(items), metric_value, calculation_time))

    print("Total calculation time: %f s" % total_time)
    print("Avg calculation time: %f s" % (total_time / len(items)))
    print("Avg EMD: %f" % avg_emd)

    return avg_emd


def save_result(filename, avg_emd):
    with open(filename, "w") as fp:
        json.dump(dict(avg_emd=avg_emd), fp=fp, indent=4)


def read_paths(filename):
    with open(filename, "r") as fp:
        return json.load(fp)


def main():
    args = parse_command_prompt()

    if osp.isfile(args.input):
        paths = read_paths(args.input)

        for path in paths:
            _, task_name = osp.split(path)
            print("Getting average EMD for %s has started." % task_name)

            avg_emd = get_avg_emd(path)
            save_result(osp.join(args.output, "%s.json" % task_name), avg_emd)
    else:
        avg_emd = get_avg_emd(args.input)
        save_result(args.output, avg_emd)

    print("Done.")


if __name__ == "__main__":
    main()
