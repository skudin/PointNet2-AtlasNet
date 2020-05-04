import argparse
import os
import os.path as osp

import open3d as o3d
import numpy as np
import torch
from emd import earth_mover_distance


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to inference results")
    parser.add_argument("--output", required=True, help="output filename")

    return parser.parse_args()


def get_avg_emd(path):
    avg_emd = 0.0
    counter = 0

    for file_obj in os.listdir(path):
        file_obj_path = osp.join(path, file_obj)

        if not osp.isdir(file_obj_path):
            continue

        input_point_cloud = torch.from_numpy(
            np.asarray(o3d.io.read_point_cloud(osp.join(file_obj_path, "input_point_cloud.ply")).points)[
                np.newaxis, ...]).cuda()
        output_point_cloud = torch.from_numpy(
            np.asarray(o3d.io.read_point_cloud(osp.join(file_obj_path, "output_point_cloud.ply")).points)[
                np.newaxis, ...]).cuda()

        metric_value = earth_mover_distance(input_point_cloud, output_point_cloud, transpose=False).item()

        avg_emd += metric_value
        counter += 1

    return avg_emd / counter


def main():
    args = parse_command_prompt()

    avg_emd = get_avg_emd(args.input)

    print("Avg EMD: %f" % avg_emd)

    print("Done.")


if __name__ == "__main__":
    main()
