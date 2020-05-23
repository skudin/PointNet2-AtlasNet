import argparse
import os
import os.path as osp
import time
import json

import numpy as np
import open3d as o3d
import torch

import dist_chamfer

import pointnet2_atlasnet.libs.helpers as h


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to input data")
    parser.add_argument("--data_type", required=True, choices=("dataset", "inference_output"),
                        help="type of input data")
    parser.add_argument("--output", required=True, help="output filename")

    return parser.parse_args()


def index_dir(path, data_type):
    index = list()
    for file_obj in os.listdir(path):
        file_obj_path = osp.join(path, file_obj)

        if not osp.isdir(file_obj_path):
            continue

        if data_type == "dataset":
            point_cloud_filename = osp.join(file_obj_path, "point_cloud.npy")
        else:
            point_cloud_filename = osp.join(file_obj_path, "output_point_cloud.ply")

        index.append(point_cloud_filename)

    return index


def read_point_cloud(filename, data_type):
    if data_type == "dataset":
        point_cloud = np.load(filename)
    else:
        point_cloud = np.asarray(o3d.io.read_point_cloud(filename).points)

    return torch.from_numpy(point_cloud[np.newaxis, ...]).cuda()


def get_distances(index, data_type):
    distances = []
    chamfer_dist = dist_chamfer.chamferDist()
    counter = 0
    amount_iterations = (len(index) ** 2 - len(index)) // 2
    avg_time = h.AverageValueMeter()

    start_time = time.time()

    for i in range(len(index)):
        first_point_cloud = read_point_cloud(index[i], data_type)

        for j in range(i + 1, len(index)):
            start_iter_time = time.time()

            second_point_cloud = read_point_cloud(index[j], data_type)

            dist_1, dist_2 = chamfer_dist(first_point_cloud, second_point_cloud)
            dist = (torch.mean(dist_1) + torch.mean(dist_2)).item()

            distances.append(dist)

            counter += 1
            calculation_iter_time = time.time() - start_iter_time
            avg_time.update(calculation_iter_time)
            print(
                "Dist between %d and %d is ready (%d/%d). Chamfer distance: %f. Calculation metric time: %f s" % (
                    i, j, counter, amount_iterations, dist, calculation_iter_time))

    total_time = time.time() - start_time

    print("Avg iteration time: %f s" % avg_time.avg)
    print("Total time: %f s" % total_time)

    return distances


def save_result(output, distances):
    output_dir = osp.split(output)[0]
    os.makedirs(output_dir, exist_ok=True)

    with open(output, "w") as fp:
        json.dump(distances, fp=fp)


def main():
    args = parse_command_prompt()

    index = index_dir(args.input, args.data_type)
    distances = get_distances(index, args.data_type)
    save_result(args.output, distances)

    print("Done.")


if __name__ == "__main__":
    main()
