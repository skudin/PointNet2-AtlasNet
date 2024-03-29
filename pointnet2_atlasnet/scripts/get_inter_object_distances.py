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
    parser.add_argument("--input", required=True, help="path to input data or file with configuration for all paths")
    parser.add_argument("--data_type", choices=("dataset", "inference_input", "inference_output"),
                        help="type of input data")
    parser.add_argument("--output", required=True, help="output filename or output directory")

    return parser.parse_args()


def read_point_clouds(path, data_type):
    point_clouds = list()
    for file_obj in os.listdir(path):
        file_obj_path = osp.join(path, file_obj)

        if not osp.isdir(file_obj_path):
            continue

        if data_type == "dataset":
            point_cloud = np.load(osp.join(file_obj_path, "point_cloud.npy")).astype(np.float32)
        elif data_type == "inference_input":
            point_cloud = np.asarray(
                o3d.io.read_point_cloud(osp.join(file_obj_path, "input_point_cloud.ply")).points).astype(np.float32)
        else:
            point_cloud = np.asarray(
                o3d.io.read_point_cloud(osp.join(file_obj_path, "output_point_cloud.ply")).points).astype(np.float32)

        point_clouds.append(torch.from_numpy(point_cloud[np.newaxis, ...]).cuda())

    return point_clouds


def get_distances(point_clouds):
    distances = []
    chamfer_dist = dist_chamfer.chamferDist()
    counter = 0
    amount_iterations = (len(point_clouds) ** 2 - len(point_clouds)) // 2
    avg_time = h.AverageValueMeter()

    start_time = time.time()

    for i in range(len(point_clouds)):
        first_point_cloud = point_clouds[i]

        for j in range(i + 1, len(point_clouds)):
            start_iter_time = time.time()

            second_point_cloud = point_clouds[j]

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


def read_input_data(filename):
    with open(filename, "r") as fp:
        return json.load(fp)


def main():
    args = parse_command_prompt()

    if osp.isdir(args.input):
        point_clouds = read_point_clouds(args.input, args.data_type)
        distances = get_distances(point_clouds)
        save_result(args.output, distances)
    else:
        data = read_input_data(args.input)

        for path, data_type, output_filename in data:
            point_clouds = read_point_clouds(path, data_type)
            distances = get_distances(point_clouds)
            save_result(osp.join(args.output, output_filename), distances)

    print("Done.")


if __name__ == "__main__":
    main()
