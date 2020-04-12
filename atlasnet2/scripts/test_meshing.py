import argparse
import os
import os.path as osp
import time

import numpy as np

import atlasnet2.libs.meshing as meshing
import atlasnet2.libs.helpers as h


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to folder with point clouds")
    parser.add_argument("--output", required=True, help="output path")
    parser.add_argument("--margin_approx_points_number", required=True, type=int,
                        help="margin approximation points number")

    return parser.parse_args()


def meshing_point_clouds(input_path, output_path, margin_approx_points_number):
    mean_time = 0.0
    counter = 0
    for file_obj in os.listdir(input_path):
        file_obj_path = osp.join(input_path, file_obj)

        if not osp.isfile(file_obj_path) or not file_obj.endswith(".npy"):
            continue

        start_time = time.time()

        output_dir = osp.join(output_path, file_obj)
        os.makedirs(output_dir)
        point_cloud = np.load(file_obj_path).squeeze()
        _ = meshing.wax_up_meshing(point_cloud, margin_approx_points_number, None)
        # _ = meshing.meshing(point_cloud, output_dir=None)

        execution_time = time.time() - start_time
        mean_time += execution_time
        counter += 1

        print("Case %s is completed. Execution time: %f s" % (file_obj, execution_time))

    mean_time /= counter
    print("Mean execution time: %f s" % mean_time)


def main():
    args = parse_command_prompt()

    h.create_folder_with_dialog(args.output)

    meshing_point_clouds(args.input, args.output, args.margin_approx_points_number)

    print("Hello world")


if __name__ == "__main__":
    main()
