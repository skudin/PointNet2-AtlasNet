import argparse
import os
import os.path as osp
import concurrent.futures as cf
import json
import time

import numpy as np
import open3d as o3d

import atlasnet2.libs.helpers as h
from atlasnet2.libs.meshing import meshing, wax_up_meshing


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+", help="input paths with inference results")
    parser.add_argument("--output", required=True, help="output path")

    return parser.parse_args()


def meshing_point_cloud(item_path, output_filename):
    point_cloud = np.asarray(o3d.io.read_point_cloud(osp.join(item_path, "output_point_cloud.ply")).points)

    with open(osp.join(item_path, "metadata.json"), "r") as fp:
        category = json.load(fp)["category"]

    if category == "wax_up":
        if point_cloud.shape[0] >= 10000:
            margin_approx_points_number = 50
        else:
            margin_approx_points_number = 25

        mesh = wax_up_meshing(point_cloud=point_cloud, margin_approx_points_number=margin_approx_points_number)
    else:
        mesh = meshing(point_cloud=point_cloud)

    o3d.io.write_triangle_mesh(output_filename, mesh, write_ascii=True, write_vertex_colors=False)

    return output_filename


def wait_futures(futures):
    for future in cf.as_completed(futures):
        error = future.exception()
        if error is None:
            print("%s is ready." % future.result())
        else:
            print("Exception: %s" % str(error))


def meshing_point_clouds(input_paths, output_path):
    start_time = time.time()

    futures = set()
    with cf.ProcessPoolExecutor(max_workers=2) as executor:
        for path in input_paths:
            output_dir = osp.join(output_path, osp.split(osp.normpath(path))[1])
            os.makedirs(output_dir)

            for file_obj in os.listdir(path):
                file_obj_path = osp.join(path, file_obj)

                if not osp.isdir(file_obj_path):
                    continue

                futures.add(
                    executor.submit(meshing_point_cloud, file_obj_path, osp.join(output_dir, "%s.ply" % file_obj)))

        wait_futures(futures)

    print("Mean time meshing: %f" % ((time.time() - start_time) / len(futures)))


def main():
    args = parse_command_prompt()

    h.create_folder_with_dialog(args.output)

    meshing_point_clouds(args.input, args.output)

    print("Done.")


if __name__ == "__main__":
    main()
