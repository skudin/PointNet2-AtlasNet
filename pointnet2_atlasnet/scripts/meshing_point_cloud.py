import argparse
import os.path as osp
import json

import numpy as np
import open3d as o3d

from pointnet2_atlasnet.libs.meshing import meshing, wax_up_meshing


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input item path")
    parser.add_argument("output", help="output filename")

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


def main():
    args = parse_command_prompt()

    meshing_point_cloud(args.input, args.output)

    print("Done.")


if __name__ == "__main__":
    main()
