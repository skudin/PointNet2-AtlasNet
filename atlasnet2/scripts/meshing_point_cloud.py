from collections import Counter, defaultdict

import numpy as np
import open3d as o3d


NETWORK_RESULT_FILENAME = "data/debug_meshing/input/1_primitive_10000_points.npy"
OUTPUT_PREFIX = "data/debug_meshing/output/1_primitive_2500_points"
CAMERA_LOCATION = np.array([0.0, -100.0, 0.0])
EPS = 1e-8
STEP = 10.0


def estimate_normals(point_cloud, radius=1.0, max_nn=30):
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=max_nn))

    center = point_cloud.get_center()

    dirs = (np.array((1.0, 0.0, 0.0), dtype=np.float64),
            np.array((0.0, 1.0, 0.0), dtype=np.float64),
            np.array((0.0, 0.0, 1.0), dtype=np.float64))
    camera_locations = [center + sgn * STEP * dir for dir in dirs for sgn in (-1, 1)]
    camera_locations.append(center)

    normals_stat = defaultdict(Counter)
    for camera_location in camera_locations:
        point_cloud.orient_normals_towards_camera_location(camera_location=camera_location)

        for i in range(len(point_cloud.points)):
            normals_stat[i][tuple(point_cloud.normals[i])] += 1

    points_with_problem_counter = 0
    for key, normal_stat in normals_stat.items():
        normal, counter = normal_stat.most_common(1)[0]
        if counter < 7:
            points_with_problem_counter += 1
        point_cloud.normals[key] = np.array(normal, dtype=np.float64)

    return points_with_problem_counter


def main():
    point_cloud_np = np.load(NETWORK_RESULT_FILENAME).squeeze()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

    o3d.io.write_point_cloud(OUTPUT_PREFIX + "_point_cloud.ply", pcd)

    points_with_problem_counter = estimate_normals(pcd)

    o3d.io.write_point_cloud(OUTPUT_PREFIX + "_point_cloud_with_normals.ply", pcd)

    # mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, scale=1.1)
    # mesh_2 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii=o3d.utility.DoubleVector((0.1, 0.2)))
    # o3d.io.write_triangle_mesh("data/debug_meshing/1_primitive_2500_points.ply", mesh, write_ascii=True,
    #                            write_vertex_colors=False)
    # o3d.io.write_triangle_mesh("data/debug_meshing/1_primitive_2500_points_cloud_ball_pivoting.ply", mesh_2, write_ascii=True,
    #                            write_vertex_colors=False)
    # o3d.io.write_point_cloud("data/debug_meshing/1_primitive_2500_points_point_cloud.ply", pcd)
    print("Done.")


if __name__ == "__main__":
    main()
