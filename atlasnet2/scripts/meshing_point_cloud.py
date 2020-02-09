from collections import Counter, defaultdict

import numpy as np
import open3d as o3d


NETWORK_RESULT_FILENAME = "data/debug_meshing/input/1_primitive_10000_points.npy"
OUTPUT_PREFIX = "data/debug_meshing/output/1_primitive_2500_points"
CAMERA_LOCATION = np.array([0.0, -100.0, 0.0])
EPS = 1e-8
STEP = 10.0


def estimate_normals(point_cloud, radius=0.5, max_nn=30):
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=max_nn))

    point_cloud.orient_normals_towards_camera_location(camera_location=point_cloud.get_center())

    normals = np.asarray(point_cloud.normals)
    normals *= -1
    point_cloud.normals = o3d.utility.Vector3dVector(normals)


def main():
    point_cloud_np = np.load(NETWORK_RESULT_FILENAME).squeeze()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

    o3d.io.write_point_cloud(OUTPUT_PREFIX + "_point_cloud.ply", pcd)

    estimate_normals(pcd)

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
