import numpy as np
import open3d as o3d


def main():
    NETWORK_RESULT_FILENAME = "data/debug_meshing/input/1_primitive_2500_points.npy"
    OUTPUT_PREFIX = "data/debug_meshing/output/1_primitive_2500_points"

    point_cloud_np = np.load(NETWORK_RESULT_FILENAME).squeeze()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=1.0, max_nn=30), fast_normal_computation=True)

    o3d.io.write_point_cloud(OUTPUT_PREFIX + "_point_cloud_with_raw_normals.ply", pcd)

    pcd.orient_normals_towards_camera_location(camera_location=[0.0, 0.0, 0.0])

    o3d.io.write_point_cloud(OUTPUT_PREFIX + "_point_cloud_with_normals_after_orient.ply", pcd)

    normals = np.asarray(pcd.normals)
    pcd.normals = o3d.utility.Vector3dVector(-normals)

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
