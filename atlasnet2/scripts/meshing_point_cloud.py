import numpy as np
import open3d as o3d


def main():
    point_cloud_np = np.load("data/debug_meshing/1_primitive_10000_points.npy").squeeze()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=1.0, max_nn=30), fast_normal_computation=False)
    pcd.orient_normals_towards_camera_location(camera_location=[0.0, 0.0, 0.0])

    normals = np.asarray(pcd.normals)
    pcd.normals = o3d.utility.Vector3dVector(-normals)

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, scale=1.1)
    mesh_2 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii=o3d.utility.DoubleVector((0.1, 0.2)))
    o3d.io.write_triangle_mesh("data/debug_meshing/1_primitive_2500_points.ply", mesh, write_ascii=True,
                               write_vertex_colors=False)
    o3d.io.write_triangle_mesh("data/debug_meshing/1_primitive_2500_points_cloud_ball_pivoting.ply", mesh_2, write_ascii=True,
                               write_vertex_colors=False)
    o3d.io.write_point_cloud("data/debug_meshing/1_primitive_2500_points_point_cloud.ply", pcd)
    print("Hello world!")


if __name__ == "__main__":
    main()
