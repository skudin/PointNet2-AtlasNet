from collections import Counter, defaultdict

import numpy as np
import open3d as o3d


NETWORK_RESULT_FILENAME = "data/debug_meshing/input/1_primitive_10000_points.npy"
OUTPUT_PREFIX = "data/debug_meshing/output/1_primitive_2500_points"
EPS = 1e-12


def compute_search_radius(point_cloud):
    nearest_neighbor_distance = np.asarray(point_cloud.compute_nearest_neighbor_distance())

    mean = np.mean(nearest_neighbor_distance)
    std = np.std(nearest_neighbor_distance)

    return 3.0 * (mean + 3.0 * std)


def fix_normals(point_cloud, max_iteration=5):
    kd_tree = o3d.geometry.KDTreeFlann(point_cloud)

    radius = compute_search_radius(point_cloud)

    changed_normals_counter = 1
    iteration_counter = 0
    while changed_normals_counter > 0 and iteration_counter < max_iteration:
        changed_normals_counter = 0
        iteration_counter += 1

        for i in range(len(point_cloud.points)):
            normal = np.asarray(point_cloud.normals[i])
            count, indices, distances  = kd_tree.search_radius_vector_3d(query=point_cloud.points[i], radius=radius)

            sgn = 0
            for neighbor_num in indices:
                if neighbor_num == i:
                    continue

                dot = np.dot(normal, np.asarray(point_cloud.normals[neighbor_num]))
                if dot < -EPS:
                    sgn -= 1
                elif dot > EPS:
                    sgn += 1

            if sgn < 0:
                point_cloud.normals[i] = -normal
                changed_normals_counter += 1


def estimate_normals(point_cloud, radius=0.5, max_nn=30):
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=max_nn))

    point_cloud.orient_normals_towards_camera_location(camera_location=point_cloud.get_center())

    normals = np.asarray(point_cloud.normals)
    normals *= -1
    point_cloud.normals = o3d.utility.Vector3dVector(normals)

    fix_normals(point_cloud)


def compute_threshold(point_cloud):
    nearest_neighbor_distance = np.asarray(point_cloud.compute_nearest_neighbor_distance())

    mean = np.mean(nearest_neighbor_distance)
    std = np.std(nearest_neighbor_distance)

    return 2.0 * (mean + 3.0 * std)


def create_mesh(point_cloud, depth=9, scale=1.1):
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth, scale=scale)
    mesh = mesh.subdivide_midpoint(number_of_iterations=1)

    mesh_point_cloud = o3d.geometry.PointCloud(mesh.vertices)

    threshold = compute_threshold(point_cloud)
    distances = np.asarray(mesh_point_cloud.compute_point_cloud_distance(target=point_cloud))

    vertex_indices_to_remove = list()
    for i in range(len(mesh.vertices)):
        if distances[i] > threshold:
            vertex_indices_to_remove.append(i)

    mesh.remove_vertices_by_index(vertex_indices=vertex_indices_to_remove)

    return mesh


def main():
    point_cloud_np = np.load(NETWORK_RESULT_FILENAME).squeeze()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

    o3d.io.write_point_cloud(OUTPUT_PREFIX + "_point_cloud.ply", pcd)

    estimate_normals(pcd)

    o3d.io.write_point_cloud(OUTPUT_PREFIX + "_point_cloud_with_normals.ply", pcd)

    mesh = create_mesh(pcd)

    o3d.io.write_triangle_mesh(OUTPUT_PREFIX + "_mesh.ply", mesh, write_ascii=True,
                               write_vertex_colors=False)

    print("Done.")


if __name__ == "__main__":
    main()
