import bisect
import os.path as osp

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

import atlasnet2.configuration as conf


MAX_MIN_VALUE = 1e5


def compute_search_radius(point_cloud):
    nearest_neighbor_distance = np.asarray(point_cloud.compute_nearest_neighbor_distance())

    mean = np.mean(nearest_neighbor_distance)
    std = np.std(nearest_neighbor_distance)

    return 3.0 * (mean + 3.0 * std)


def fix_normals(point_cloud, max_iteration=10):
    kd_tree = o3d.geometry.KDTreeFlann(point_cloud)

    radius = compute_search_radius(point_cloud)

    changed_normals_counter = 1
    iteration_counter = 0
    while changed_normals_counter > 0 and iteration_counter < max_iteration:
        changed_normals_counter = 0
        iteration_counter += 1

        for i in range(len(point_cloud.points)):
            normal = np.asarray(point_cloud.normals[i])
            count, indices, distances = kd_tree.search_radius_vector_3d(query=point_cloud.points[i], radius=radius)

            sgn = 0
            for neighbor_num in indices:
                if neighbor_num == i:
                    continue

                dot = np.dot(normal, np.asarray(point_cloud.normals[neighbor_num]))
                if dot < -conf.EPS:
                    sgn -= 1
                elif dot > conf.EPS:
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


def create_cylindrical_proj_image(points, output_dir, image_name):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("phi")
    ax.set_ylabel("y")

    ax.scatter([point[0] for point in points], [point[1] for point in points], s=0.5)
    fig.savefig(osp.join(output_dir, "%s.png" % image_name))


def create_margin_with_cylindrical_projection_image(points, margin, output_dir, image_name):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("phi")
    ax.set_ylabel("y")

    ax.scatter([point[0] for point in points], [point[1] for point in points], s=0.5)
    ax.plot([point[0] for point in margin], [point[1] for point in margin], color="red")

    fig.savefig(osp.join(output_dir, "%s.png" % image_name))


def get_cylindrical_projection(points):
    return [(np.arctan2(point[2], point[0]), point[1]) for num, point in enumerate(points)]


def create_margin_approximation(projection, approximation_points_number):
    start_point = -np.pi - conf.EPS
    end_point = np.pi + conf.EPS
    segment_len = (end_point - start_point) / (approximation_points_number - 1)

    margin = list()
    start_segment_point = start_point
    for i in range(approximation_points_number):
        end_segment_point = start_segment_point + segment_len

        min_value = MAX_MIN_VALUE
        phi = None
        for point in projection:
            if start_segment_point <= point[0] < end_segment_point:
                if point[1] < min_value:
                    min_value = point[1]
                    phi = point[0]

        if min_value != MAX_MIN_VALUE:
            margin.append((phi, min_value))
        start_segment_point = end_segment_point

    value = (margin[0][1] + margin[-1][1]) / 2
    margin.insert(0, (start_point, value))
    margin.append((end_point, value))

    for i in range(len(margin) - 1):
        start_point = margin[i]
        end_point = margin[i + 1]

        a = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
        c = start_point[1] - a * start_point[0]

        margin[i] = (*start_point, a, c)

    return margin


def cut_mesh(mesh, mesh_points_proj, margin):
    base_points = [point[0] for point in margin]
    vertex_indices_to_remove = list()
    for num, point in enumerate(mesh_points_proj):
        segment_num = bisect.bisect_left(base_points, point[0]) - 1
        a, c = margin[segment_num][2], margin[segment_num][3]
        border = a * point[0] + c

        if point[1] < border - conf.EPS:
            vertex_indices_to_remove.append(num)

    mesh.remove_vertices_by_index(vertex_indices=vertex_indices_to_remove)

    return mesh


def create_mesh(point_cloud, margin_approx_points_number=25, depth=9, scale=1.1, output_dir=None):
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth, scale=scale)

    point_cloud_proj = get_cylindrical_projection(point_cloud.points)
    mesh_points_proj = get_cylindrical_projection(mesh.vertices)

    if output_dir is not None:
        create_cylindrical_proj_image(point_cloud_proj, output_dir, "point_cloud_cylindrical_proj")
        create_cylindrical_proj_image(mesh_points_proj, output_dir, "mesh_vertexes_cylindrical_proj")

    margin = create_margin_approximation(point_cloud_proj, margin_approx_points_number)

    if output_dir is not None:
        create_margin_with_cylindrical_projection_image(point_cloud_proj, margin, output_dir, "point_cloud_with_margin")
        create_margin_with_cylindrical_projection_image(mesh_points_proj, margin, output_dir, "mesh_points_with_margin")

    mesh = cut_mesh(mesh, mesh_points_proj, margin)

    return mesh


def meshing(point_cloud, margin_approx_points_number=25, output_dir=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    estimate_normals(point_cloud)

    if output_dir is not None:
        o3d.io.write_point_cloud(osp.join(output_dir, "point_cloud_with_normals.ply"), pcd)

    mesh = create_mesh(point_cloud=pcd, margin_approx_points_number=margin_approx_points_number, output_dir=output_dir)

    if output_dir is not None:
        o3d.io.write_triangle_mesh(osp.join(output_dir, "mesh.ply"), mesh, write_ascii=True, write_vertex_colors=False)
