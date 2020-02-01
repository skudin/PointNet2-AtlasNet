import bisect
import os.path as osp

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

import atlasnet2.configuration as conf


MAX_MIN_VALUE = 1e5


class Plane:
    def __init__(self, point, normal):
        self.normal = normal
        self._d = -np.dot(point, normal)

    def distance_to_point(self, point):
        return np.abs(np.dot(self.normal, point) + self._d) / np.sqrt(np.sum(np.square(self.normal)))


class AxisAlignedBoundingBox:
    def __init__(self, point_cloud):
        self._init_planes(point_cloud)

        pass

    def _init_planes(self, point_cloud):
        bounding_box_tmp = point_cloud.get_axis_aligned_bounding_box()
        min_bound = bounding_box_tmp.min_bound
        max_bound = bounding_box_tmp.max_bound

        self._planes = list()
        self._create_planes(min_bound, "min")
        self._create_planes(max_bound, "max")

    def get_nearest_plane(self, point):
        distances = -1.0 * np.ones((6,))

        for i in range(6):
            distances[i] = self._planes[i].distance_to_point(point)

        return self._planes[np.argmin(distances)]

    def _create_planes(self, point, point_type):
        value = 1.0
        if point_type == "min":
            value = -1.0
        normals = (np.array((value, 0.0, 0.0)), np.array((0.0, value, 0.0)), np.array((0.0, 0.0, value)))

        for normal in normals:
            self._planes.append(Plane(point, normal))


def estimate_normals_by_bonding_box(point_cloud):
    bounding_box = AxisAlignedBoundingBox(point_cloud)

    for i in range(len(point_cloud.points)):
        plane = bounding_box.get_nearest_plane(point_cloud.points[i])
        sgn = np.dot(point_cloud.normals[i], plane.normal)

        if sgn < -conf.EPS:
            point_cloud.normals[i] *= -1.0

    return np.asarray(point_cloud.normals).copy()


def estimate_normals_by_camera_location(point_cloud):
    point_cloud.orient_normals_towards_camera_location(camera_location=point_cloud.get_center())

    normals = np.asarray(point_cloud.normals)
    normals *= -1
    point_cloud.normals = o3d.utility.Vector3dVector(normals)

    return np.asarray(point_cloud.normals).copy()


def estimate_normals_by_direction(point_cloud):
    point_cloud.orient_normals_to_align_with_direction(orientation_reference=(0.0, 1.0, 0.0))

    return np.asarray(point_cloud.normals).copy()


def compute_search_radius(point_cloud):
    nearest_neighbor_distance = np.asarray(point_cloud.compute_nearest_neighbor_distance())

    mean = np.mean(nearest_neighbor_distance)
    std = np.std(nearest_neighbor_distance)

    return 3.0 * (mean + 3.0 * std)


def fix_normals(pcd, max_iteration=10):
    kd_tree = o3d.geometry.KDTreeFlann(pcd)
    radius = compute_search_radius(pcd)

    point_cloud = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    changed_normals_counter = 1
    iteration_counter = 0
    while changed_normals_counter > 0 and iteration_counter < max_iteration:
        changed_normals_counter = 0
        iteration_counter += 1

        for i in range(len(point_cloud)):
            count, indices, distances = kd_tree.search_radius_vector_3d(query=point_cloud[i], radius=radius)

            neighbor_normals = normals[indices[1:]]
            dots = np.dot(neighbor_normals, normals[i])
            codirectional_count = np.count_nonzero(dots > -conf.EPS)

            if codirectional_count < dots.shape[0] - codirectional_count:
                normals[i] *= -1
                changed_normals_counter += 1

    pcd.normals = o3d.utility.Vector3dVector(normals)


def estimate_normals(point_cloud, radius=0.5, max_nn=30):
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=max_nn))

    normals_by_camera_location = estimate_normals_by_camera_location(point_cloud)
    normals_by_direction = estimate_normals_by_direction(point_cloud)
    normals_by_bounding_box = estimate_normals_by_bonding_box(point_cloud)

    normals = list()
    for i in range(normals_by_camera_location.shape[0]):
        counter = 0
        variants = [normals_by_direction[i], normals_by_bounding_box[i]]
        candidate = normals_by_camera_location[i]
        for vec_num in range(2):
            if np.dot(candidate, variants[vec_num]) > conf.EPS:
                counter += 1

        if counter > 0:
            normals.append(candidate)
        else:
            normals.append(-candidate)

    point_cloud.normals = o3d.utility.Vector3dVector(normals)

    fix_normals(point_cloud, max_iteration=10)


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


def wax_up_meshing(point_cloud, margin_approx_points_number=25, debug_output_dir=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    estimate_normals(pcd)

    if debug_output_dir is not None:
        o3d.io.write_point_cloud(osp.join(debug_output_dir, "point_cloud_with_normals.ply"), pcd)

    mesh = create_mesh(point_cloud=pcd, margin_approx_points_number=margin_approx_points_number,
                       output_dir=debug_output_dir)

    if debug_output_dir is not None:
        o3d.io.write_triangle_mesh(osp.join(debug_output_dir, "mesh.ply"), mesh, write_ascii=True,
                                   write_vertex_colors=False)

    return mesh


def meshing(point_cloud, radius=0.5, max_nn=30, depth=9, scale=1.1, output_dir=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=max_nn), fast_normal_computation=False)

    estimate_normals_by_camera_location(pcd)

    if output_dir is not None:
        o3d.io.write_point_cloud(osp.join(output_dir, "point_cloud_with_normals.ply"), pcd)

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd, depth=depth, scale=scale)

    if output_dir is not None:
        o3d.io.write_triangle_mesh(osp.join(output_dir, "mesh.ply"), mesh, write_ascii=True, write_vertex_colors=False)

    return mesh
