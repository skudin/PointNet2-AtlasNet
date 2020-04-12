import bisect
import time

import numpy as np
import open3d as o3d
import shapely.geometry as geom
import shapely.ops as ops
from matplotlib import pyplot as plt


NETWORK_RESULT_FILENAME = "data/debug_meshing/input/1_primitive_10000_points.npy"
OUTPUT_PREFIX = "data/debug_meshing/output/1_primitive_10000_points"
EPS = 1e-12
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


def estimate_normals_by_bonding_box(point_cloud, radius=1.0, max_nn=30):
    bounding_box = AxisAlignedBoundingBox(point_cloud)

    for i in range(len(point_cloud.points)):
        plane = bounding_box.get_nearest_plane(point_cloud.points[i])
        sgn = np.dot(point_cloud.normals[i], plane.normal)

        if sgn < -EPS:
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

            sgn = 0
            for neighbor_num in indices:
                if neighbor_num == i:
                    continue

                dot = np.dot(normals[i], normals[neighbor_num])
                if dot < -EPS:
                    sgn -= 1
                elif dot > EPS:
                    sgn += 1

            if sgn < 0:
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
            if np.dot(candidate, variants[vec_num]) > EPS:
                counter += 1

        if counter > 0:
            normals.append(candidate)
        else:
            normals.append(-candidate)

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


def get_cylindrical_projection(points):
    projection = [(float(np.arctan2(point[2], point[0])), point[1]) for num, point in
                  enumerate(points)]
    # projections.sort(key=itemgetter(0, 1))
    # projections = np.array(projections, dtype=np.float64).reshape(2, len(points))

    return projection


def create_triangulation_image(polygons, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("phi")
    ax.set_ylabel("y")

    for polygon in polygons:
        x = [point[0] for point in polygon.boundary.coords]
        x.append(x[0])
        y = [point[1] for point in polygon.boundary.coords]
        y.append(y[0])

        ax.plot(x, y)

    fig.savefig(OUTPUT_PREFIX + "_" + name + ".png")


def get_margin(projection):
    # Delone's triangulation.
    point_cloud_triangulation = ops.triangulate(geom.MultiPoint(projection), tolerance=1.0)
    union_polygon = ops.unary_union(point_cloud_triangulation)

    create_triangulation_image(point_cloud_triangulation, "triangulation")
    create_triangulation_image([union_polygon], "union_polygon")

    print("Debug")

    # Debug.
    # fig = pyplot.figure(1, figsize=(600, 600), dpi=90)
    # fig.set_frameon(True)
    # ax = fig.add_subplot(111)
    #
    # for triangle in point_cloud_triangulation:
    #     patch = PolygonPatch(triangle, facecolor="blue", edgecolor="blue", alpha=0.5, zorder=2)
    #     ax.add_patch(patch)
    #
    # tmp_points = geom.MultiPoint(projection)
    # for point in tmp_points:
    #     pyplot.plot(point.x, point.y, 'o', color="gray")
    #
    # pyplot.savefig(OUTPUT_PREFIX + "triangulation.png")
    #
    # boundary = ops.orient(union_polygon.boundary, sign=1.0)
    # bounds = union_polygon.bounds

    # margin = list()
    # start_point = (bounds[0], bounds[1])
    # for point in boundary.coords:
    #     if point == start_point:
    #         pass

    pass


def create_cylindrical_proj_image(points, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("phi")
    ax.set_ylabel("y")

    x = [point[0] for point in points]
    y = [point[1] for point in points]

    ax.scatter(x, y, s=0.5)
    fig.savefig(OUTPUT_PREFIX + "_" + name + ".png")


def create_mesh_with_margin(point_cloud, depth=9, scale=1.1):
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth, scale=scale)

    point_cloud_proj = get_cylindrical_projection(point_cloud.points)
    mesh_points_proj = get_cylindrical_projection(mesh.vertices)

    create_cylindrical_proj_image(point_cloud_proj, "point_cloud")
    create_cylindrical_proj_image(mesh_points_proj, "mesh_points")

    margin = get_margin(point_cloud_proj)

    vertex_indices_to_remove = list()
    for i in range(len(mesh_points_proj)):
        point = geom.Point(mesh_points_proj[i])
        if not point.within(union_polygon):
            vertex_indices_to_remove.append(i)

    mesh.remove_vertices_by_index(vertex_indices=vertex_indices_to_remove)

    return mesh


def create_mesh_using_dencity(point_cloud, depth=10, scale=1.1):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth, scale=scale)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.04)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh

    # densities = np.asarray(densities)
    # density_colors = plt.get_cmap('plasma')(
    #     (densities - densities.min()) / (densities.max() - densities.min()))
    # density_colors = density_colors[:, :3]
    # density_mesh = o3d.geometry.TriangleMesh()
    # density_mesh.vertices = mesh.vertices
    # density_mesh.triangles = mesh.triangles
    # density_mesh.triangle_normals = mesh.triangle_normals
    # density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)

    # return density_mesh


def create_margin_approximation(proj, approximation_points_number):
    start_point = -np.pi - EPS
    end_point = np.pi + EPS
    segment_len = (end_point - start_point) / (approximation_points_number - 1)

    margin = list()
    start_segment_point = start_point
    for i in range(approximation_points_number):
        end_segment_point = start_segment_point + segment_len

        min_value = MAX_MIN_VALUE
        phi = None
        for point in proj:
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


def draw_margin_with_cylindrical_projection(points, margin, name):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("phi")
    ax.set_ylabel("y")

    x = [point[0] for point in points]
    y = [point[1] for point in points]

    ax.scatter(x, y, s=0.5)

    x = [point[0] for point in margin]
    y = [point[1] for point in margin]

    ax.plot(x, y)

    fig.savefig(OUTPUT_PREFIX + "_" + name + ".png")


def cut_mesh(mesh, mesh_points_proj, margin):
    base_points = [point[0] for point in margin]
    vertex_indices_to_remove = list()
    for num, point in enumerate(mesh_points_proj):
        segment_num = bisect.bisect_left(base_points, point[0]) - 1
        a, c = margin[segment_num][2], margin[segment_num][3]
        border = a * point[0] + c

        if point[1] < border - EPS:
            vertex_indices_to_remove.append(num)

    mesh.remove_vertices_by_index(vertex_indices=vertex_indices_to_remove)

    return mesh


def create_mesh_using_cylindrical_projection_and_margin_approximation(point_cloud, depth=9, scale=1.1,
                                                                      approximation_points_number=25):
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth, scale=scale)

    point_cloud_proj = get_cylindrical_projection(point_cloud.points)
    mesh_points_proj = get_cylindrical_projection(mesh.vertices)

    # create_cylindrical_proj_image(point_cloud_proj, "point_cloud")
    # create_cylindrical_proj_image(mesh_points_proj, "mesh_points")

    margin = create_margin_approximation(point_cloud_proj, approximation_points_number)

    # draw_margin_with_cylindrical_projection(point_cloud_proj, margin, "point_cloud_with_margin")
    # draw_margin_with_cylindrical_projection(mesh_points_proj, margin, "mesh_points_with_margin")

    mesh = cut_mesh(mesh, mesh_points_proj, margin)

    return mesh


def main():
    point_cloud_np = np.load(NETWORK_RESULT_FILENAME).squeeze()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

    o3d.io.write_point_cloud(OUTPUT_PREFIX + "_point_cloud.ply", pcd)

    start_time = time.time()

    estimate_normals(pcd)

    # o3d.io.write_point_cloud(OUTPUT_PREFIX + "_point_cloud_with_normals.ply", pcd)

    # mesh = create_mesh(pcd)
    # mesh = create_mesh_with_margin(pcd)
    # mesh = create_mesh_using_dencity(pcd)
    mesh = create_mesh_using_cylindrical_projection_and_margin_approximation(pcd, approximation_points_number=50)

    print("Spent time: %f" % (time.time() - start_time))

    o3d.io.write_triangle_mesh(OUTPUT_PREFIX + "_mesh.ply", mesh, write_ascii=True,
                               write_vertex_colors=False)

    print("Done.")


if __name__ == "__main__":
    main()
