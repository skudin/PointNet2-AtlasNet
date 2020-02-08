import numpy as np
import open3d as o3d


NETWORK_RESULT_FILENAME = "data/debug_meshing/input/1_primitive_2500_points.npy"
OUTPUT_PREFIX = "data/debug_meshing/output/1_primitive_2500_points"
CAMERA_LOCATION = np.array([0.0, -100.0, 0.0])
EPS = 1e-8


def find_bad_normals(pcd, camera_location):
    bad_normals = list()

    for index in range(len(pcd.points)):
        point = pcd.points[index]
        normal = pcd.normals[index]

        radius_vector = camera_location - point
        radius_vector /= np.linalg.norm(radius_vector)

        normal /= np.linalg.norm(normal)

        dot_product = np.dot(radius_vector, normal)

        if dot_product < EPS:
            bad_normals.append((index, dot_product))
    pass


class Plane:
    def __init__(self, point, normal):
        self.normal = normal
        self._d = -np.dot(point, normal)

    def distance_to_point(self, point):
        return np.abs(np.dot(self.normal, point) + self._d) / np.sqrt(np.sum(np.square(self.normal)))


class AxisAlignedBoundingBox:
    def __init__(self, point_cloud):
        bounding_box_tmp = point_cloud.get_axis_aligned_bounding_box()
        min_bound = bounding_box_tmp.min_bound
        max_bound = bounding_box_tmp.max_bound

        self._planes = list()
        self._create_planes(min_bound, "min")
        self._create_planes(max_bound, "max")

    def get_nearest_plane(self, point):
        distances = -1.0 * np.ones((6, ))

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


def estimate_normals(point_cloud, radius=1.0, max_nn=30):
    bounding_box = AxisAlignedBoundingBox(point_cloud)

    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=max_nn))

    for i in range(len(point_cloud.points)):
        plane = bounding_box.get_nearest_plane(point_cloud.points[i])
        sgn = np.dot(point_cloud.normals[i], plane.normal)

        if sgn < -EPS:
            point_cloud.normals[i] *= -1.0


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
