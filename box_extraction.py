import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d


import cv2
from sklearn.cluster import DBSCAN
from skimage.color import deltaE_ciede2000
import time


import numpy as np
import open3d as o3d


def extract_biggest_cluster(
    cloud: o3d.geometry.PointCloud, eps: float = 0.5, min_points: int = 50
):
    labels = cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    point_clouds_group_idx = dict()
    noise_cloud_idx = []
    for i, group in enumerate(labels):
        if group == -1:
            noise_cloud_idx.append(i)
            continue
        group_list = point_clouds_group_idx.get(group)
        if group_list is None:
            point_clouds_group_idx[group] = [i]
            continue
        group_list.append(i)

    return max(point_clouds_group_idx.values(), key=lambda x: len(x))


def get_aligned_with_direction(
    cloud: o3d.geometry.PointCloud, direction: np.ndarray, tol_radians: float = 0.05
):
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    normals = np.asarray(cloud.normals)
    theta = normals @ direction.T
    theta = np.acos(theta / np.linalg.norm(normals, axis=1))

    return np.flatnonzero((theta < tol_radians) | (theta > (np.pi - tol_radians)))


def extract_plane(
    cloud: o3d.geometry.PointCloud,
    should_plot: bool = False,
    distance_threshold=0.01,
    ransac_n=3,
    num_iterations=100,
):
    plane, plane_idx = cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    plane /= np.linalg.norm(plane[:3])

    if should_plot:
        plane_cloud = cloud.select_by_index(plane_idx)
        plane_cloud.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([plane_cloud])

    return plane_idx, plane


def extract_table(
    cloud: o3d.geometry.PointCloud,
    should_plot: bool = False,
    distance_threshold=0.01,
    ransac_n=3,
    num_iterations=100,
):

    # ALTERNATIVA TENTAR ENCONTRAR PLANO MAIS PRÓXIMO AO CHÃO PARA FICAR MAIS ROBUSTO
    table_plane, table_idx = extract_plane(
        cloud, should_plot, distance_threshold, ransac_n, num_iterations
    )

    original_idx = original_idx[table_idx]

    table = cloud.select_by_index(table_idx)
    table_idx = table_idx[extract_biggest_cluster(table)]

    original_idx = original_idx[table_idx]

    if should_plot:
        table = table.select_by_index(table_idx)
        o3d.visualization.draw_geometries([table])
    return table_idx, table_plane


def extract_above_table(
    remaining_cloud: o3d.geometry.PointCloud,
    table_plane: np.ndarray,
    floor_plane: np.ndarray,
    should_plot: bool = False,
    tol_table: float = 0.02,
):
    vertical_direction = floor_plane[:3]
    table_plane *= np.sign(table_plane[:3] @ vertical_direction.T)
    table_normal = table_plane[:3]

    above_table_points = np.asarray(remaining_cloud.points)
    above_sign = -np.sign(table_plane[3] - floor_plane[3])
    above_table_mask = (
        (above_table_points @ table_normal.T + table_plane[3]) * above_sign
    ) > (tol_table)
    above_table_idx = np.flatnonzero(above_table_mask)

    if should_plot:
        print("vertical_direction:", vertical_direction.ravel())
        print("table_normal     :", table_normal)
        print("table_plane[3]   :", table_plane[3], " floor_plane[3]:", floor_plane[3])
        print("s                :", np.sign(table_plane[3] - floor_plane[3]))
    return above_table_idx


def extracted_seeded_cloud(
    cloud_seed: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    tol: float = 0.02,
):
    combined = cloud_seed + target_cloud
    n_seed = len(cloud_seed.points)
    labels = np.asarray(combined.cluster_dbscan(eps=tol, min_points=30))

    seed_labels = labels[:n_seed]
    seed_labels = seed_labels[seed_labels >= 0]
    biggest_label = np.bincount(seed_labels).argmax()

    cluster_idx = np.flatnonzero(labels == biggest_label)
    clusters = combined.select_by_index(cluster_idx)

    # Remove Seed Cloud
    clusters_idx = cluster_idx[cluster_idx >= n_seed]
    clusters = combined.select_by_index(clusters_idx)

    return clusters


def extract_above_table_clusters(
    cloud: o3d.geometry.PointCloud, should_plot: bool = False
):
    floor_idx, floor_plane = extract_plane(cloud, should_plot)
    vertical_direction = floor_plane[:3]

    cloud_without_floor = cloud.select_by_index(floor_idx, invert=True)
    original_idx = np.arange(len(cloud_without_floor.points))

    vertical_idx = get_aligned_with_direction(cloud_without_floor, vertical_direction)
    original_idx = original_idx[vertical_idx]
    vertical_cloud = cloud_without_floor.select_by_index(vertical_idx)
    if should_plot:
        o3d.visualization.draw_geometries([vertical_cloud], point_show_normal=False)

    table_idx, table_plane = extract_plane(vertical_cloud, should_plot)
    original_idx = original_idx[table_idx]

    table = cloud_without_floor.select_by_index(original_idx)
    remaining = cloud_without_floor.select_by_index(original_idx, invert=True)

    tol_table = 0.02
    above_table_idx = extract_above_table(
        remaining, table_plane, floor_plane, should_plot, tol_table
    )
    above_table_cloud = remaining.select_by_index(above_table_idx)

    if should_plot:
        table.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([table, above_table_cloud])

    clusters = extracted_seeded_cloud(table, above_table_cloud, tol_table * 1.5)

    if should_plot:
        o3d.visualization.draw_geometries([clusters])

    return clusters


def extract_box(cloud: o3d.geometry.PointCloud, should_plot: bool = False):
    clusters = extract_above_table_clusters(cloud, should_plot)
    # clusters = clusters.random_down_sample(0.1)


def main():
    cloud_path = Path(__file__).parent / "nuvem_teste.ply"

    point_cloud = o3d.io.read_point_cloud(cloud_path)
    o3d.visualization.draw_geometries([point_cloud])

    extract_box(point_cloud, should_plot=True)
    # boxes = find_boxes(clusters.points, clusters.colors,
    #                geo_eps=0.03, plane_tol=0.5, de_thresh=8.0)
    # palette = np.array([
    # [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    # [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
    # [1.0, 0.5, 0.0], [0.5, 0.0, 1.0], [0.0, 0.5, 0.5],
    # ])

    # pts = np.asarray(clusters.points)     # (N,3)
    # geoms = []
    # for k, b in enumerate(boxes):
    #     idx = b['indices']                # indices into `clusters` (same array you fed find_boxes)
    #     sub = o3d.geometry.PointCloud()
    #     sub.points = o3d.utility.Vector3dVector(pts[idx])
    #     sub.colors = o3d.utility.Vector3dVector(
    #         np.tile(palette[k % len(palette)], (len(idx), 1)))
    #     geoms.append(sub)

    # o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
