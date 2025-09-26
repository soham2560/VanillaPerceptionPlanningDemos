import argparse
import json
import numpy as np
import open3d as o3d
from pathlib import Path
import copy

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate ground truth data where all frames see the same set of points.")
    parser.add_argument("--num_map_points", type=int, default=2000, help="Number of points to sample for the global ground truth map.")
    parser.add_argument("--num_poses", type=int, default=75, help="Number of sensor poses to generate.")
    parser.add_argument("--output_dir", type=str, default="slam_data", help="Directory to save the generated data.")
    parser.add_argument("--visualize", action="store_true", help="Display a clean visualization of the generated data.")
    return parser.parse_args()

def create_look_at_pose(eye, target, up=np.array([0, 0, 1])):
    x_axis = target - eye
    x_axis /= np.linalg.norm(x_axis)

    z_axis = up
    if np.abs(np.dot(x_axis, z_axis)) > 0.999:
        z_axis = np.array([0, 1, 0])
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    
    z_axis = np.cross(x_axis, y_axis)
    
    pose = np.identity(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye

    return pose


def generate_drifting_elliptical_trajectory(num_poses, center, radius_x, radius_z, height_variation, drift_vector):
    poses = []
    total_azimuth_sweep = 3.0 * np.pi

    for i in range(num_poses):
        progress = i / (num_poses - 1)
        current_center = center + progress * drift_vector
        azimuth = progress * total_azimuth_sweep

        x = radius_x * np.cos(azimuth)
        z = radius_z * np.sin(azimuth)
        y = height_variation * np.sin(progress * 4 * np.pi)

        eye_position = current_center + np.array([x, y, z])
        pose = create_look_at_pose(eye_position, center)
        poses.append(pose)

    return poses


def main():
    args = parse_arguments()
    output_path = Path(args.output_dir)
    gt_path = output_path / "ground_truth"
    pcd_path = gt_path / "pcds"
    pcd_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading the default bunny mesh...")
    bunny_path = o3d.data.BunnyMesh().path
    mesh = o3d.io.read_triangle_mesh(bunny_path)

    print(f"Sampling a single ground truth map with {args.num_map_points} points...")
    map_gt = mesh.sample_points_uniformly(number_of_points=args.num_map_points)
    center = map_gt.get_center()
    map_gt.translate(-center, relative=True)

    map_gt_path = gt_path / "map_gt.pcd"
    o3d.io.write_point_cloud(str(map_gt_path), map_gt)
    print(f"Ground truth map saved to: {map_gt_path}")

    aabb = map_gt.get_axis_aligned_bounding_box()
    object_radius = np.linalg.norm(aabb.get_max_bound() - aabb.get_center())

    print("Generating sensor trajectory...")
    poses_gt = generate_drifting_elliptical_trajectory(
        num_poses=args.num_poses,
        center=np.array([0, 0, 0]),
        radius_x=object_radius * 2.0,
        radius_z=object_radius * 1.3,
        height_variation=object_radius * 0.4,
        drift_vector=np.array([object_radius * 0.6, -object_radius * 0.3, object_radius * 0.5])
    )

    all_poses_for_json = {}
    print(f"Generating {args.num_poses} local scans...")
    for i, pose in enumerate(poses_gt):
        view_matrix = np.linalg.inv(pose)
        local_scan = copy.deepcopy(map_gt)
        local_scan.transform(view_matrix)
        
        frame_name = f"frame_{i:04d}"
        frame_filename = pcd_path / f"{frame_name}.pcd"
        o3d.io.write_point_cloud(str(frame_filename), local_scan)
        
        all_poses_for_json[frame_name] = pose.tolist()

    poses_file = gt_path / "poses_gt.json"
    with open(poses_file, "w") as f:
        json.dump(all_poses_for_json, f, indent=4)
        
    print("-" * 30)
    print(f"Successfully generated {args.num_poses} ground truth frames.")
    print(f"Data saved in: {gt_path}")
    print("-" * 30)

    if args.visualize:
        print("Preparing clean visualization...")
        geometries = []

        map_gt.paint_uniform_color([0.7, 0.7, 0.7])
        map_gt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        geometries.append(map_gt)

        gt_points = [pose[:3, 3] for pose in poses_gt]
        gt_lines = [[i, i + 1] for i in range(len(gt_points) - 1)]
        gt_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(gt_points),
            lines=o3d.utility.Vector2iVector(gt_lines)
        )
        gt_line_set.paint_uniform_color([0.0, 0.8, 0.0])
        geometries.append(gt_line_set)

        axis_size = object_radius * 0.05
        for pose in poses_gt:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
            axis.transform(pose)
            geometries.append(axis)

        print("Showing visualization...")
        print("  - Gray Cloud = Ground Truth Map")
        print("  - Green Line = Ground Truth Trajectory")
        o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    main()