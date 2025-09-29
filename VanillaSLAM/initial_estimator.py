import argparse
import json
import numpy as np
import open3d as o3d
from pathlib import Path
import copy

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate the initial problem file for Ceres optimization.")
    parser.add_argument("--data_dir", type=str, default="slam_data", help="Directory containing the ground truth data.")
    parser.add_argument("--noise_std_dev", type=float, default=0.0105, help="Standard deviation of Gaussian noise added to scans.")
    parser.add_argument("--icp_max_correspondence", type=float, default=0.1, help="Max correspondence distance for ICP.")
    parser.add_argument("--output_file", type=str, default="optimization_problem.json", help="Final output file for the C++ optimizer.")
    parser.add_argument("--visualize", action="store_true", help="Display an interactive visualization to switch between GT and Estimate.")
    parser.add_argument("--averaging_window_size", type=int, default=5, help="Number of recent frames to average for each map point.")
    return parser.parse_args()

def add_noise_to_pcd(pcd, noise_std_dev):
    points = np.asarray(pcd.points)
    noise = np.random.normal(scale=noise_std_dev, size=points.shape)
    pcd.points = o3d.utility.Vector3dVector(points + noise)
    return pcd

def flatten_pose_matrix(pose_matrix):
    pose_array = np.array(pose_matrix)
    return pose_array.flatten().tolist()

def transform_points_to_world(points_sensor, pose_world_sensor):
    points_homogeneous = np.hstack([points_sensor, np.ones((points_sensor.shape[0], 1))])
    points_world = (pose_world_sensor @ points_homogeneous.T).T
    return points_world[:, :3]

def visualize_interactive_switch(map_gt, map_est, gt_line_set, est_line_set):
    active_map_pcd = o3d.geometry.PointCloud(map_gt)
    state = { "is_showing_gt": True }

    def toggle_view(vis):
        state["is_showing_gt"] = not state["is_showing_gt"]
        
        if state["is_showing_gt"]:
            active_map_pcd.points = map_gt.points
            active_map_pcd.colors = map_gt.colors
            active_map_pcd.normals = map_gt.normals
        else:
            active_map_pcd.points = map_est.points
            active_map_pcd.colors = map_est.colors
            active_map_pcd.normals = map_est.normals

        vis.update_geometry(active_map_pcd)
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    vis.add_geometry(active_map_pcd)
    vis.add_geometry(gt_line_set)
    vis.add_geometry(est_line_set)
    
    vis.register_key_callback(ord("S"), toggle_view)
    
    vis.run()
    vis.destroy_window()

def main():
    args = parse_arguments()
    data_path = Path(args.data_dir)
    gt_path = data_path / "ground_truth"
    pcd_gt_path = gt_path / "pcds"
    
    with open(gt_path / "poses_gt.json", 'r') as f:
        poses_gt = json.load(f)

    frame_names = sorted(poses_gt.keys())
    
    unaligned_poses = {}
    last_absolute_pose = np.identity(4)
    unaligned_poses[frame_names[0]] = last_absolute_pose.tolist()

    pcd_prev_local = o3d.io.read_point_cloud(str(pcd_gt_path / f"{frame_names[0]}.pcd"))
    last_relative_transform = np.identity(4)

    for i in range(len(frame_names) - 1):
        source_frame_name = frame_names[i+1]
        
        pcd_source_local = o3d.io.read_point_cloud(str(pcd_gt_path / f"{source_frame_name}.pcd"))
        
        pcd_source_noisy = add_noise_to_pcd(copy.deepcopy(pcd_source_local), args.noise_std_dev)
        pcd_target_noisy = add_noise_to_pcd(copy.deepcopy(pcd_prev_local), args.noise_std_dev)
        
        pcd_source_noisy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd_target_noisy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        reg_p2l = o3d.pipelines.registration.registration_icp(
            pcd_source_noisy, pcd_target_noisy, args.icp_max_correspondence,
            last_relative_transform, o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        relative_transform = reg_p2l.transformation
        
        current_absolute_pose = last_absolute_pose @ relative_transform
        
        unaligned_poses[source_frame_name] = current_absolute_pose.tolist()
        
        pcd_prev_local = pcd_source_local
        last_absolute_pose = current_absolute_pose
        last_relative_transform = relative_transform
    
    first_gt_pose = np.array(poses_gt[frame_names[0]])
    alignment_transform = first_gt_pose
    
    aligned_poses = {}
    for name, pose in unaligned_poses.items():
        aligned_pose = alignment_transform @ np.array(pose)
        aligned_poses[name] = aligned_pose.tolist()

    num_map_points = len(pcd_prev_local.points)
    all_point_estimates = np.zeros((num_map_points, len(frame_names), 3))

    for i, name in enumerate(frame_names):
        pcd_local = o3d.io.read_point_cloud(str(pcd_gt_path / f"{name}.pcd"))
        pcd_local_noisy = add_noise_to_pcd(pcd_local, args.noise_std_dev)
        
        pose_world_sensor = np.array(aligned_poses[name])
        
        points_sensor = np.asarray(pcd_local_noisy.points)
        points_world = transform_points_to_world(points_sensor, pose_world_sensor)
        
        all_point_estimates[:, i, :] = points_world

    window_size = args.averaging_window_size
    map_points_estimated = np.zeros_like(all_point_estimates[:, 0, :])
    map_points_estimated = np.mean(all_point_estimates[:, 0:window_size, :], axis=1)
    
    observations = []
    for pose_id, pose_name in enumerate(frame_names):
        scan_points = np.asarray(o3d.io.read_point_cloud(str(pcd_gt_path / f"{pose_name}.pcd")).points)
        for point_idx in range(num_map_points):
            observations.append({
                "pose_id": pose_id, 
                "point_id": point_idx, 
                "observed_xyz": scan_points[point_idx].tolist()
            })
    
    poses_flattened = []
    for name in frame_names:
        pose_matrix = aligned_poses[name]
        pose_flattened = flatten_pose_matrix(pose_matrix)
        poses_flattened.append({"name": name, "pose": pose_flattened})
    
    optimization_problem = {
        "coordinate_system": {
            "description": "Right-handed coordinate system",
            "world_frame": "Z-up, X-forward, Y-left", 
            "pose_convention": "T_world_sensor (transforms sensor to world)",
            "observations": "Points in local sensor frame"
        },
        "poses": poses_flattened,
        "points": map_points_estimated.tolist(),
        "observations": observations
    }

    output_filepath = data_path / args.output_file
    with open(output_filepath, 'w') as f: 
        json.dump(optimization_problem, f, indent=2)

    if args.visualize:
        map_gt = o3d.io.read_point_cloud(str(gt_path / "map_gt.pcd"))
        map_gt.paint_uniform_color([0.6, 0.8, 1.0])
        map_gt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        
        map_est_pcd = o3d.geometry.PointCloud()
        map_est_pcd.points = o3d.utility.Vector3dVector(map_points_estimated)
        map_est_pcd.paint_uniform_color([1.0, 0.8, 0.6])
        map_est_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        
        gt_points = []
        est_points = []
        
        for name in frame_names:
            gt_pose = np.array(poses_gt[name])
            est_pose = np.array(aligned_poses[name])
            
            gt_points.append(gt_pose[:3, 3])
            est_points.append(est_pose[:3, 3])
        
        gt_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(gt_points), 
            lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(gt_points) - 1)])
        )
        gt_line_set.paint_uniform_color([0.0, 0.8, 0.0])
        
        est_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(est_points), 
            lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(est_points) - 1)])
        )
        est_line_set.paint_uniform_color([1.0, 0.2, 0.2])

        visualize_interactive_switch(map_gt, map_est_pcd, gt_line_set, est_line_set)

if __name__ == "__main__":
    main()