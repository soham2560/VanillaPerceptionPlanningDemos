import argparse
import json
import numpy as np
import open3d as o3d
from pathlib import Path
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize and compare initial, optimized, and ground truth SLAM results.")
    parser.add_argument("--data_dir", type=str, default="slam_data", help="Directory containing all SLAM data.")
    parser.add_argument("--initial_problem_file", type=str, default="optimization_problem.json", help="The initial problem file fed to Ceres.")
    parser.add_argument("--optimized_file", type=str, default="optimizer/optimized_results.json", help="The final output file from the Ceres optimizer.")
    return parser.parse_args()

def extract_pose_translation(pose_matrix):
    if isinstance(pose_matrix, list):
        pose_matrix = np.array(pose_matrix)
    
    if pose_matrix.shape == (16,):
        pose_matrix = pose_matrix.reshape(4, 4)
    
    return pose_matrix[:3, 3]

def create_trajectory_visualization(pose_positions, color, name):
    if len(pose_positions) < 2:
        return None
        
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pose_positions),
        lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(pose_positions) - 1)])
    )
    line_set.paint_uniform_color(color)
    return line_set

def create_point_cloud_visualization(points, color, name):
    if isinstance(points, list):
        points = np.array(points)
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color(color)
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    return point_cloud

def visualize_interactive_comparison(map_gt, map_initial, map_optimized, 
                                   line_gt, line_initial, line_optimized):
    
    active_map_pcd = o3d.geometry.PointCloud(map_gt)
    active_trajectory = o3d.geometry.LineSet(line_gt)
    
    state = {
        "mode": "gt_vs_initial",
        "showing_gt": True,
        "last_update": time.time(),
        "has_optimized": map_optimized is not None and len(map_optimized.points) > 0
    }

    def update_display(vis, show_gt):
        if show_gt:
            active_map_pcd.points = map_gt.points
            active_map_pcd.colors = map_gt.colors
            active_map_pcd.normals = map_gt.normals
            
            active_trajectory.points = line_gt.points
            active_trajectory.lines = line_gt.lines
            active_trajectory.colors = line_gt.colors
        else:
            if state["mode"] == "gt_vs_optimized" and state["has_optimized"]:
                active_map_pcd.points = map_optimized.points
                active_map_pcd.colors = map_optimized.colors  
                active_map_pcd.normals = map_optimized.normals
                
                active_trajectory.points = line_optimized.points
                active_trajectory.lines = line_optimized.lines
                active_trajectory.colors = line_optimized.colors
            else:
                active_map_pcd.points = map_initial.points
                active_map_pcd.colors = map_initial.colors
                active_map_pcd.normals = map_initial.normals
                
                active_trajectory.points = line_initial.points
                active_trajectory.lines = line_initial.lines
                active_trajectory.colors = line_initial.colors
        
        vis.update_geometry(active_map_pcd)
        vis.update_geometry(active_trajectory)

    def toggle_mode(vis):
        if state["has_optimized"]:
            if state["mode"] == "gt_vs_initial":
                state["mode"] = "gt_vs_optimized"
            else:
                state["mode"] = "gt_vs_initial"
            
            state["showing_gt"] = True
            state["last_update"] = time.time()
            update_display(vis, True)
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="SLAM Results Comparison", width=1200, height=800)
    
    vis.add_geometry(active_map_pcd)
    vis.add_geometry(active_trajectory)
    
    vis.register_key_callback(ord("S"), toggle_mode)
    
    update_display(vis, True)
    
    def animation_callback(vis):
        current_time = time.time()
        if current_time - state["last_update"] > 0.25:
            state["showing_gt"] = not state["showing_gt"]
            update_display(vis, state["showing_gt"])
            state["last_update"] = current_time
        return False

    vis.register_animation_callback(animation_callback)
    
    vis.run()
    vis.destroy_window()

def main():
    args = parse_arguments()
    data_path = Path(args.data_dir)
    gt_path = data_path / "ground_truth"
    
    with open(gt_path / "poses_gt.json", 'r') as f:
        poses_gt_data = json.load(f)
    
    frame_names = sorted(poses_gt_data.keys())
    
    gt_points = []
    for name in frame_names:
        pose_matrix = np.array(poses_gt_data[name])
        if pose_matrix.shape == (16,):
            pose_matrix = pose_matrix.reshape(4, 4)
        position = extract_pose_translation(pose_matrix)
        gt_points.append(position)
    
    gt_trajectory = create_trajectory_visualization(
        gt_points, [0.0, 1.0, 0.0], "Ground Truth"
    )
    
    gt_map_path = gt_path / "map_gt.pcd"
    if gt_map_path.exists():
        map_gt = o3d.io.read_point_cloud(str(gt_map_path))
        map_gt.paint_uniform_color([0.0, 1.0, 0.0])
        map_gt.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
    else:
        map_gt = o3d.geometry.PointCloud()
    
    initial_problem_path = data_path / args.initial_problem_file
    with open(initial_problem_path, 'r') as f:
        initial_problem_data = json.load(f)
    
    initial_poses = {item['name']: item['pose'] for item in initial_problem_data['poses']}
    
    initial_points = []
    for name in frame_names:
        if name in initial_poses:
            pose_matrix = np.array(initial_poses[name])
            position = extract_pose_translation(pose_matrix)
            initial_points.append(position)
    
    initial_trajectory = create_trajectory_visualization(
        initial_points, [1.0, 0.0, 0.0], "Initial Estimate"
    )
    
    map_initial = create_point_cloud_visualization(
        initial_problem_data['points'], [1.0, 0.0, 0.0], "Initial Map"
    )
    
    optimized_path = Path(args.optimized_file)
    map_optimized = None
    optimized_trajectory = None
    
    if optimized_path.exists():
        with open(optimized_path, 'r') as f:
            optimized_data = json.load(f)
        
        optimized_poses = optimized_data['poses_optimized']
        
        optimized_points = []
        for name in frame_names:
            if name in optimized_poses:
                pose_matrix = np.array(optimized_poses[name])
                position = extract_pose_translation(pose_matrix)
                optimized_points.append(position)
        
        optimized_trajectory = create_trajectory_visualization(
            optimized_points, [0.0, 0.0, 1.0], "Optimized Result"
        )
        
        map_optimized = create_point_cloud_visualization(
            optimized_data['map_points_optimized'], [0.0, 0.0, 1.0], "Optimized Map"
        )
    
    visualize_interactive_comparison(
        map_gt=map_gt,
        map_initial=map_initial,
        map_optimized=map_optimized,
        line_gt=gt_trajectory,
        line_initial=initial_trajectory,
        line_optimized=optimized_trajectory
    )

if __name__ == "__main__":
    main()