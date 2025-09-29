import argparse
import json
import numpy as np
import open3d as o3d
from pathlib import Path
import time
import glob

def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize the SLAM optimization process iteration by iteration.")
    parser.add_argument("--data_dir", type=str, default="slam_data", help="Directory for GT and initial problem data.")
    # This now correctly points to where the C++ program saves the final file by default
    parser.add_argument("--final_result_file", type=str, default="optimizer/build/optimized_results.json", help="Path to the final optimized result file.")
    return parser.parse_args()

def load_trajectory_and_map(file_path, frame_names):
    """Loads poses and map points from a JSON file, handling different formats."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Unified logic to handle initial, intermediate, and final file formats
    poses = data.get("poses_optimized") or (
        {item['name']: item['pose'] for item in data.get("poses", [])}
    )
    map_points = np.array(data.get("map_points_optimized") or data.get("points", []))

    if not poses or map_points.size == 0:
        return None, None

    pose_positions = []
    for name in frame_names:
        if name in poses:
            pose_matrix = np.array(poses[name]).reshape(4, 4)
            pose_positions.append(pose_matrix[:3, 3])
    
    return pose_positions, map_points

def main():
    args = parse_arguments()
    data_path = Path(args.data_dir)
    gt_path = data_path / "ground_truth"
    
    print("Loading reference data (Ground Truth)...")
    with open(gt_path / "poses_gt.json", 'r') as f:
        poses_gt_data = json.load(f)
    frame_names = sorted(poses_gt_data.keys())

    gt_positions = [np.array(poses_gt_data[name])[:3, 3] for name in frame_names]
    gt_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(gt_positions),
        lines=o3d.utility.Vector2iVector([[i, i+1] for i in range(len(gt_positions)-1)])
    )
    gt_line.paint_uniform_color([0.0, 0.8, 0.0])

    print("Finding and loading all result files...")
    # --- CHANGE: Look in the CURRENT directory for intermediate files ---
    search_pattern = "intermediate_results_iter_*.json"
    iteration_files = sorted(glob.glob(search_pattern), key=lambda x: int(Path(x).stem.split('_')[-1]))
    
    # Add the final result file to the list
    final_result_file = Path(args.final_result_file)
    if final_result_file.exists():
        iteration_files.append(str(final_result_file))
    
    if not iteration_files:
        print(f"Error: No result files found matching '{search_pattern}' in the current directory.")
        return

    # Load all steps into memory
    all_steps_data = []
    for i, f in enumerate(iteration_files):
        print(f"  Loading {f}...")
        positions, map_points = load_trajectory_and_map(f, frame_names)
        if positions and map_points is not None:
            is_final = (i == len(iteration_files) - 1)
            label = f"Final Result (Iter {Path(f).stem.split('_')[-1]})" if is_final and "intermediate" in f else "Final Result"
            if "intermediate" in f:
                label = f"Iteration {Path(f).stem.split('_')[-1]}"

            all_steps_data.append({'positions': positions, 'map': map_points, 'label': label})
    
    print(f"Loaded {len(all_steps_data)} total steps to visualize.")

    # --- Setup Animation ---
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Optimization Progress", width=1280, height=720)

    active_line = o3d.geometry.LineSet()
    active_map = o3d.geometry.PointCloud()
    
    vis.add_geometry(gt_line)
    vis.add_geometry(active_line)
    vis.add_geometry(active_map)

    state = {"index": 0, "is_paused": False, "last_update": time.time()}

    def toggle_pause(vis):
        state["is_paused"] = not state["is_paused"]
        print(f"\nAnimation {'PAUSED' if state['is_paused'] else 'PLAYING'}")
        
    vis.register_key_callback(ord(" "), toggle_pause)

    print("---------------------------------------------------------")
    print("              OPTIMIZATION VISUALIZATION                 ")
    print("-> Press [SPACE] to pause/play the animation.")
    print("-> Press [Q] or close the window to exit.")
    print("---------------------------------------------------------")

    keep_running = True
    while keep_running:
        current_time = time.time()
        if not state["is_paused"] and current_time - state["last_update"] > 0.2: # Slower update
            current_data = all_steps_data[state["index"]]
            print(f"Displaying: {current_data['label']}", end='\r')

            active_line.points = o3d.utility.Vector3dVector(current_data['positions'])
            active_line.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(current_data['positions'])-1)])
            active_line.paint_uniform_color([0.0, 0.6, 1.0])
            
            active_map.points = o3d.utility.Vector3dVector(current_data['map'])
            active_map.paint_uniform_color([0.7, 0.7, 0.7])
            active_map.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

            vis.update_geometry(active_line)
            vis.update_geometry(active_map)
            
            state["last_update"] = current_time
            state["index"] = (state["index"] + 1) % len(all_steps_data)

        if not vis.poll_events():
            keep_running = False
        vis.update_renderer()

    print("\nVisualization finished.")
    vis.destroy_window()

if __name__ == "__main__":
    main()