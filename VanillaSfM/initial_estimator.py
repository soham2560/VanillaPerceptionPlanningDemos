import argparse
import json
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_arguments():
    parser = argparse.ArgumentParser(description="Iterative pose and structure estimation for SLAM.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing generated SLAM data.")
    parser.add_argument("--output_file", type=str, default="results_v3.json", help="File to save estimation results.")
    return parser.parse_args()

def load_data(data_dir):
    data_dir = Path(data_dir)
    with open(data_dir / "config.json", 'r') as f:
        config = json.load(f)
    with open(data_dir / "correspondences.json", 'r') as f:
        correspondences = json.load(f)
    K = np.array(config['K'])
    return K, config, correspondences

def create_visualizer():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax

def draw_camera(ax, K, R, t, scene_scale, vis_scale=0.1, color='b'):
    h, w = 480, 640
    corners = np.array([[0, 0, 1],[w, 0, 1],[w, h, 1],[0, h, 1]])
    K_inv = np.linalg.inv(K)
    
    actual_scale = scene_scale * vis_scale
    
    frustum_corners_cam = (K_inv @ corners.T).T * actual_scale
    C = -R.T @ t.reshape(3,1)
    
    frustum_corners_world = (R.T @ frustum_corners_cam.T).T + C.T

    ax.plot([C[0,0], frustum_corners_world[0,0]], [C[1,0], frustum_corners_world[0,1]], [C[2,0], frustum_corners_world[0,2]], color=color)
    ax.plot([C[0,0], frustum_corners_world[1,0]], [C[1,0], frustum_corners_world[1,1]], [C[2,0], frustum_corners_world[1,2]], color=color)
    ax.plot([C[0,0], frustum_corners_world[2,0]], [C[1,0], frustum_corners_world[2,1]], [C[2,0], frustum_corners_world[2,2]], color=color)
    ax.plot([C[0,0], frustum_corners_world[3,0]], [C[1,0], frustum_corners_world[3,1]], [C[2,0], frustum_corners_world[3,2]], color=color)
    ax.plot(frustum_corners_world[[0,1,2,3,0], 0], frustum_corners_world[[0,1,2,3,0], 1], frustum_corners_world[[0,1,2,3,0], 2], color=color)

def update_plot(ax, points_3d, point_colors, poses, K):
    ax.cla()

    scene_points = []
    if points_3d is not None and len(points_3d) > 0:
        scene_points.append(points_3d)

    cam_centers = []
    for pose in poses.values():
        R = np.array(pose['R'])
        t = np.array(pose['t'])
        C = -R.T @ t.reshape(3, 1)
        cam_centers.append(C.T)
    
    if cam_centers:
        scene_points.append(np.vstack(cam_centers))

    if not scene_points:
        plt.pause(0.1)
        return

    all_points = np.vstack(scene_points)
    x, y, z = all_points[:, 0], all_points[:, 1], all_points[:, 2]

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    if max_range == 0: max_range = 1.0

    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range * 0.6, mid_x + max_range * 0.6)
    ax.set_ylim(mid_y - max_range * 0.6, mid_y + max_range * 0.6)
    ax.set_zlim(mid_z - max_range * 0.6, mid_z + max_range * 0.6)

    if points_3d is not None and len(points_3d) > 0:
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=point_colors, marker='.', s=2, alpha=0.6)

    for i, pose_data in poses.items():
        R = np.array(pose_data['R'])
        t = np.array(pose_data['t'])
        color = 'blue' if i == 0 else ('green' if i == 1 else 'red')
        draw_camera(ax, K, R, t, scene_scale=max_range, vis_scale=0.1, color=color)
    
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'SLAM Estimation - Poses: {len(poses)}, 3D Points: {len(points_3d) if points_3d is not None else 0}')
    plt.pause(0.5)

def recover_pose_from_essential(E, pts1, pts2, K):
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def triangulate_points(P1, P2, pts1, pts2):
    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d_hom[:3] / points_4d_hom[3]
    return points_3d.T

def build_projection_matrix(K, R, t):
    return K @ np.hstack((R, t))

def main():
    args = parse_arguments()
    data_dir = Path(args.data_dir)
    K, config, all_corresp = load_data(args.data_dir)
    num_images = config['num_images']

    fig, ax = create_visualizer()
    plt.ion()

    print("Step 1: Initializing from views 0 and 1...")
    pts0 = np.array(all_corresp["0"], dtype=np.float32)
    pts1 = np.array(all_corresp["1"], dtype=np.float32)

    E, mask_e = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R_rel, t_rel, mask_p = cv2.recoverPose(E, pts0, pts1, K, mask=mask_e)

    R0, t0 = np.eye(3), np.zeros((3, 1))
    estimated_poses = {0: {'R': R0, 't': t0}, 1: {'R': R_rel, 't': t_rel}}

    P0 = build_projection_matrix(K, R0, t0)
    P1 = build_projection_matrix(K, R_rel, t_rel)
    
    # Triangulate points and immediately filter for valid ones
    points_4d_hom = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)
    points_3d = points_4d_hom[:3] / points_4d_hom[3]
    points_3d = points_3d.T
    
    # Create a mask for valid, finite points
    valid_mask = np.all(np.isfinite(points_3d), axis=1)
    points_3d = points_3d[valid_mask]
    
    # Update all correspondence arrays to only use valid points
    for i in range(num_images):
        all_corresp[str(i)] = np.array(all_corresp[str(i)])[valid_mask]
    
    pts0 = np.array(all_corresp["0"], dtype=np.float32) # Reload with filtered points

    point_colors = np.array(['black'] * len(points_3d))
    update_plot(ax, points_3d, point_colors, estimated_poses, K)

    print(f"Initial triangulation successful. Found {len(points_3d)} valid 3D points.")
    print("Step 2: Iteratively processing views and refining the map...")
    for i in range(2, num_images):
        print(f"  - Processing frame {i}...")
        pts_i = np.array(all_corresp[f"{i}"], dtype=np.float32)

        if len(points_3d) < 8: continue
        
        _, rvec, tvec, _ = cv2.solvePnPRansac(points_3d, pts_i, K, None)
        R_new, _ = cv2.Rodrigues(rvec)
        estimated_poses[i] = {'R': R_new, 't': tvec}
        
        P_new = build_projection_matrix(K, R_new, tvec)
        re_triangulated_points_4d = cv2.triangulatePoints(P0, P_new, pts0.T, pts_i.T)
        re_triangulated_points = (re_triangulated_points_4d[:3] / re_triangulated_points_4d[3]).T
        
        # Only update with valid re-triangulated points
        valid_retri_mask = np.all(np.isfinite(re_triangulated_points), axis=1)
        
        num_prior = float(i - 1)
        points_3d[valid_retri_mask] = (points_3d[valid_retri_mask] * num_prior + re_triangulated_points[valid_retri_mask]) / (num_prior + 1)
        
        update_plot(ax, points_3d, point_colors, estimated_poses, K)

    results = {
        'poses_estimated': {k: {'R': v['R'].tolist(), 't': v['t'].flatten().tolist()} for k,v in estimated_poses.items()},
        'points_3d_estimated': points_3d.tolist()
    }

    with open(data_dir / args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nEstimation complete.")
    input("Press ENTER in this terminal to close the visualization and exit.")
    plt.close(fig)
    print("Visualization closed. Exiting.")

if __name__ == "__main__":
    main()