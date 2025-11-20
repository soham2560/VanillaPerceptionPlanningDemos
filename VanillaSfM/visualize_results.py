import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle config.json (GT) vs results.json (Est)
    if "poses_gt" in data:
        poses = data["poses_gt"]
        points = [] # GT points might be in ply, handled separately or ignored here
        # Convert list to dict for consistent processing if needed
        if isinstance(poses, list):
            poses = {str(i): p for i, p in enumerate(poses)}
        points_key = "points_3d" # Not always present in config
    else:
        poses = data["poses_estimated"]
        points_key = "points_3d_estimated"
        points = np.array(data.get(points_key, []))
    
    if isinstance(poses, list):
         poses = {str(i): p for i, p in enumerate(poses)}

    cam_centers = []
    # Sort by frame index
    indices = sorted([int(k) for k in poses.keys()])
    
    for i in indices:
        pose = poses[str(i)]
        R = np.array(pose['R'])
        t = np.array(pose['t']).reshape(3,1)
        # Camera Center C = -R^T * t
        C = -R.T @ t
        cam_centers.append(C.flatten())
        
    return np.array(cam_centers), points

def align_data(data_to_align_c, data_to_align_p, ref_c):
    if data_to_align_c.shape[0] == 0:
        return data_to_align_c, data_to_align_p
    
    # Procrustes analysis to align trajectories (resolves scale ambiguity)
    mtx1, mtx2, disparity = procrustes(ref_c, data_to_align_c)
    
    # Recover the transformation scale, rotation, translation
    # standardized_1 = (ref_c - mean1) / norm1
    # standardized_2 = (data - mean2) / norm2
    # mtx2 is the transformed version of standardized_2 to match standardized_1
    
    # We want to apply the transformation to the raw points (data_to_align_p)
    # procrustes() doesn't give R, t, s explicitly easy enough for points without re-deriving.
    # So we stick to the aligned camera centers returned by procrustes (mtx2)
    # but we need to un-standardize mtx2 to match the scale of ref_c.
    
    # Actually, scipy.spatial.procrustes aligns to a normalized space.
    # For visualization, it's often easier to compute the alignment explicitly:
    
    # 1. Center data
    mu_ref = np.mean(ref_c, axis=0)
    mu_dat = np.mean(data_to_align_c, axis=0)
    ref_centered = ref_c - mu_ref
    dat_centered = data_to_align_c - mu_dat
    
    # 2. Scale data
    s_ref = np.linalg.norm(ref_centered)
    s_dat = np.linalg.norm(dat_centered)
    scale = s_ref / s_dat
    dat_scaled = dat_centered * scale
    
    # 3. Rotate (Kabsch algorithm)
    # H = dat_scaled^T * ref_centered
    H = dat_scaled.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Apply transform to cameras
    aligned_c = (data_to_align_c - mu_dat) * scale @ R.T + mu_ref
    
    # Apply transform to points
    aligned_p = np.array([])
    if data_to_align_p.shape[0] > 0:
        aligned_p = (data_to_align_p - mu_dat) * scale @ R.T + mu_ref

    return aligned_c, aligned_p

# Visual Mapping: World(x,y,z) -> Plot(x, z, -y)
def to_vis(p):
    if p.shape[0] == 0: return p
    # Column stack: col 0, col 2, -col 1
    return np.column_stack([p[:, 0], p[:, 2], -p[:, 1]])

def plot_scene(ax, cams, points, color, label, marker_cam='^'):
    # Convert to visual frame before plotting
    if cams.shape[0] > 0:
        cams_vis = to_vis(cams)
        ax.scatter(cams_vis[:, 0], cams_vis[:, 1], cams_vis[:, 2], 
                  c=color, marker=marker_cam, s=50, label=f'{label} Poses', depthshade=False)
        # Draw trajectory line
        ax.plot(cams_vis[:, 0], cams_vis[:, 1], cams_vis[:, 2], c=color, alpha=0.5)

    if points.shape[0] > 0:
        # Subsample points for performance if too many
        if points.shape[0] > 5000:
            idx = np.random.choice(points.shape[0], 5000, replace=False)
            points = points[idx]
        
        points_vis = to_vis(points)
        ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], 
                  c=color, marker='.', s=1, alpha=0.3, label=f'{label} Points')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing all results files.")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    print("Loading Ground Truth...")
    gt_cams, _ = load_data(f"{data_dir}/config.json")
    
    print("Loading Initial Estimates...")
    init_cams, init_points = load_data(f"{data_dir}/results.json")
    
    print("Loading Optimized Results...")
    try:
        opt_cams, opt_points = load_data(f"{data_dir}/optimized_results.json")
        has_opt = True
    except FileNotFoundError:
        print("No optimized_results.json found. Skipping optimization plot.")
        has_opt = False
        opt_cams, opt_points = np.array([]), np.array([])

    print("Aligning trajectories...")
    init_cams_aligned, init_points_aligned = init_cams, init_points
    opt_cams_aligned, opt_points_aligned = opt_cams, opt_points

    # # Align Initial to GT
    init_cams_aligned, init_points_aligned = align_data(init_cams, init_points, gt_cams)
    
    # Align Optimized to GT
    if has_opt:
        opt_cams_aligned, opt_points_aligned = align_data(opt_cams, opt_points, gt_cams)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot GT
    plot_scene(ax, gt_cams, np.array([]), 'green', 'Ground Truth', 'o')
    
    # Plot Initial
    plot_scene(ax, init_cams_aligned, init_points_aligned, 'blue', 'Initial')
    
    # Plot Optimized
    if has_opt:
        plot_scene(ax, opt_cams_aligned, opt_points_aligned, 'red', 'Optimized')

    ax.set_xlabel('X')
    ax.set_ylabel('Z (Depth)')
    ax.set_zlabel('-Y (Up)')
    
    # Set nice view angle (similar to initial_estimator)
    ax.view_init(elev=20, azim=-45)
    
    # Equal aspect ratio hack for 3D
    # Create cubic bounding box to force equal aspect ratio
    all_cams = np.vstack([gt_cams, init_cams_aligned])
    if has_opt:
        all_cams = np.vstack([all_cams, opt_cams_aligned])
        
    vis_cams = to_vis(all_cams)
    
    max_range = np.array([vis_cams[:,0].max()-vis_cams[:,0].min(), 
                          vis_cams[:,1].max()-vis_cams[:,1].min(), 
                          vis_cams[:,2].max()-vis_cams[:,2].min()]).max() / 2.0

    mid_x = (vis_cams[:,0].max()+vis_cams[:,0].min()) * 0.5
    mid_y = (vis_cams[:,1].max()+vis_cams[:,1].min()) * 0.5
    mid_z = (vis_cams[:,2].max()+vis_cams[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    ax.set_title(f"Bundle Adjustment Results\nAligned to Ground Truth")
    plt.show()

if __name__ == '__main__':
    main()