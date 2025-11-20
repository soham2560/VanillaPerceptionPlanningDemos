import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if "poses_gt" in data:
        poses_key = "poses_gt"
        points_key = "points_3d"
    else:
        poses_key = "poses_estimated"
        points_key = "points_3d_estimated"
    
    poses = data[poses_key]
    if isinstance(poses, list):
        # This is the corrected line. It now creates string keys.
        poses = {str(i): p for i, p in enumerate(poses)}

    points = np.array(data.get(points_key, []))
    
    cam_centers = []
    for i in sorted([int(k) for k in poses.keys()]):
        pose = poses[str(i)]
        R = np.array(pose['R'])
        t = np.array(pose['t']).reshape(3,1)
        C = -R.T @ t
        cam_centers.append(C.flatten())
        
    return np.array(cam_centers), points

def align_data(data_to_align_c, data_to_align_p, ref_c):
    if data_to_align_c.shape[0] == 0:
        return data_to_align_c, data_to_align_p
    
    mtx1, mtx2, disparity = procrustes(ref_c, data_to_align_c)
    
    scale = np.linalg.norm(mtx2) / np.linalg.norm(mtx1)
    t = mtx1.mean(axis=0) - scale * mtx2.mean(axis=0)
    
    R = mtx2.T @ mtx1 / (scale * np.linalg.norm(mtx1)**2)
    
    aligned_c = scale * data_to_align_c @ R + t
    aligned_p = scale * data_to_align_p @ R + t if data_to_align_p.shape[0] > 0 else data_to_align_p

    # A simpler alignment calculation can also be used
    # mtx1, mtx2, disparity = procrustes(ref_c, data_to_align_c)
    # aligned_c = mtx2
    # if data_to_align_p.shape[0] > 0:
    #      # This is a less formal way but works for visualization
    #      c_mean_orig = data_to_align_c.mean(axis=0)
    #      p_mean_orig = data_to_align_p.mean(axis=0)
    #      c_mean_aligned = aligned_c.mean(axis=0)
    #      aligned_p = data_to_align_p - c_mean_orig + c_mean_aligned
    # else:
    #      aligned_p = data_to_align_p

    return aligned_c, aligned_p

def plot_scene(ax, cams, points, color, label):
    if cams.shape[0] > 0:
        ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], c=color, marker='^', s=100, label=f'{label} Poses')
    if points.shape[0] > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker='.', s=1, alpha=0.5, label=f'{label} Points')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing all results files.")
    args = parser.parse_args()
    
    gt_cams, _ = load_data(f"{args.data_dir}/config.json")
    init_cams, init_points = load_data(f"{args.data_dir}/results.json")
    opt_cams, opt_points = load_data(f"{args.data_dir}/optimized_results.json")

    init_cams_aligned, init_points_aligned = align_data(init_cams, init_points, gt_cams)
    opt_cams_aligned, opt_points_aligned = align_data(opt_cams, opt_points, gt_cams)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    plot_scene(ax, gt_cams, np.array([]), 'green', 'Ground Truth')
    plot_scene(ax, init_cams_aligned, init_points_aligned, 'blue', 'Initial')
    plot_scene(ax, opt_cams_aligned, opt_points_aligned, 'red', 'Optimized')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("Bundle Adjustment Results")
    plt.show()

if __name__ == '__main__':
    main()