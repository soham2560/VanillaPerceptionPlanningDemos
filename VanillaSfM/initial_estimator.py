import argparse
import json
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import open3d as o3d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_file", type=str, default="results.json")
    parser.add_argument("--gt_ply", type=str, default=None)
    return parser.parse_args()

def load_data(data_dir, gt_ply_path):
    with open(data_dir / "config.json") as f: config = json.load(f)
    with open(data_dir / "correspondences.json") as f: corresp = json.load(f)
    
    pts_gt = None
    if gt_ply_path:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        pts_gt = np.asarray(o3d.io.read_point_cloud(str(gt_ply_path)).points)
        
    return np.array(config['K']), config, corresp, pts_gt

def to_vis(p):
    return np.column_stack([p[:, 0], -p[:, 2], p[:, 1]]) if p.ndim > 1 else np.array([p[0], -p[2], p[1]])

def draw_cameras(ax, K, poses, scale, color):
    w, h = 640, 480
    corn = np.array([[0,0,1], [w,0,1], [w,h,1], [0,h,1]]).T
    frust_cam = (np.linalg.inv(K) @ corn).T * (scale * 0.05)
    
    for pose in poses.values():
        R, t = np.array(pose['R']), np.array(pose['t']).reshape(3, 1)
        C = (-R.T @ t).flatten()
        frust_w = (R.T @ frust_cam.T).T + C
        C_v, F_v = to_vis(C), to_vis(frust_w)
        
        for i in range(4):
            ax.plot([C_v[0], F_v[i,0]], [C_v[1], F_v[i,1]], [C_v[2], F_v[i,2]], color=color)
        idx = [0, 1, 2, 3, 0]
        ax.plot(F_v[idx, 0], F_v[idx, 1], F_v[idx, 2], color=color)

def render_scene(ax, points, poses, K, title, bounds=None, colors=None):
    ax.cla()
    centers = np.array([(-np.array(p['R']).T @ np.array(p['t']).reshape(3,1)).flatten() for p in poses.values()]) if poses else np.empty((0,3))
    
    if bounds is None:
        all_pts = np.vstack([points, centers]) if points is not None and len(points) > 0 else centers
        if len(all_pts) == 0: mid, rng = np.zeros(3), 1.0
        else:
            mid = (all_pts.max(0) + all_pts.min(0)) / 2
            rng = (all_pts.max(0) - all_pts.min(0)).max()
            if rng == 0: rng = 1.0
    else:
        mid, rng = bounds

    m_v = to_vis(mid)
    ax.set_xlim(m_v[0]-rng*0.7, m_v[0]+rng*0.7)
    ax.set_ylim(m_v[1]-rng*0.7, m_v[1]+rng*0.7)
    ax.set_zlim(m_v[2]-rng*0.7, m_v[2]+rng*0.7)
    ax.set(xlabel='X', ylabel='-Z', zlabel='Y', title=title, box_aspect=[1,1,1])
    ax.view_init(elev=20, azim=-45)

    if points is not None and len(points) > 0:
        pv = to_vis(points)
        c = colors if colors is not None else 'gray'
        ax.scatter(pv[:,0], pv[:,1], pv[:,2], c=c, marker='.', s=2, alpha=0.6)

    draw_cameras(ax, K, poses, rng, 'blue')
    return mid, rng

def compute_alignment(poses_gt, poses_est):
    """Computes Sim3 alignment (Scale, Rotation, Translation) between estimated and GT camera centers."""
    # Extract camera centers
    gt_centers = []
    est_centers = []
    common_indices = sorted(list(set(poses_gt.keys()) & set(poses_est.keys())))
    
    for i in common_indices:
        R_gt, t_gt = np.array(poses_gt[i]['R']), np.array(poses_gt[i]['t']).reshape(3,1)
        R_est, t_est = np.array(poses_est[i]['R']), np.array(poses_est[i]['t']).reshape(3,1)
        gt_centers.append((-R_gt.T @ t_gt).flatten())
        est_centers.append((-R_est.T @ t_est).flatten())

    gt_centers = np.array(gt_centers).T
    est_centers = np.array(est_centers).T

    if est_centers.shape[1] < 3:
        return 1.0, np.eye(3), np.zeros((3,1))

    # Centroids
    mu_gt = np.mean(gt_centers, axis=1, keepdims=True)
    mu_est = np.mean(est_centers, axis=1, keepdims=True)

    # Center data
    gt_centered = gt_centers - mu_gt
    est_centered = est_centers - mu_est

    # Scale
    var_gt = np.sum(gt_centered**2) / gt_centered.shape[1]
    var_est = np.sum(est_centered**2) / est_centered.shape[1]
    scale = np.sqrt(var_gt / var_est)

    # Rotation (Kabsch Algorithm)
    H = est_centered @ gt_centered.T
    U, _, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T
    
    if np.linalg.det(R_align) < 0:
        Vt[2, :] *= -1
        R_align = Vt.T @ U.T

    # Translation
    t_align = mu_gt - scale * R_align @ mu_est

    return scale, R_align, t_align

def align_pose(pose, s, R_a, t_a):
    """Aligns a single pose [R|t] using the Sim3 params."""
    R, t = np.array(pose['R']), np.array(pose['t']).reshape(3,1)
    # C_new = s * R_a * C_old + t_a
    # R_new = R_old * R_a.T 
    # t_new = -R_new * C_new
    
    C_old = -R.T @ t
    C_new = s * R_a @ C_old + t_a
    R_new = R @ R_a.T
    t_new = -R_new @ C_new
    
    return {'R': R_new, 't': t_new}

def compute_metrics(poses_gt, poses_est, pts_gt, pts_est):
    # 1. Compute Alignment based on Camera Centers
    s, R_a, t_a = compute_alignment(poses_gt, poses_est)
    
    # 2. Align Estimated Poses
    aligned_poses = {}
    for k, v in poses_est.items():
        aligned_poses[k] = align_pose(v, s, R_a, t_a)
        
    # 3. Align Estimated Points
    if pts_est is not None and len(pts_est) > 0:
        pts_est_aligned = (s * R_a @ pts_est.T + t_a).T
    else:
        pts_est_aligned = pts_est

    # 4. Compute Errors
    r_err, t_err = [], []
    for i in aligned_poses:
        if i in poses_gt:
            R_gt = np.array(poses_gt[i]['R'])
            t_gt = np.array(poses_gt[i]['t']).flatten()
            R_est = aligned_poses[i]['R']
            t_est = aligned_poses[i]['t'].flatten()
            
            # Rotation error (geodesic)
            R_diff = R_gt @ R_est.T
            trace = np.trace(R_diff)
            trace = np.clip((trace - 1) / 2, -1.0, 1.0)
            ang = np.arccos(trace)
            r_err.append(np.degrees(ang))
            
            # Translation error
            t_err.append(np.linalg.norm(t_gt - t_est))
            
    p_err = 0
    if pts_gt is not None and pts_est_aligned is not None:
        # Assuming 1-to-1 correspondence by index (guaranteed by data_collector now)
        if len(pts_gt) == len(pts_est_aligned):
            p_err = np.mean(np.linalg.norm(pts_gt - pts_est_aligned, axis=1))
    
    return {
        'rotation_errors': r_err,
        'translation_errors': t_err,
        'point_reconstruction_error': p_err,
        'mean_rotation_error_deg': np.mean(r_err) if r_err else None,
        'mean_translation_error': np.mean(t_err) if t_err else None
    }

def main():
    args = parse_args()
    K, cfg, corresp, pts_gt = load_data(args.data_dir, args.gt_ply)
    # Ensure keys are integers for processing
    poses_gt = {int(k): v for k,v in enumerate(cfg['poses_gt'])}
    corresp = {int(k): v for k,v in corresp.items()}

    plt.ion()
    fig = plt.figure(figsize=(18, 8))
    ax_gt, ax_est = fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')

    print("Step 1: Init views 0-1...")
    # Initial pair
    p0 = np.array(corresp[0], dtype=np.float32)
    p1 = np.array(corresp[1], dtype=np.float32)
    
    E, mask = cv2.findEssentialMat(p0, p1, K, cv2.RANSAC, 0.999, 1.0)
    _, R, t, _ = cv2.recoverPose(E, p0, p1, K, mask=mask)
    
    poses_est = {0: {'R': np.eye(3), 't': np.zeros((3,1))}, 1: {'R': R, 't': t}}
    
    P0 = K @ np.eye(3, 4)
    P1 = K @ np.hstack((R, t))
    pts_4d = cv2.triangulatePoints(P0, P1, p0.T, p1.T)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    
    # Filter bad points (behind camera or far away) but KEEP INDICES ALIGNED
    # We won't delete points to keep index synchronization with data_collector
    # Just mark them as 0,0,0 or handle validity separately.
    # For this specific demo to work with bundle_adjuster.cc which expects strict array sizes,
    # we keep the array size but maybe zero out bad points or rely on the fact that 
    # synthetic data is clean enough.
    
    # Note: In a real pipeline we would use sparse bundle adjustment and point IDs.
    # Here, we trust triangulation for the demo.

    def update_viz(txt=None):
        render_scene(ax_gt, pts_gt, poses_gt, K, f"Ground Truth ({len(poses_gt)} poses)")
        render_scene(ax_est, pts_3d, poses_est, K, f"Estimated ({len(poses_est)} poses)", bounds=None, colors='black')
        if txt: ax_est.text2D(0.05, 0.95, txt, transform=ax_est.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        plt.pause(0.01)

    update_viz()

    print("Step 2: Processing views...")
    num_images = cfg['num_images']
    
    for i in range(2, num_images):
        print(f"  - Frame {i}")
        pi = np.array(corresp[i], dtype=np.float32)
        
        # Use PnP on existing points
        # We use all points; in real world we'd use inliers.
        # Assuming corresp[i] corresponds to pts_3d
        _, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pi, K, None)
        R_new, _ = cv2.Rodrigues(rvec)
        poses_est[i] = {'R': R_new, 't': tvec}
        
        # Triangulate new observations to refine points (averaging)
        # This is a simple running average for the demo
        P_new = K @ np.hstack((R_new, tvec))
        pts_new_hom = cv2.triangulatePoints(P0, P_new, p0.T, pi.T)
        pts_new = (pts_new_hom[:3] / pts_new_hom[3]).T
        
        # Update points running average
        valid_update = np.abs(pts_new_hom[3]) > 1e-5
        pts_3d[valid_update] = (pts_3d[valid_update] * (i-1) + pts_new[valid_update]) / i
        
        update_viz()

    # Compute Scale-Aligned Metrics
    res = compute_metrics(poses_gt, poses_est, pts_gt, pts_3d)
    
    out = {
        'poses_estimated': {str(k): {'R': v['R'].tolist(), 't': v['t'].flatten().tolist()} for k,v in poses_est.items()},
        'points_3d_estimated': pts_3d.tolist(), 
        'errors': res 
    }
    
    with open(args.data_dir / args.output_file, 'w') as f: json.dump(out, f, indent=4)
    print(f"Done. Saved to {args.output_file}")
    print(f"Aligned Errors -> Rot: {res['mean_rotation_error_deg']:.4f} deg, Trans: {res['mean_translation_error']:.4f}, Pts: {res['point_reconstruction_error']:.4f}")
    
    err_txt = f"Rot: {res['mean_rotation_error_deg']:.2f} deg\nTrans: {res['mean_translation_error']:.2f}\nScale-Aligned"
    update_viz(err_txt)
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()