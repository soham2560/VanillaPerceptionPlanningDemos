import argparse
import json
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate synthetic SLAM data from a point cloud.")
    parser.add_argument("--point_cloud", type=str, required=True, help="Path to the .ply point cloud file.")
    parser.add_argument("--num_images", type=int, default=10, help="Number of random camera views to generate.")
    parser.add_argument("--output_dir", type=str, default="slam_data", help="Directory to save the generated data.")
    parser.add_argument("--add_noise", action="store_true", help="Add Gaussian noise to 2D correspondences.")
    parser.add_argument("--noise_level", type=float, default=0.5, help="Standard deviation of Gaussian noise in pixels.")
    return parser.parse_args()

def load_point_cloud(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    return np.asarray(pcd.points)

def get_point_cloud_bounds(points):
    center = points.mean(axis=0)
    max_dist = np.max(np.linalg.norm(points - center, axis=1))
    return center, max_dist

def create_look_at_matrix(eye, target, up=np.array([0, -1, 0])):
    z_axis = target - eye
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, up)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.vstack((x_axis, y_axis, z_axis))
    return R

def generate_camera_poses(num_poses, center, radius):
    poses = []
    distance_scale = 2.5
    
    base_radius = radius * distance_scale
    radial_deviation = base_radius * 0.15 * (10 / num_poses)
    height_deviation = base_radius * 0.2 * (10 / num_poses)
    angular_deviation = 0.1 * (10 / num_poses)
    
    for i in range(num_poses):
        base_angle = (i / num_poses) * 2 * np.pi
        angle_offset = angular_deviation * np.sin(3 * base_angle + np.random.uniform(-0.5, 0.5))
        angle = base_angle + angle_offset
        
        dist_var = radial_deviation * (np.cos(4 * base_angle) + 0.3 * np.random.uniform(-1, 1))
        curr_radius = base_radius + dist_var
        
        x = curr_radius * np.cos(angle)
        z = curr_radius * np.sin(angle)
        y = height_deviation * (np.sin(2 * base_angle) + 0.3 * np.random.uniform(-1, 1))
        
        camera_pos = center + np.array([x, y, z])
        R = create_look_at_matrix(camera_pos, center)
        t = -R @ camera_pos
        
        poses.append({'R': R.tolist(), 't': t.tolist()})
    return poses

def generate_views_and_correspondences(output_dir, points_3d, poses, K, img_size, add_noise, noise_level):
    """
    Projects 3D points.
    CRITICAL CHANGE: We now export ALL points, even if they are outside the image bounds.
    This ensures that the i-th point in the 2D list corresponds to the i-th point in the 3D list,
    which is required by the simple bundle adjuster.
    """
    width, height = img_size
    correspondences = {}

    for i, pose in enumerate(poses):
        R = np.array(pose['R'])
        t = np.array(pose['t'])
        rvec, _ = cv2.Rodrigues(R)
        
        # Project points
        projections, _ = cv2.projectPoints(points_3d, rvec, t, K, None)
        projections = projections.reshape(-1, 2)
        
        # Add noise if requested
        if add_noise:
            noise = np.random.normal(0, noise_level, projections.shape)
            projections += noise
            
        # Store ALL correspondences directly to maintain index alignment
        correspondences[str(i)] = projections.tolist()
        
        # Create visualization image (only draw visible points)
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Filter for visualization only
        for pt in projections:
            if 0 <= pt[0] < width and 0 <= pt[1] < height:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 1, (255, 0, 0), -1) # Blue in BGR
            
        cv2.imwrite(str(output_dir / f"{i}.png"), img)
        
    return correspondences

def main():
    args = parse_arguments()
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        print(f"Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # STEP 1 -> Define Camera Parameters
    width, height = 640, 480
    focal_length = 500
    
    K = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ])

    # STEP 2 -> Load Point Cloud and Generate Camera Poses 
    points_3d = load_point_cloud(args.point_cloud)
    center, radius = get_point_cloud_bounds(points_3d)
    poses = generate_camera_poses(args.num_images, center, radius)
    
    # STEP 3 & 4 -> Project Points, Save Images, and Build Correspondences
    correspondences = generate_views_and_correspondences(
        output_dir, points_3d, poses, K, (width, height), args.add_noise, args.noise_level
    )
    
    # STEP 5 -> Save Configurations
    config_data = {
        'num_images': args.num_images,
        'num_3d_points': len(points_3d),
        'image_size': (width, height),
        'K': K.tolist(),
        'poses_gt': poses
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=4)   
    with open(output_dir / "correspondences.json", "w") as f:
        json.dump(correspondences, f)
        
    print(f"Data generation complete. Output saved to '{output_dir}'.")
    print(f"Generated {args.num_images} images with {len(points_3d)} points tracked per image.")

if __name__ == "__main__":
    main()