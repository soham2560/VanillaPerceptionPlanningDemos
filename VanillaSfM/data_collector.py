import argparse
import json
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

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
    z_axis = eye - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
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
        distance_variation = radial_deviation * (np.cos(4 * base_angle) + 0.3 * np.random.uniform(-1, 1))
        current_radius = base_radius + distance_variation
        x = current_radius * np.cos(angle)
        z = current_radius * np.sin(angle)
        height_variation = height_deviation * (np.sin(2 * base_angle) + 0.3 * np.random.uniform(-1, 1))
        y = height_variation
        
        camera_pos = center + np.array([x, y, z])
        
        R = create_look_at_matrix(camera_pos, center)
        t = -R @ camera_pos
        
        poses.append({'R': R.tolist(), 't': t.tolist()})
    return poses

def project_and_save_views(output_dir, points_3d, poses, K, img_size, add_noise, noise_level):
    width, height = img_size
    all_projections = []

    for i, pose in enumerate(poses):
        R = np.array(pose['R'])
        t = np.array(pose['t'])
        rvec, _ = cv2.Rodrigues(R)
        
        projections, _ = cv2.projectPoints(points_3d, rvec, t, K, None)
        projections = projections.reshape(-1, 2)
        
        if add_noise:
            noise = np.random.normal(0, noise_level, projections.shape)
            projections += noise
            
        valid_indices = (projections[:, 0] >= 0) & (projections[:, 0] < width) & \
                        (projections[:, 1] >= 0) & (projections[:, 1] < height)
        
        if(valid_indices.sum() != projections.shape[0]):
            print(f"Warning: Some projections are outside the image bounds for view {i}.")
        
        valid_projections = projections[valid_indices]
        all_projections.append(valid_projections)
        
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(valid_projections[:, 0], valid_projections[:, 1], s=1, c='blue')
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        ax.invert_yaxis()
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis('off')
        plt.savefig(output_dir / f"{i}.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
    return all_projections

def main():
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # STEP 1 -> Define Camera Parameters
    img_size = (640, 480)
    width, height = img_size
    focal_length = 500
    
    K = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ])

    # STEP 2 -> Load Point Cloud and Generate Random Camera Poses 
    points_3d = load_point_cloud(args.point_cloud)
    center, radius = get_point_cloud_bounds(points_3d)
    poses = generate_camera_poses(args.num_images, center, radius)
    
    # STEP 3 -> Project Points on each image and save images
    projections = project_and_save_views(
        output_dir, points_3d, poses, K, img_size, args.add_noise, args.noise_level
    )

    # STEP 4 -> Collect Correspondences
    correspondences = {}
    for i in range(args.num_images):
        key = f"{i}"
        correspondences[key] = projections[i].tolist()
    
    # STEP 5 -> Save Configurations and Correspondences
    config_data = {
        'num_images': args.num_images,
        'num_3d_points': len(points_3d),
        'image_size': img_size,
        'K': K.tolist(),
        'poses_gt': poses
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=4)   
    with open(output_dir / "correspondences.json", "w") as f:
        json.dump(correspondences, f)
    print(f"Data generation complete. Output saved to '{output_dir}'.")
    print(f"Generated {args.num_images} images")

if __name__ == "__main__":
    main()