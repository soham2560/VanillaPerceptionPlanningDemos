// optimiser.cc
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <memory>

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include "ceres/ceres.h"
#include "json.hpp"

using json = nlohmann::json;
using namespace Eigen;

// Convert 4x4 pose matrix (flattened row-major 16-vector) -> SE3 tangent (6-vector)
void PoseMatrixToSE3Tangent(const std::vector<double>& pose_matrix, double* se3_tangent) {
    Matrix4d T;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            T(i, j) = pose_matrix[i * 4 + j];

    Sophus::SE3d se3_pose(T);
    Sophus::Vector6d tangent = se3_pose.log();
    for (int i = 0; i < 6; ++i) se3_tangent[i] = tangent[i];
}

// Convert SE3 tangent (6-vector) -> 4x4 pose matrix (flattened row-major 16-vector)
void SE3TangentToPoseMatrix(const double* se3_tangent, std::vector<double>& pose_matrix) {
    pose_matrix.resize(16);
    Sophus::Vector6d tangent;
    for (int i = 0; i < 6; ++i) tangent[i] = se3_tangent[i];
    Sophus::SE3d se3_pose = Sophus::SE3d::exp(tangent);
    Matrix4d T = se3_pose.matrix();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            pose_matrix[i * 4 + j] = T(i, j);
}

struct LidarSlamProblem {
    int num_poses_ = 0;
    int num_points_ = 0;
    int num_observations_ = 0;

    std::vector<std::string> pose_names_;
    std::vector<int> pose_indices_;
    std::vector<int> point_indices_;
    std::vector<std::vector<double>> observations_;

    // single contiguous parameter buffer: [poses (num_poses_*6), points (num_points_*3)]
    double* parameters_ = nullptr;

    LidarSlamProblem() = default;
    ~LidarSlamProblem() {
        if (parameters_) delete[] parameters_;
    }

    double* mutable_poses() { return parameters_; }
    double* mutable_points() { return parameters_ + num_poses_ * 6; }

    double* mutable_pose_for_observation(int i) { return mutable_poses() + pose_indices_[i] * 6; }
    double* mutable_point_for_observation(int i) { return mutable_points() + point_indices_[i] * 3; }

    const double* poses() const { return parameters_; }
    const double* points() const { return parameters_ + num_poses_ * 6; }

    bool LoadFile(const std::string& filepath) {
        json problem_json;
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open input JSON: " << filepath << "\n";
            return false;
        }
        file >> problem_json;

        // Expect "poses" to be an array of objects with fields "name" and "pose"
        num_poses_ = problem_json.at("poses").size();
        num_points_ = problem_json.at("points").size();
        num_observations_ = problem_json.at("observations").size();

        // allocate parameter buffer
        parameters_ = new double[num_poses_ * 6 + num_points_ * 3];

        // read poses and names
        pose_names_.resize(num_poses_);
        for (int i = 0; i < num_poses_; ++i) {
            pose_names_[i] = problem_json.at("poses").at(i).at("name").get<std::string>();
            std::vector<double> pose_matrix =
                problem_json.at("poses").at(i).at("pose").get<std::vector<double>>();
            PoseMatrixToSE3Tangent(pose_matrix, mutable_poses() + i * 6);
        }

        // read points
        for (int i = 0; i < num_points_; ++i) {
            std::vector<double> point_vec =
                problem_json.at("points").at(i).get<std::vector<double>>();
            mutable_points()[i * 3 + 0] = point_vec[0];
            mutable_points()[i * 3 + 1] = point_vec[1];
            mutable_points()[i * 3 + 2] = point_vec[2];
        }

        // observations (expected: pose_id, point_id, observed_xyz (3-vector))
        pose_indices_.reserve(num_observations_);
        point_indices_.reserve(num_observations_);
        observations_.reserve(num_observations_);
        for (int i = 0; i < num_observations_; ++i) {
            pose_indices_.push_back(problem_json.at("observations").at(i).at("pose_id").get<int>());
            point_indices_.push_back(problem_json.at("observations").at(i).at("point_id").get<int>());
            observations_.push_back(problem_json.at("observations").at(i).at("observed_xyz").get<std::vector<double>>());
        }

        return true;
    }

    void WriteToFile(const std::string& filename) const {
        json output_data;
        // write poses with original names as keys so visualizer can find them by name
        for (int i = 0; i < num_poses_; ++i) {
            std::vector<double> pose_matrix;
            SE3TangentToPoseMatrix(poses() + i * 6, pose_matrix);
            output_data["poses_optimized"][pose_names_[i]] = pose_matrix;
        }

        std::vector<std::vector<double>> points_vec;
        points_vec.reserve(num_points_);
        for (int i = 0; i < num_points_; ++i) {
            points_vec.push_back({
                points()[i * 3 + 0],
                points()[i * 3 + 1],
                points()[i * 3 + 2]
            });
        }
        output_data["map_points_optimized"] = points_vec;

        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file: " << filename << "\n";
            return;
        }
        file << output_data.dump(4);
    }
};

// Cost functor: observed point is in sensor frame; parameters are
// se3_tangent (6) encoding T_world_sensor (t, phi) and world_point (3)
struct LidarPointError {
    LidarPointError(const std::array<double,3>& observed_xyz)
        : observed_{observed_xyz[0], observed_xyz[1], observed_xyz[2]} {}

    template <typename T>
    bool operator()(const T* const se3_tangent, const T* const world_point, T* residuals) const {
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using SE3T  = Sophus::SE3<T>;

        // map inputs
        Eigen::Map<const Eigen::Matrix<T,6,1>> xi(se3_tangent);
        Eigen::Map<const Vec3T> pw(world_point);

        // reconstruct SE3 (T_world_sensor)
        SE3T T_ws = SE3T::exp(xi);

        // convert world point to sensor frame: p_s = T_sw * p_w = T_ws.inverse() * p_w
        Vec3T ps = T_ws.inverse() * pw;

        // residual = ps - observed (observed is in sensor coordinates)
        residuals[0] = ps[0] - T(observed_[0]);
        residuals[1] = ps[1] - T(observed_[1]);
        residuals[2] = ps[2] - T(observed_[2]);

        return true;
    }

    static ceres::CostFunction* Create(const std::array<double,3>& observed_xyz) {
        return new ceres::AutoDiffCostFunction<LidarPointError, 3, 6, 3>(
            new LidarPointError(observed_xyz)
        );
    }

    std::array<double,3> observed_;
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_lidar_problem.json>\n";
        return 1;
    }

    LidarSlamProblem problem_data;
    if (!problem_data.LoadFile(argv[1])) {
        std::cerr << "Failed to load problem file.\n";
        return 1;
    }

    ceres::Problem problem;

    // Loss function
    ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);

    // Create residuals
    for (int i = 0; i < problem_data.num_observations_; ++i) {
        // observed_xyz: vector<double> size 3
        const auto& obs = problem_data.observations_[i];
        std::array<double,3> observed_xyz = { obs[0], obs[1], obs[2] };

        ceres::CostFunction* cost_function = LidarPointError::Create(observed_xyz);

        double* pose_param = problem_data.mutable_pose_for_observation(i);   // pointer to 6 doubles
        double* point_param = problem_data.mutable_point_for_observation(i); // pointer to 3 doubles

        problem.AddResidualBlock(cost_function, loss_function, pose_param, point_param);
    }

    if (problem_data.num_poses_ > 0) {
        problem.SetParameterBlockConstant(problem_data.mutable_poses() + 0 * 6);
    }

    // Solver options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 500;
    options.num_threads = std::max(1u, std::thread::hardware_concurrency());
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    // write optimized result
    problem_data.WriteToFile("lidar_slam_optimized.json");

    return 0;
}
