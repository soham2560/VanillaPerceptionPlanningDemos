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
#include <filesystem>

using json = nlohmann::json;
using namespace Eigen;
namespace fs = std::filesystem;

const std::string kOutputFolder = "slam_data";

struct LidarSlamProblem {
    int num_poses_ = 0;
    int num_points_ = 0;
    int num_observations_ = 0;

    std::vector<std::string> pose_names_;
    std::vector<int> pose_indices_;
    std::vector<int> point_indices_;
    std::vector<std::vector<double>> observations_;

    // single contiguous parameter buffer: [poses (num_poses_*7), points (num_points_*3)]
    double* parameters_ = nullptr;

    LidarSlamProblem() = default;
    ~LidarSlamProblem() {
        if (parameters_) delete[] parameters_;
    }

    double* mutable_poses() { return parameters_; }
    double* mutable_points() { return parameters_ + num_poses_ * 7; }

    double* mutable_pose_for_observation(int i) { return mutable_poses() + pose_indices_[i] * 7; }
    double* mutable_point_for_observation(int i) { return mutable_points() + point_indices_[i] * 3; }

    const double* poses() const { return parameters_; }
    const double* points() const { return parameters_ + num_poses_ * 7; }

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
        parameters_ = new double[num_poses_ * 7 + num_points_ * 3];

        // read poses and names
        pose_names_.resize(num_poses_);
        for (int i = 0; i < num_poses_; ++i) {
            pose_names_[i] = problem_json.at("poses").at(i).at("name").get<std::string>();
            std::vector<double> pose_matrix =
                problem_json.at("poses").at(i).at("pose").get<std::vector<double>>();

            Eigen::Matrix4d T;
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    T(r, c) = pose_matrix[r * 4 + c];

            Sophus::SE3d se3_pose(T);
            Eigen::Quaterniond q(se3_pose.rotationMatrix());
            Eigen::Vector3d t = se3_pose.translation();

            double* pose = mutable_poses() + i * 7;
            pose[0] = q.w();
            pose[1] = q.x();
            pose[2] = q.y();
            pose[3] = q.z();
            pose[4] = t.x();
            pose[5] = t.y();
            pose[6] = t.z();
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
            const double* pose = poses() + i * 7;
            Eigen::Quaterniond q(pose[0], pose[1], pose[2], pose[3]);
            Eigen::Vector3d t(pose[4], pose[5], pose[6]);
            Sophus::SE3d se3_pose(q, t);

            std::vector<double> pose_matrix(16);
            Eigen::Matrix4d T = se3_pose.matrix();
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    pose_matrix[r * 4 + c] = T(r, c);

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
// quaternion+translation (7) encoding T_world_sensor and world_point (3)
struct LidarPointError {
    LidarPointError(const std::array<double,3>& observed_xyz)
        : observed_{observed_xyz[0], observed_xyz[1], observed_xyz[2]} {}

    template <typename T>
    bool operator()(const T* const pose, const T* const world_point, T* residuals) const {
        const Eigen::Quaternion<T> q(pose[0], pose[1], pose[2], pose[3]);
        Eigen::Map<const Eigen::Matrix<T,3,1>> t(pose + 4);
        Sophus::SE3<T> T_ws(q, t);
        Eigen::Map<const Eigen::Matrix<T,3,1>> pw(world_point);
        Eigen::Matrix<T,3,1> ps_observed;
        ps_observed << T(observed_[0]), T(observed_[1]), T(observed_[2]);
        Eigen::Matrix<T,3,1> pw_predicted = T_ws * ps_observed;
        residuals[0] = pw_predicted[0] - pw[0];
        residuals[1] = pw_predicted[1] - pw[1];
        residuals[2] = pw_predicted[2] - pw[2];
        return true;
    }


    static ceres::CostFunction* Create(const std::array<double,3>& observed_xyz) {
        return new ceres::AutoDiffCostFunction<LidarPointError, 3, 7, 3>(
            new LidarPointError(observed_xyz)
        );
    }

    std::array<double,3> observed_;
};

class IterationDataSaverCallback : public ceres::IterationCallback {
public:
    explicit IterationDataSaverCallback(const LidarSlamProblem* problem_data)
        : problem_data_(problem_data) {}
    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
        std::string filename = kOutputFolder + "/solver_iterations/intermediate_results_iter_" + std::to_string(summary.iteration) + ".json";
        problem_data_->WriteToFile(filename);
        return ceres::SOLVER_CONTINUE;
    }

private:
    const LidarSlamProblem* problem_data_;
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
    
    const std::string iter_folder = kOutputFolder + "/solver_iterations";
    if (fs::exists(iter_folder)) {
        for (auto& entry : fs::directory_iterator(iter_folder)) {
            fs::remove_all(entry.path());
        }
    } else {
        fs::create_directory(iter_folder);
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

        double* pose_param = problem_data.mutable_pose_for_observation(i);
        double* point_param = problem_data.mutable_point_for_observation(i);

        problem.AddResidualBlock(cost_function, loss_function, pose_param, point_param);
    }
    
    // Set manifolds for quaternion+translation parameterization
    for (int i = 0; i < problem_data.num_poses_; ++i) {
        double* pose_param = problem_data.mutable_poses() + i * 7;
        ceres::Manifold* se3_manifold =
            new ceres::ProductManifold<ceres::QuaternionManifold, ceres::EuclideanManifold<3>>();
        problem.SetManifold(pose_param, se3_manifold);
    }

    if (problem_data.num_poses_ > 0) {
        problem.SetParameterBlockConstant(problem_data.mutable_poses() + 0 * 7);
    }

    // Solver options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 500;
    options.num_threads = std::max(1u, std::thread::hardware_concurrency());
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    IterationDataSaverCallback callback(&problem_data);
    options.callbacks.push_back(&callback);
    options.update_state_every_iteration = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    problem_data.WriteToFile(kOutputFolder + "/lidar_slam_optimized.json");

    return 0;
}