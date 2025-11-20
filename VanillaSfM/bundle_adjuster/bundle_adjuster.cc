#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "glog/logging.h"
#include "json.hpp"

using json = nlohmann::json;

struct SlamProblem {
    ~SlamProblem() {
        delete[] parameters_;
    }

    double* mutable_cameras() { return parameters_; }
    double* mutable_points() { return parameters_ + 9 * num_cameras_; }
    const double* cameras() const { return parameters_; }
    const double* points() const { return parameters_ + 9 * num_cameras_; }

    int num_observations() const { return num_observations_; }
    int num_cameras() const { return num_cameras_; }
    int num_points() const { return num_points_; }
    
    double* mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_indices_[i] * 9;
    }
    double* mutable_point_for_observation(int i) {
        return mutable_points() + point_indices_[i] * 3;
    }

    const std::vector<double>& observations() const { return observations_; }

    bool LoadFiles(const std::string& results_path, const std::string& corresp_path, const std::string& config_path) {
        json results_data, corresp_data, config_data;
        std::ifstream results_file(results_path); results_file >> results_data;
        std::ifstream corresp_file(corresp_path); corresp_file >> corresp_data;
        std::ifstream config_file(config_path); config_file >> config_data;

        num_cameras_ = results_data["poses_estimated"].size();
        num_points_ = results_data["points_3d_estimated"].size();
        num_observations_ = num_cameras_ * num_points_;
        num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
        parameters_ = new double[num_parameters_];

        double fx = config_data["K"][0][0].get<double>();
        double cx = config_data["K"][0][2].get<double>();
        double cy = config_data["K"][1][2].get<double>();

        for (int i = 0; i < num_cameras_; ++i) {
            auto pose = results_data["poses_estimated"][std::to_string(i)];
            double R_mat[3][3];
            for(int r = 0; r < 3; ++r) for(int c = 0; c < 3; ++c) R_mat[r][c] = pose["R"][r][c].get<double>();
            
            double t[3];
            for(int j = 0; j < 3; ++j) t[j] = pose["t"][j].get<double>();
            
            // Snavely model conversion
            double R_snavely[3][3];
            for(int r=0; r<2; ++r) for(int c=0; c<3; ++c) R_snavely[r][c] = R_mat[r][c];
            for(int c=0; c<3; ++c) R_snavely[2][c] = -R_mat[2][c];
            double t_snavely[3] = {t[0], t[1], -t[2]};

            // Transpose for Ceres (Column-Major)
            double R_col_major[9];
            for(int c=0; c<3; ++c) for(int r=0; r<3; ++r) R_col_major[c*3 + r] = R_snavely[r][c];
            
            double* camera = mutable_cameras() + i * 9;
            ceres::RotationMatrixToAngleAxis(R_col_major, camera);
            camera[3] = t_snavely[0]; camera[4] = t_snavely[1]; camera[5] = t_snavely[2];
            camera[6] = fx; camera[7] = 0.0; camera[8] = 0.0;
        }

        for (int i = 0; i < num_points_; ++i) {
            for(int j = 0; j < 3; ++j) {
                mutable_points()[i * 3 + j] = results_data["points_3d_estimated"][i][j].get<double>();
            }
        }

        camera_indices_.resize(num_observations_);
        point_indices_.resize(num_observations_);
        observations_.resize(2 * num_observations_);
        for (int i = 0; i < num_cameras_; ++i) {
            auto corresp = corresp_data[std::to_string(i)];
            for (int j = 0; j < num_points_; ++j) {
                int obs_idx = i * num_points_ + j;
                camera_indices_[obs_idx] = i;
                point_indices_[obs_idx] = j;
                observations_[2 * obs_idx + 0] = corresp[j][0].get<double>() - cx;
                observations_[2 * obs_idx + 1] = corresp[j][1].get<double>() - cy;
            }
        }
        return true;
    }

    void WriteToFile(const std::string& filename) const {
        json output_data;
        for (int i = 0; i < num_cameras_; ++i) {
            const double* camera = cameras() + i * 9;
            double R_col_major[9];
            ceres::AngleAxisToRotationMatrix(camera, R_col_major);

            // Convert from Column-Major back to Row-Major for our logic
            double R_snavely[3][3];
            for(int c=0; c<3; ++c) for(int r=0; r<3; ++r) R_snavely[r][c] = R_col_major[c*3 + r];

            // Convert from Snavely back to OpenCV model
            double R_mat[3][3];
            for(int r=0; r<2; ++r) for(int c=0; c<3; ++c) R_mat[r][c] = R_snavely[r][c];
            for(int c=0; c<3; ++c) R_mat[2][c] = -R_snavely[2][c];
            double t[3] = {camera[3], camera[4], -camera[5]};

            json pose;
            for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) pose["R"][r][c] = R_mat[r][c];
            for(int c=0; c<3; ++c) pose["t"][c] = t[c];
            output_data["poses_estimated"][std::to_string(i)] = pose;
        }

        for(int i=0; i < num_points_; ++i) {
            for(int j=0; j<3; ++j) {
                 output_data["points_3d_estimated"][i][j] = points()[i * 3 + j];
            }
        }
        std::ofstream file(filename);
        file << std::setw(4) << output_data << std::endl;
    }

private:
    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    std::vector<int> point_indices_;
    std::vector<int> camera_indices_;
    std::vector<double> observations_;
    double* parameters_;
};

struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const {
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];
        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = xp * xp + yp * yp;
        T distortion = 1.0 + r2 * (l1 + l2 * r2);
        const T& focal = camera[6];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;
        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
        return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
            new SnavelyReprojectionError(observed_x, observed_y));
    }
    double observed_x;
    double observed_y;
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    if (argc != 2) {
        std::cerr << "usage: bundle_adjuster <data_directory>\n";
        return 1;
    }
    std::string data_dir = argv[1];

    SlamProblem problem_loader;
    if (!problem_loader.LoadFiles(data_dir + "/results.json", data_dir + "/correspondences.json", data_dir + "/config.json")) {
        std::cerr << "ERROR: unable to load files from " << data_dir << "\n";
        return 1;
    }

    const auto& observations = problem_loader.observations();
    ceres::Problem problem;
    for (int i = 0; i < problem_loader.num_observations(); ++i) {
        ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(
            observations[2 * i + 0],
            observations[2 * i + 1]);
        problem.AddResidualBlock(cost_function,
                                 new ceres::HuberLoss(1.0),
                                 problem_loader.mutable_camera_for_observation(i),
                                 problem_loader.mutable_point_for_observation(i));
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 200;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    problem_loader.WriteToFile(data_dir + "/optimized_results.json");

    return 0;
}