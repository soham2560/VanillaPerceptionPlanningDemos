# VanillaSLAM

A minimal, from-scratch SLAM pipeline for educational purposes. This project demonstrates data simulation, odometry drift, and global optimization with the C++ Ceres Solver using Sophus for Lie algebra.

## Pipeline Overview

1.  **`data_collector.py`**: Generates a `slam_data` directory with a ground truth 3D map and a series of "local scans" from a simulated sensor trajectory.
2.  **`initial_estimator.py`**: Simulates a real robot by using ICP on noisy scans to create a **drifty trajectory** and a **distorted map**. It packages this "problem" into `slam_data/initial_estimates.json`.
3.  **`optimizer/optimizer.cc`**: The C++ Ceres Solver reads the problem file, performs a global optimization on the 7-DoF poses to correct the drift, and saves the final result and all intermediate steps.
4.  **`iterations_visualizer.py`**: Animates the intermediate steps to show the full optimization process.

## Installation

### 1. Python Environment (from `environment.yaml`)

This project uses Conda to manage Python dependencies.

```bash
conda env create -f environment.yaml
conda activate icpslam
```

### 2. C++ Dependencies (for the optimizer)

#### Ceres Solver (v2.2)
**[http://ceres-solver.org/installation.html](http://ceres-solver.org/installation.html)**

The "Building from source" guide is recommended.

---

## How to Run (Full Pipeline)

Execute all commands from the root `VanillaSLAM` directory.

### 1. Generate Data

```bash
conda activate icpslam
python data_collector.py --visualize
```

### 2. Create Initial Problem for Optimizer

```bash
python initial_estimator.py --visualize
```
*(Use the `--visualize` flag to see the "before" state. Press `S` to switch between the ground truth and the drifty initial guess.)*

### 3. Build and Run the C++ Optimizer

This is the corrected two-step CMake process, run from the project root.

```bash
cmake -S optimizer -B optimizer/build
cmake --build optimizer/build
./optimizer/build/optimizer slam_data/initial_estimates.json
```

### 4. Visualize the Results

This script animates the `intermediate_results_iter_*.json` files that the optimizer created in the `slam_data/solver_iterations` directory.

```bash
python iterations_visualizer.py
```
*   Press **SPACE** to pause/play the animation.
*   **Green Line**: Ground Truth Trajectory.
*   **Blue Line / Gray Cloud**: The state of the optimization, animating from the drifty guess to the final corrected result.