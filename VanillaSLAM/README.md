# VanillaSLAM
Demos for VanillaSLAM for educational purposes

```bash
conda activate icpslam
```
```bash
cd VanillaSLAM
```
```bash
python data_collector.py --visualize
```
```bash
python initial_estimator.py --visualize
```
```bash
cmake --build optimizer/build
```
```bash
./optimizer/build/optimizer ./slam_data/initial_estimates.json
```
```bash
python iterations_visualizer.py
```