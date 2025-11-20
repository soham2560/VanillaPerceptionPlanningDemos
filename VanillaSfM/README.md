# VanillaMonocularSLAMDemo
A Demo of a Vanilla MonocularSLAM Pipeline for educational purposes

## Important Commands
- Data Collection   
    ```bash
    python data_collector.py --point_cloud sample_data/bunny/bun_zipper_res3.ply --num_images 15 --output_dir bunny_data_noisy --add_noise --noise_level 2.0
    ```
    ```bash
    python initial_estimator.py --data_dir bunny_data_noisy --output_file results_v2.json
    ```
    ```bash
    make -C bundle_adjuster/build
    ```