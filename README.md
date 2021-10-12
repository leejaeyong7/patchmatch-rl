# PatchMatch-RL: Deep MVS with Pixelwise Depth, Normal, and Visibility
#### Jae Yong Lee, Joseph DeGol, Chuhang Zou, Derek Hoiem

## Installation

To install necessary python package for our work:
```
conda install pytorch torchvision numpy matplotlib pandas tqdm tensorboard cudatoolkit=11.1 -c pytorch -c conda-forge
pip install opencv-python tabulate moviepy openpyxl pyntcloud open3d==0.9 pytorch-lightning==1.4.9
```

To setup dataset for training for our work, please download:
- [BlendedMVS Dataset](https://github.com/YoYo000/BlendedMVS)

To setup dataset for testing, please use:
- [ETH3D High-Res](https://github.com/FangjinhuaWang/PatchmatchNet) (PatchMatchNet pre-processed sets)
  - NOTE: We use our own script to pre-process. We are currently preparing code for the script. We will post update once it is available.
- [Tanks and Temples](https://github.com/YoYo000/MVSNet) (MVSNet pre-processed sets)

## Training
To train out method:
```
python bin/train.py --experiment_name=EXPERIMENT_NAME \
                    --log_path=TENSORBOARD_LOG_PATH \
                    --checkpoint_path=CHECKPOINT_PATH \
                    --dataset_path=ROOT_PATH_TO_DATA \
                    --dataset={BlendedMVS,DTU} \
                    --resume=True # if want to resume training with the same experiment_name
```

## Testing
To test our method, we need two scripts. First script to generate geometetry, and the second script to fuse the geometry. 
Geometry generation code:
```
python bin/generate.py --experiment_name=EXPERIMENT_USED_FOR_TRAINING \
                       --checkpoint_path=CHECKPOINT_PATH \
                       --epoch_id=EPOCH_ID \
                       --num_views=NUMBER_OF_VIEWS \
                       --dataset_path=ROOT_PATH_TO_DATA \
                       --output_path=PATH_TO_OUTPUT_GEOMETRY \
                       --width=(optional)WIDTH \
                       --height=(optional)HEIGHT \
                       --dataset={ETH3DHR, TanksAndTemples} \
                       --device=DEVICE
```
This will generate depths / normals / images into the folder specified by `--output_path`. To be more precise:
```
OUTPUT_PATH/
    EXPERIMENT_NAME/
        CHECKPOINT_FILE_NAME/
            SCENE_NAME/
                000000_camera.pth <-- contains intrinsics / extrinsics
                000000_depth_map.pth
                000000_normal_map.pth
                000000_meta.pth <-- contains src_image ids
                ...
```

Once the geometries are generated, we can use the fusion code to fuse them into point cloud:
GPU Fusion code:
```
python bin/fuse_output.py --output_path=OUTPUT_PATH_USED_IN_GENERATE.py
                          --experiment_name=EXPERIMENT_NAME \
                          --epoch_id=EPOCH_ID \
                          --dataset=DATASET \
                          # fusion related args
                          --proj_th=PROJECTION_DISTANCE_THRESHOLD \
                          --dist_th=DISTANCE_THRESHOLD \
                          --angle_th=ANGLE_THRESHOLD \
                          --num_consistent=NUM_CONSITENT_IMAGES \
                          --target_width=(Optional) target image width for fusion \
                          --target_height=(Optional) target image height for fusion \
                          --device=DEVICE \
```
The target width / height are useful for fusing depth / normal after upsampling. 

We also provide ETH3D testing script:
```
python bin/evaluate_eth3d.py --eth3d_binary_path=PATH_TO_BINARY_EXE \
                             --eth3d_gt_path=PATH_TO_GT_MLP_FOLDER \
                             --output_path=PATH_TO_FOLDER_WITH_POINTCLOUDS \
                             --experiment_name=NAME_OF_EXPERIMENT \
                             --epoch_id=EPOCH_OF_CHECKPOINT_TO_LOAD (default last.ckpt)
```

## Resources
- [Project Page](https://jyl.kr/patchmatch-rl)
- [Paper](https://arxiv.org/abs/2108.08943)

## Citation
If you want to use our work in your project, please cite:

```bibtex
@InProceedings{lee2021patchmatchrl,
    author    = {Lee, Jae Yong and DeGol, Joseph and Zou, Chuhang and Hoiem, Derek},
    title     = {PatchMatch-RL: Deep MVS with Pixelwise Depth, Normal, and Visibility},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
    month     = {October},
    year      = {2021}
}
```
