# PatchMatch-RL: Deep MVS with Pixelwise Depth, Normal, and Visibility
#### Jae Yong Lee, Joseph DeGol, Chuhang Zou, Derek Hoiem
![Architecture_up](https://user-images.githubusercontent.com/5545126/127352929-498537bd-1139-4241-bd8a-7e3ef5c96ae8.png)

## Abstract
Recent learning-based multi-view stereo (MVS) methods show excellent performance with dense cameras and small depth ranges. However, non-learning based approaches still outperform for scenes with large depth ranges and sparser wide-baseline views, in part due to their PatchMatch optimization over pixelwise estimates of depth, normals, and visibility. In this paper, we propose an end-to-end trainable PatchMatch-based MVS approach that combines advantages of trainable costs and regularizations with pixelwise estimates. To overcome the challenge of the non-differentiable PatchMatch optimization that involves iterative sampling and hard decisions, we use reinforcement learning to minimize expected photometric cost and maximize likelihood of ground truth depth and normals. We incorporate normal estimation by using dilated patch kernels, and propose a recurrent cost regularization that applies beyond frontal plane-sweep algorithms to our pixelwise depth/normal estimates. We evaluate our method on widely used MVS benchmarks, ETH3D and Tanks and Temples (TnT), and compare to other state of the art learning based MVS models. On ETH3D, our method outperforms other recent learning-based approaches and performs comparably on advanced TnT.

## Qualitative Results

|Reference Image| COLMAP | Ours |
|---|---|---| 
|<img src="https://user-images.githubusercontent.com/5545126/127351607-8bb7bb45-7f3f-484a-8dae-df261f48c95d.jpg" width="480">|<img src="https://user-images.githubusercontent.com/5545126/127351579-a34335f7-68fa-42e0-b618-d9e9f9af43e2.png" width="480"> |<img src="https://user-images.githubusercontent.com/5545126/127351573-2eeea200-fd94-4457-b7c1-c9a8201c37d2.png" width="480">|


## Resources
- [Code](https://github.com/leejaeyong7/patch-match-mvs-net)
- Paper / Supp.Material : Coming Soon

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
