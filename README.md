
# DoNet: Deep De-overlapping Network for Cytology Instance Segmentation [CVPR2023]

This is the official pytorch implementation of DoNet, please refer the [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Jiang_DoNet_Deep_De-Overlapping_Network_for_Cytology_Instance_Segmentation_CVPR_2023_paper.html) and [demo](https://www.youtube.com/watch?v=VhDYeudFCFQ) for more details.


## Introduction
### Abstract
Cell instance segmentation in cytology images has significant importance for biology analysis and cancer screening, while remains challenging due to 1) the extensive overlapping translucent cell clusters that cause the ambigu- ous boundaries, and 2) the confusion of mimics and debris as nuclei. In this work, we proposed a De-overlapping Network (DoNet) in a decompose-and-recombined strategy. A Dual-path Region Segmentation Module (DRM) explicitly decomposes the cell clusters into intersection and complement regions, followed by a Semantic Consistency-guided Recombination Module (CRM) for integration. To further introduce the containment relationship of the nucleus in the cytoplasm, we design a Mask-guided Region Proposal Strategy (MRP) that integrates the cell attention maps for inner-cell instance prediction. We validate the proposed approach on ISBI2014 and CPS datasets. Experiments show that our proposed DoNet significantly outperforms other state-of-the-art (SOTA) cell instance seg- mentation methods. 
![](https://github.com/DeepDoNet/DoNet/blob/main/figure/F2_framework.png)

### Visualization
![](https://github.com/DeepDoNet/DoNet/blob/main/figure/F5_visualization.png)
### Results
ISBI2014 dataset
| Methods                | mAP | Dice | F1 | AJI |
|------------------------|---------------|----------------|--------------|---------------|
| Mask R-CNN          | 59.09         | 91.15          | 92.54        | 77.07         |
| Cascade R-CNN   | 62.45         | 91.29          | 92.51        | 77.91         |
| Mask Scoring R-CNN| 63.56         | 91.28          | 91.87        | 75.14         |
| HTC          | 59.62         | 91.39          | 88.08        | 75.00         |
| Occlusion R-CNN | 62.35 | 91.75          | 93.18        | 78.64         |
| Xiao et al.   | 57.34         | 91.70          | 92.75        | 78.29         |
| DoNet                  | **64.02**     | **92.13**      | **93.23**    | **79.05**     |

CPS dataset
| Methods                | mAP | Dice| F1| AJI|
|------------------------|---------------|----------------|--------------|---------------|
| Mask R-CNN    | 48.28 $\pm$ 3.10  | 89.32 $\pm$ 0.50  | 85.07 $\pm$ 2.01 | 69.20 $\pm$ 2.27 |
| Cascade R-CNN | 47.87 $\pm$ 3.27  | 89.24 $\pm$ 0.44  | 83.33 $\pm$ 1.65 | 68.86 $\pm$ 3.55 |
| Mask Scoring R-CNN| 48.38 $\pm$ 3.13  | 89.39 $\pm$ 0.24  | 82.98 $\pm$ 1.86 | 67.45 $\pm$ 2.45 |
| HTC           | 47.60 $\pm$ 3.56  | 89.08 $\pm$ 0.51  | 81.30 $\pm$ 2.56 | 66.35 $\pm$ 2.84 |
| Occlusion R-CNN| 48.14 $\pm$ 2.84  | 89.08 $\pm$ 0.28  | 85.69 $\pm$ 2.28 | 69.51 $\pm$ 2.45 |
| Xiao et al.| 48.53 $\pm$ 2.85  | 89.29 $\pm$ 0.24  | 85.46 $\pm$ 2.60 | 69.37 $\pm$ 2.88 |
| DoNet                  | 49.43 $\pm$ 3.83  | **89.54 $\pm$ 0.25** | 85.51 $\pm$ 2.33 | 70.08 $\pm$ 2.84 |
| DoNet w/ Aug.          | **49.65 $\pm$ 3.52** | 89.50 $\pm$ 0.38 | **86.30 $\pm$ 2.01** | **70.56 $\pm$ 2.34** |

## Dataset
Download official ISBI 2014 dataset from [Overlapping Cervical Cytology Image Segmentation Challenge](https://cs.adelaide.edu.au/~carneiro/isbi14_challenge/dataset.html).

You can also download our [preprocessed ISBI 2014 datasets](https://drive.google.com/drive/folders/15PEz7JQNDQ9Y_sR1LkaGOB3yZjzSgbek?usp=sharing) (COCO format annotations, amodal instance GT generation).

## Installation
See [INSTALL.md](INSTALL.md).

## QuickStart
### Inference and Visualization with Pre-trained Models (on ISBI)
Use `customized_tools/inference.py` 

Download [ISBI pre-trained model](https://drive.google.com/file/d/1ORIWqIotRVK5YNc3r4p0NHcUsWYSYGgs/view?usp=drive_link)
### Evaluation with Pre-trained Models (on ISBI)
Use `customized_tools/evaluate.py`
**The evaluation code rely on a modified pycoco package that provides a new function `iouIntUni` to compute intersection over union between masks, return iou, intersection, union together. For installation of the modified pycoco package, please refer to https://github.com/Amandaynzhou/MMT-PSM**

### Training on your own dataset
Use `customized_tools/preprocess.py` to process your dataset.
Then, pick a model in `customized_tools/trainings`.
For details, see [customized_tools/trainings/README.md](README.md)

## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citation
```
@InProceedings{Jiang_2023_CVPR,
    author    = {Jiang, Hao and Zhang, Rushan and Zhou, Yanning and Wang, Yumeng and Chen, Hao},
    title     = {DoNet: Deep De-Overlapping Network for Cytology Instance Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {15641-15650}
}
```


## Acknowledgment
The code of DoNet is built on [detectron2](https://github.com/facebookresearch/detectron2) and [ORCNN](https://github.com/waiyulam/ORCNN), many thanks for the Third Party Libs.

## Question
Feel free to email us if you have any questions:

Rushan Zhang(rzhangbq@connect.ust.hk), Hao Jiang(hjiangaz@cse.ust.hk)

