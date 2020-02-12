# AGNN
Code for ICCV 2019 paper: Zero-shot Video Object Segmentation via Attentive Graph Neural Networks
#
![](../master/framework.png)
### Quick Start

## 说明
基于图注意力的视频目标分割算法
平台： windows 10 、 GTX1060

#### Testing

1. Install pytorch (version:1.0.1).

2. Download the pretrained model, put in the snapshots folder. Run 'test_iteration_conf_gnn.py' and change the davis dataset path, pretrainde model path and result path.

3. Run command:  python test_iteration_conf_gnn.py --dataset davis --gpus 0

4. Post CRF processing code: https://github.com/lucasb-eyer/pydensecrf

The pretrained weight can be download from [GoogleDrive](https://drive.google.com/open?id=1w4hWVC7ZTTVDJCQN6-vOVLY9JLJCru7G).

The segmentation results on DAVIS-2016, Youtube-objects and DAVIS-2017 datasets can be download from [GoogleDiver](https://drive.google.com/open?id=1w5nRgUdUz-OxUhEYjytYDXB_xa2r983_).

### Citation
If you find the code and dataset useful in your research, please consider citing:

@InProceedings{Wang_2019_ICCV,

author = {Wang, Wenguan and Lu, Xiankai and Shen, Jianbing and Crandall, David J. and Shao, Ling},

title = {Zero-Shot Video Object Segmentation via Attentive Graph Neural Networks},

booktitle = {The IEEE International Conference on Computer Vision (ICCV)},

year = {2019}
}

### Other related projects/papers:
[See More, Know More: Unsupervised Video Object Segmentation with Co-Attention Siamese Networks(CVPR19)](https://github.com/carrierlxk/COSNet)

[Saliency-Aware Geodesic Video Object Segmentation (CVPR15)](https://github.com/wenguanwang/saliencysegment)

[Learning Unsupervised Video Primary Object Segmentation through Visual Attention (CVPR19)](https://github.com/wenguanwang/AGS)

Any comments, please email: carrierlxk@gmail.com
