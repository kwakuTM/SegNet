# PyTorch-SegNet
## Semantic Segmentation using SegNet implemented in PyTorch

<p align="left">
<a href="https://www.youtube.com/watch?v=iXh9aCK3ubs" target="_blank"><img src="https://i.imgur.com/agvJOPF.gif" width="350"/></a>
<img src="images/i_3.jpg" width="126"/>
<img src="images/s3.png" width="126"/>
<img src="images/i_5.jpg" width="126"/>
<img src="images/s5.png" width="126"/>
</p>
<p align="center">
<img src="images/segnet.png" width="900"/>
</p>

### Requirements

* pytorch >=0.4.0
* torchvision ==0.2.0
* tensorboard_logger
* scipy
* tqdm

### Usage

**To train the model :**

```
python main.py [-h] [--dataroot [DATAROOT]]
               [[--train-data [IMG_PATH]] --label-data [LBL_PATH]
 
  --dataroot            root folder for project
  --img_path            Path of the input image
  --lbl_path            Path of the labelled image
```
