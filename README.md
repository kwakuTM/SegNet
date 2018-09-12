# PyTorch-SegNet
## Semantic Segmentation using SegNet implemented in PyTorch

<p align="center">
<a href="https://www.youtube.com/watch?v=iXh9aCK3ubs" target="_blank"><img src="https://i.imgur.com/agvJOPF.gif" width="364"/></a>
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
python main.py [-h]


```

**To test the model w.r.t. a dataset on custom images(s):**

```
python test.py [-h] [--model_path [MODEL_PATH]] [--dataroot [DATAROOT]]
               [[--train-data [IMG_PATH]] --label-data [LBL_PATH] [--out_path [OUT_PATH]]
 
  --model_path          Path to the saved model
  --dataset             Dataset to use ['pascal, camvid, ade20k etc']
  --dcrf                Enable DenseCRF based post-processing
  --img_path            Path of the input image
  --out_path            Path of the output segmap
```

=> README under constrcution