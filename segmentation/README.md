# Applying ViT-CoMer to Semantic Segmentation

Our segmentation code is developed on top of [MMSegmentation v0.20.2](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2).



## Usage

Install [MMSegmentation v0.20.2](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2).

```
# recommended environment: torch1.9 + cuda11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
pip install mmdet==2.22.0 # for Mask2Former
pip install mmsegmentation==0.20.2
ln -s ../detection/ops ./
cd ops & sh make.sh # compile deformable attention
```

## Main Results and Models

**ADE20K val**

| Method  | Backbone | Pretrain  | Lr schd | Crop Size | mIoU(SS/MS) | #Param  | Config  | Ckpt | Log |
|:----------:|:-------------:|:----------:|:-------:|:------:|:-------:|:-----:|:----:|:-------------:|
| UperNet | ViT-CoMer-T | []()                                                                                                 | 1×   | 52.1   | 45.8   | [config]()         | [ckpt]()  | [log]() |
| UperNet | ViT-CoMer-S | []()                                                                                                 | 3×   | 48.6   | 42.9   | [config]()         | [ckpt]()  | [log]() |
| UperNet | ViT-CoMer-B | []()                                                                                                 | 1×   | 52.0   | 45.5   | [config]()         | [ckpt]()  | [log]() |
| UperNet | ViT-CoMer-L | []()                                                                                                 | 3×   | 54.2   | 47.6   | [config]()         | [ckpt]()  | [log]() |

## Evaluation

To evaluate ViT-CoMer-T + UperNet (512) on ADE20k val on a single node with 8 gpus run:

```shell
sh test.sh
```


## Training

To train ViT-Adapter-T + UperNet on ADE20k on a single node with 8 gpus run:

```shell
sh train.sh
```
