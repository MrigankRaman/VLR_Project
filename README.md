# Back to Source: Diffusion-Driven Test-Time Adaptation
Official repository of "Back to Source: Diffusion-Driven Test-Time Adaptation".

## Introduction
This repo is based on [ilvr](https://github.com/jychoi118/ilvr_adm) and [mim](https://github.com/open-mmlab/mim). We mainly provide the following functionality:
+ Adapt an image using a diffusion model.
+ Test using self-ensemble given image pairs.

## File_structure

The basic file structure is shown as follows:
```
DDA
├── ckpt
│   └── *.pth
├── dataset
│   ├── generated
│   ├── imagenetc
│   └── README.md
├── image_adapt
│   ├── guided_diffusion
│   ├── scripts
│   └── *.py
├── model_adapt
│   ├── configs
│   └── *.py
├── README.md
├── download_ckpt.sh
├── image_adapt.sh
└── test.sh
```

Structure of dataset can be found [here](./dataset/README.md).

## Installation
```bash
conda create -n DDA python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch  # Should be cudatoolkit=[CUDA_VERSION]
pip install cupy_cuda113  # Should be cupy_cuda[CUDA_VERSION]
pip install openmim blobfile tqdm pandas
conda install mpi4py
mim install mmcv-full 
mim install mmcls
```

## Pre-trained Models

We provide a bash script for easy downloading by just run ```bash download_ckpt.sh```.
If you want to download a certain model, you can find the corresponding ```wget``` command and only run the line.
We also provide the source of such checkpoints, more details of which are hidden in the links as follows.

The pre-trained diffusion model: [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) from [guided-diffusion](https://github.com/openai/guided-diffusion).

The pre-trained recognition model: [mm_models](./ckpt/README.md). 


## Usage

### Diffusion Generation

```bash
bash image_adapt.sh
```

### Ensemble Test

You can choose corruption type/severity in [configs](./model_adapt/configs/_base_/datasets). Ensemble methods can be set according to [args](./model_adapt/test_ensemble.py#L99).


The basic command form is 
```bash
python model_adapt/test_ensemble.py [config] [checkpoint] --metrics accuracy --ensemble [ensemble method]
```

Or you can just run
```bash
bash test.sh
```


## Results

| Architecture    | Data/Size   | Params/FLOPs| ImageNet Acc. |Source-Only* | MEMO* | DDA*|
|:---------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----:|:---:|
| RedNet-26      | 1K/224^2  | 1.7/9.2   | 76.0          | 15.0 | 20.6 | **25.0** |
| ResNet-50      | 1K/224^2  | 4.1/25.6  | 76.6          | 18.7 | 24.7 | **27.3** |
| Swin-T         | 1K/224^2  | 4.5/28.3  | 81.2          | 33.1 | 29.5 | **37.0** |
| ConvNeXt-T     | 1K/224^2  | 4.5/28.6  | 81.7          | 39.3 | 37.8 | **41.4** |
| Swin-B         | 1K/224^2  | 15.1/87.8 | 83.4          | 41.0 | 37.0 | **42.0** |
| ConvNeXt-B     | 1K/224^2  | 15.4/88.6 | 83.9          | 45.6 | 45.8 | **46.1** |

Columns with * are ImageNet-C Acc.
