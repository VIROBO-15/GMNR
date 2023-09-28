<h1 align="center">Generative Multiplane Neural Radiance (GMNR) (ICCV 2023)</h1>
<p align="center">for 3D-Aware Image Generation</p>

<p align="center">
  <img width="90%" src="Figures/intro.jpg"/>
</p>

**Generative Multiplane Neural Radiance for 3D-Aware Image Generation.**<br>
[Amandeep Kumar](https://virobo-15.github.io/), [Ankan Kumar Bhunia](https://ankanbhunia.github.io/), [Sanath Narayan](https://sites.google.com/view/sanath-narayan/home), [Hisham Cholakkal](https://scholar.google.com/citations?user=bZ3YBRcAAAAJ&hl=en), [Rao Muhammad Anwer](https://scholar.google.fi/citations?user=_KlvMVoAAAAJ&hl=en), [Salman Khan](https://scholar.google.com.pk/citations?user=M59O9lkAAAAJ&hl=en), [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en), [Fahad Shahbaz Khan](https://scholar.google.com/citations?user=zvaeYnUAAAAJ&hl=en)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2304.01172.pdf)
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)](miscellaneous/to_be_announced.md)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](miscellaneous/to_be_announced.md)

Abstract: *We present a method to efficiently generate 3D-aware high-resolution images that are view-consistent across multiple target views. The proposed multiplane neural radiance model, named GMNR, consists of a novel α-guided view-dependent representation (α-VdR) module for learning view-dependent information. The α-VdR module, faciliated by an α-guided pixel sampling technique, computes the view-dependent representation efficiently by learning
viewing direction and position coefficients. Moreover, we propose a view-consistency loss to enforce photometric similarity across multiple views. The GMNR model can generate 3D-aware high-resolution images that are viewconsistent across multiple camera poses, while maintaining the computational efficiency in terms of both training and inference time. Experiments on three datasets demonstrate the effectiveness of the proposed modules, leading to favorable results in terms of both generation quality and inference time, compared to existing approaches. Our GMNR model generates 3D-aware images of 1024X1024 pixels with 17.6 FPS on a single V100.*

## :rocket: News
- **September 28, 2023** : Released code for GMNR
- **July 13, 2023** : GMNR accepted in ICCV 2023 &nbsp;&nbsp; :confetti_ball:

## Environment Setup

This code has been tested on Ubuntu 18.04 with CUDA 10.2.

```
conda env create -f environment.yml
```

# Training

Assume `GMNR_ROOT` represents the path to this repo:
```bash
cd /path/to/this/repo
export GMNR_ROOT=$PWD
```

## Set Up Virtual Environments

We need [MTCNN](https://github.com/ipazc/mtcnn), [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21), and [DeepFace](https://github.com/serengil/deepface) to complete the data processing and evaluation steps.

### MTCNN and DeepFace

We provide the `conda` environment yaml files for MTCNN and DeepFace:
- [mtcnn_env.yaml](../virtual_envs/mtcnn_env.yaml) for MTCNN;
- [deepface_env.yaml](../virtual_envs/deepface_env.yaml) for DeepFace.
```bash
conda env create -f mtcnn_env.yaml      # mtcnn_env
conda env create -f deepface_env.yaml   # deepface
```

### Deep3DFaceRecon_pytorch

**Note: GMPI authors has made the modification in code we have used the same modification. Please use [modified version](https://github.com/Xiaoming-Zhao/Deep3DFaceRecon_pytorch).** Please follow [the official instruction](https://github.com/Xiaoming-Zhao/Deep3DFaceRecon_pytorch#requirements) to setup the virtual environments and to download the pretrained models. There are two major steps:
1. Install some packages and setup the environment: see [this link](https://github.com/Xiaoming-Zhao/Deep3DFaceRecon_pytorch#installation);
2. Download some data: see [this link](https://github.com/Xiaoming-Zhao/Deep3DFaceRecon_pytorch#prepare-prerequisite-models).

Assume the code repo locates at `Deep3DFaceRecon_PATH`:
```bash
export Deep3DFaceRecon_PATH=/path/to/Deep3DFaceRecon_pytorch
```

## Download StyleGAN2 Checkpoints

Download [StyleGAN2's pretrained checkpoints](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/):
```bash
mkdir -p ${GMNR_ROOT}/ckpts/stylegan2_pretrained/transfer-learning-source-nets/
cd ${GMNR_ROOT}/ckpts/stylegan2_pretrained
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl ./transfer-learning-source-nets    # FFHQ256
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl ./transfer-learning-source-nets   # FFHQ512
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl ./transfer-learning-source-nets  # FFHQ1024
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl .   # AFHQCat
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl .   # MetFaces
```
## Preprocessing and dataset
For complete Installation, and dataset preparation, follow guidelines [here](https://github.com/apple/ml-gmpi/blob/672294b56f97bd621a87a7a33a39964abb30c0bc/docs/TRAIN_EVAL.md#preprocess-data)


## Train

Run the following command to start training GMNR. Results will be saved in `${GMNR_ROOT}/experiments`. We use 8 Tesla V100 GPUs in our experiments. We recommend 32GB GPU memory if you want to train the GMNR model.

```bash
python launch.py \
--run_dataset FFHQ1024 \
--nproc_per_node 1 \
--task-type gmnr \
--run-type train \
--master_port 8370
```

- `run_dataset` can be in `["FFHQ256", "FFHQ512", "FFHQ1024", "AFHQCat", "MetFaces"]`.
- Set `nproc_per_node` to be the number of GPUs you want to use.

## Evaluation
For evaluating the  FID/KID, Identity Metrics, Depth Metrics and pose accuracy Metric, follow the guidelines [here](https://github.com/apple/ml-gmpi/blob/main/docs/TRAIN_EVAL.md#evaluation).

## Citation

If you find our work helpful, please **star🌟** this repo and **cite📑** our paper. Thanks for your support!

```
@article{kumar2023generative,
  title={Generative Multiplane Neural Radiance for 3D-Aware Image Generation},
  author={Kumar, Amandeep and Bhunia, Ankan Kumar and Narayan, Sanath and Cholakkal, Hisham and Anwer, Rao Muhammad and Khan, Salman and Yang, Ming-Hsuan and Khan, Fahad Shahbaz},
  journal={arXiv preprint arXiv:2304.01172},
  year={2023}
}
```

## Acknowledgement
Our code is designed based on Generative Multiplane Images [GMPI](https://github.com/apple/ml-gmpi/tree/main).


## Contact
If you have any question, please create an issue on this repository or contact at **amandeep.kumar@mbzuai.ac.ae**

<hr />
