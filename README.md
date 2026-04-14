# SSL-Medical-Segmentation

<p align="center">
  <strong>A configuration-driven toolkit for 2D/3D semi-supervised medical image segmentation.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-SSL%20Training%20Framework-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Task-2D%20%26%203D%20SSL%20Segmentation-0A7EA4" alt="2D and 3D SSL segmentation">
  <img src="https://img.shields.io/badge/License-MIT-2EA043" alt="MIT license">
</p>

SSL-Medical-Segmentation keeps the outer project style of `SLab-Medical-Segmentation` while implementing the semi-supervised training pipeline directly inside this repository.

The repository is organized around one training entrypoint, one inference entrypoint, YAML-driven method/model selection, and clean experiment outputs. The public-facing workflow follows the configuration-driven style of SLab, while the semi-supervised methods, data pipeline helpers, evaluation utilities, and training losses are implemented as first-party project modules.

## Highlights

- One training entrypoint: `train.py` drives both 2D and 3D semi-supervised experiments.
- One inference entrypoint: `inference.py` keeps the same single-checkpoint prediction workflow as SLab.
- Configuration-driven method switching: choose a backbone in `config/arch/` and an SSL algorithm in `config/methods/`.
- SSL-specific losses only: `config/losses/` now keeps only `ssl_multiclass` and `ssl_binary`.
- SLab-style experiment outputs: each run saves `config.yaml`, TensorBoard logs, checkpoints, plots, and validation summaries.
- Project-internal implementation: data transforms, discriminators, ramps, evaluation, and SSL helpers no longer depend on external project paths.

## At A Glance

| Module | Current implementation |
| --- | --- |
| Task types | 2D semi-supervised segmentation / 3D semi-supervised segmentation |
| Training entrypoint | `train.py` |
| Inference entrypoint | `inference.py` |
| Dataset registry | `dataset/utils.py` |
| Model registry | `models/model_utils.py` |
| Method registry | `config/methods/` + `solver/ssl_methods.py` |
| Config directories | `config/arch`, `config/methods`, `config/losses`, `config/optimizer`, `config/scheduler` |
| Validation metrics | Dice, HD95 |
| Training metrics | Dice, IoU, Precision, Recall, Specificity, F1, Accuracy |
| Log directory | `logs/<expname>_<timestamp>/` |

## Integrated SSL Methods

- `fully_supervised`
- `mean_teacher`
- `entropy_minimization`
- `adversarial_network`
- `interpolation_consistency_training`
- `cross_pseudo_supervision`
- `cross_consistency_training`
- `uncertainty_aware_mean_teacher`
- `uncertainty_rectified_pyramid_consistency`
- `regularized_dropout`
- `deep_co_training`
- `fixmatch_standard_augs`
- `cross_teaching_between_cnn_transformer`

## Integrated Backbones

### 2D Backbones

- `unet_2d`
- `unetpp_2d`
- `deeplabv3_resnet50`
- `deeplabv3_resnet101`
- `fcn_2d`
- `segnet`
- `pspnet`
- `highresnet`
- `miniseg`
- `attention_unet_2d`
- `danet_2d`
- `transunet_2d`
- `swinunet_2d`
- `medformer_2d`

### 3D Backbones

- `Unetr`
- `UNet`
- `UNETR_PP`
- `segformer3d`
- `segmamba`
- `slim_unetr`
- `VTUNET`
- `AttentionUnet`
- `SwinUNETR`
- `UNetPP`
- `3DUXNET`
- `nnFormer`
- `EfficientMedNeXt_T`
- `EfficientMedNeXt_S`
- `EfficientMedNeXt_M`
- `EfficientMedNeXt_L`
- `nnUnet`
- `VNet`
- `MedFormer`
- `TransBTS`

## Supported Datasets

- `acdc`: 2D slice-based semi-supervised setting.
- `brats2019`: 3D HDF5 volume setting.

## Repository Structure

```text
.
├── train.py
├── inference.py
├── requirements.txt
├── config/
│   ├── arch/
│   │   ├── two_d/
│   │   └── three_d/
│   ├── losses/
│   ├── methods/
│   ├── optimizer/
│   └── scheduler/
├── dataset/
│   ├── acdc.py
│   ├── brats2019.py
│   └── utils.py
├── losses/
├── models/
│   └── model_utils.py
├── solver/
│   ├── optim.py
│   ├── scheduler.py
│   └── ssl_methods.py
└── utils/
```

## Quick Start

### 1. Installation

Python 3.10 is recommended.

```bash
conda create -n sslms python=3.10 -y
conda activate sslms

# Install the correct PyTorch build for your CUDA version first
pip install torch torchvision

pip install -r requirements.txt
```

### 2. Prepare The Dataset

#### ACDC

The code expects the following ACDC layout:

```text
data/ACDC/
├── train.list
├── train_slices.list
├── val.list
└── data/
    ├── patient001_frame01.h5
    └── slices/
        ├── patient001_frame01_slice_0.h5
        └── ...
```

#### BraTS2019

The code expects the following BraTS2019 layout:

```text
data/BraTS2019/
├── train.txt
├── val.txt
└── data/
    ├── case_000.h5
    └── ...
```

### 3. Train

#### ACDC + Mean Teacher

```bash
python train.py \
  --dataset_name acdc \
  --dataset_path /path/to/ACDC \
  --model_name unet_2d \
  --dimension 2d \
  --out_channels 4 \
  --method_name mean_teacher \
  --expname acdc_mt
```

#### ACDC + Cross Consistency Training

```bash
python train.py \
  --dataset_name acdc \
  --dataset_path /path/to/ACDC \
  --model_name unet_2d \
  --dimension 2d \
  --out_channels 4 \
  --method_name cross_consistency_training \
  --expname acdc_cct
```

#### BraTS2019 + Cross Pseudo Supervision

```bash
python train.py \
  --dataset_name brats2019 \
  --dataset_path /path/to/BraTS2019 \
  --model_name UNet \
  --dimension 3d \
  --out_channels 2 \
  --patch_size 96 96 96 \
  --batch 4 \
  --labeled_bs 2 \
  --labeled_num 25 \
  --total_labeled_num 250 \
  --method_name cross_pseudo_supervision \
  --expname brats_cps
```

### 4. Inference

```bash
python inference.py \
  --config logs/your_run/config.yaml \
  --checkpoint_path logs/your_run/checkpoint/model_best.pt \
  --input_path /path/to/image_or_directory \
  --output_dir outputs/inference
```

## Notes

- For multi-model methods such as CPS, R-Drop, and CNN-Transformer cross teaching, the primary inference path uses the first model by default.
- `config/losses/` is intentionally slimmed down to the project-native supervised presets `ssl_multiclass` and `ssl_binary`, and unrelated legacy loss presets were removed.
- `swinunet_2d` or `transunet_2d` can be used as the secondary model for `cross_teaching_between_cnn_transformer`.
- The public 2D and 3D backbone catalog is aligned with `SLab-Medical-Segmentation`; methods like CCT and URPC create method-level auxiliary predictions when the selected backbone only exposes a single segmentation head.
- The semi-supervised training utilities are implemented directly inside `dataset/`, `models/`, `solver/`, and `utils/`, without cross-project path injection.

## Acknowledgements

This repository integrates many open-source SSL medical image segmentation models into a unified experimental framework.  
We sincerely thank the original authors and the open-source community for making their work available for reproduction, comparison, and further extension.

This project was organized and advanced under the support of Prof. Shuang Song and Kangneng Zhou.  
Special thanks to [SSL4MIS](https://github.com/HiLab-git/SSL4MIS).

## Contact

If you are using this repository or planning to extend it, feel free to contact:

- Kangneng Zhou
- WeChat: `kangkangellis666`
- Email: `elliszkn@163.com`

## License

This project is released under the MIT License. See `LICENSE` for details.
