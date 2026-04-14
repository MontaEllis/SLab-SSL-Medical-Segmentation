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

SSL-Medical-Segmentation 保留了 `SLab-Medical-Segmentation` 的外层工程风格，同时把半监督训练主流程完整实现在当前仓库内部。

仓库整体仍然采用单训练入口、单推理入口、YAML 驱动配置和统一实验输出的组织方式；外层保持 SLab 风格的调用体验，半监督方法、数据处理、评估和训练辅助模块则全部以内置代码实现。

## 项目亮点

- 单一训练入口：`train.py` 同时覆盖 2D/3D 半监督分割实验。
- 单一推理入口：`inference.py` 保留了和 SLab 一样的单 checkpoint 推理方式。
- 配置驱动切换：通过 `config/arch/` 选择 backbone，通过 `config/methods/` 选择 SSL 方法。
- 仅保留 SSL 专用 loss：`config/losses/` 现在只保留 `ssl_multiclass` 和 `ssl_binary`。
- 统一实验记录：自动保存 `config.yaml`、TensorBoard、checkpoint 和曲线图。
- 项目内原生实现：数据增强、判别器、ramp、评估和 SSL 辅助模块不再依赖外部项目路径。

## 一图看懂

| 模块 | 当前实现 |
| --- | --- |
| 任务类型 | 2D 半监督分割 / 3D 半监督分割 |
| 训练入口 | `train.py` |
| 推理入口 | `inference.py` |
| 数据集注册 | `dataset/utils.py` |
| 模型注册 | `models/model_utils.py` |
| 方法注册 | `config/methods/` + `solver/ssl_methods.py` |
| 配置目录 | `config/arch` / `config/methods` / `config/losses` / `config/optimizer` / `config/scheduler` |
| 验证指标 | Dice, HD95 |
| 训练指标 | Dice, IoU, Precision, Recall, Specificity, F1, Accuracy |
| 日志目录 | `logs/<expname>_<timestamp>/` |

## 已集成 SSL 方法

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

## 已集成 Backbone

### 2D Backbone

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

### 3D Backbone

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

## 支持的数据集

- `acdc`：2D slice 半监督设定。
- `brats2019`：3D HDF5 体数据设定。

## 仓库结构

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

## 快速开始

### 1. 环境安装

推荐使用 Python 3.10。

```bash
conda create -n sslms python=3.10 -y
conda activate sslms

# 请先根据你的 CUDA 环境安装对应版本的 PyTorch
pip install torch torchvision

pip install -r requirements.txt
```

### 2. 准备数据

#### ACDC

代码默认使用下面的 ACDC 目录结构：

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

代码默认使用下面的 BraTS2019 目录结构：

```text
data/BraTS2019/
├── train.txt
├── val.txt
└── data/
    ├── case_000.h5
    └── ...
```

### 3. 训练

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

### 4. 推理

```bash
python inference.py \
  --config logs/your_run/config.yaml \
  --checkpoint_path logs/your_run/checkpoint/model_best.pt \
  --input_path /path/to/image_or_directory \
  --output_dir outputs/inference
```

## 说明

- 对于 CPS、R-Drop、CNN-Transformer cross teaching 这类双模型方法，默认推理入口使用第一个模型。
- `config/losses/` 已经收紧为项目内的监督 loss 预设 `ssl_multiclass` 和 `ssl_binary`，和当前半监督方法无关的旧 loss 配置已经删除。
- `cross_teaching_between_cnn_transformer` 可以把 `swinunet_2d` 或 `transunet_2d` 作为第二个模型。
- 公开的 2D/3D backbone 列表和 `SLab-Medical-Segmentation` 对齐；像 CCT、URPC 这类需要多头输出的方法，在所选 backbone 只有单 segmentation head 时，会在方法层自动构造辅助预测分支。
- 半监督训练相关工具已经直接实现在 `dataset/`、`models/`、`solver/` 和 `utils/` 中，不再通过跨项目路径注入。

## 致谢

本仓库整合了大量公开半监督医学图像分割模型实现，并在统一训练接口下进行了工程化整理。  
感谢各原始模型作者和开源社区的贡献，使这些工作能够在统一框架中被复现、比较和扩展。

本项目由 Prof. Shuang Song 与 Kangneng Zhou 组织推进。  
特别感谢 [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)。

## 联系方式

如果你正在使用这个项目，或者想基于它继续扩展，欢迎联系：

- Kangneng Zhou
- WeChat: `kangkangellis666`
- Email: `elliszkn@163.com`

## License

This project is released under the MIT License. See `LICENSE` for details.
