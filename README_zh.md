# BridgePix2Pix: 基于简化扩散薛定谔桥的指令图像编辑

![项目横幅](docs/banner.png)

## 目录

- [项目概述](#项目概述)
- [特性](#特性)
- [架构](#架构)
- [安装](#安装)
- [数据集准备](#数据集准备)
- [使用指南](#使用指南)
  - [训练模型](#训练模型)
  - [评估模型](#评估模型)
  - [推理生成](#推理生成)
- [实验](#实验)
- [结果展示](#结果展示)
- [贡献指南](#贡献指南)
- [许可证](#许可证)
- [致谢](#致谢)

## 项目概述

**BridgePix2Pix** 是一个基于 [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) 的高级框架，旨在通过集成 **简化扩散薛定谔桥（Simplified Diffusion Schrödinger Bridge, S-DSB）** 方法，提升指令引导的图像编辑模型的性能。该项目结合了隐扩散模型的优势和 S-DSB 的理论创新，实现了更高效的训练过程和更高质量的图像生成效果。

## 特性

- **高级图像编辑能力**：能够根据复杂的自然语言指令进行多种图像编辑操作。
- **简化扩散薛定谔桥集成**：通过引入 S-DSB 方法，提升模型的收敛速度和生成质量。
- **CLIP 相似度评估**：利用 CLIP 模型确保生成图像与编辑指令的高度匹配。
- **全面评估**：包括定量指标和定性分析，全面评估模型性能。
- **模块化代码库**：易于扩展和定制，适合进一步研究和开发。

## 架构

BridgePix2Pix 主要由以下核心组件组成：

1. **编码器（Encoder）**：使用预训练的 VAE 将输入图像编码为潜在表示。
2. **简化扩散薛定谔桥（S-DSB）模块**：在潜在空间中构建从原始图像分布到编辑后图像分布的过渡。
3. **解码器（Decoder）**：将潜在表示解码为编辑后的图像。
4. **CLIP 文本编码器**：将编辑指令转换为嵌入向量，作为条件输入引导扩散过程。

![架构图](docs/architecture.png)

## 安装

### 前提条件

- **操作系统**：建议使用 Linux（如 Ubuntu 18.04+）
- **Python**：3.8 或更高版本
- **GPU**：推荐使用支持 CUDA 的 NVIDIA GPU（如 RTX 系列）

### 克隆代码仓库

```bash
git clone https://github.com/yourusername/BridgePix2Pix.git
cd BridgePix2Pix
```

### 创建虚拟环境

使用 `conda` 创建虚拟环境：

```bash
conda create -n bridgepix2pix python=3.10
conda activate bridgepix2pix
```

或者使用 `venv` 创建虚拟环境：

```bash
python3 -m venv bridgepix2pix_env
source bridgepix2pix_env/bin/activate
```

### 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*注意：确保安装了支持 CUDA 的 PyTorch 版本。可以通过以下命令安装：*

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

## 数据集准备

### MagicBrush 数据集

BridgePix2Pix 使用 [MagicBrush](https://github.com/OSU-NLP-Group/MagicBrush) 数据集，该数据集包含了大量手工标注的图像编辑指令对。

### 下载数据集

1. **克隆 MagicBrush 仓库**

    ```bash
    git clone https://github.com/OSU-NLP-Group/MagicBrush.git
    ```

2. **整理数据**

    将数据集放置在 `datasets/magicbrush/` 目录下：

    ```bash
    mkdir -p datasets/magicbrush
    cp -r MagicBrush/data/* datasets/magicbrush/
    ```

### 数据预处理

1. **图像处理**

    - **调整尺寸**：将图像统一调整为 256x256 像素。
    - **归一化**：将像素值归一化到 [-1, 1] 范围。

2. **文本处理**

    - **文本清理**：处理编辑指令文本，去除特殊字符，统一大小写等。
    - **文本编码**：使用 CLIP 文本编码器将编辑指令转换为嵌入向量。

### 更新数据加载器

修改 InstructPix2Pix 的数据加载器，适配 MagicBrush 数据集的格式，确保每个数据样本包含原始图像、编辑指令和目标图像。

## 使用指南

### 训练模型

训练 BridgePix2Pix 模型（集成 S-DSB 方法）：

```bash
python train.py --config configs/bridgepix2pix_config.yaml
```

**配置参数说明（configs/bridgepix2pix_config.yaml）：**

```yaml
model:
  encoder: "path/to/pretrained_encoder.pth"
  decoder: "path/to/pretrained_decoder.pth"
  sdsb_enabled: true
  lambda_sdsb: 0.5
  lambda_clip: 0.5

training:
  batch_size: 16
  learning_rate: 1e-4
  optimizer: "adamw"
  epochs: 100
  scheduler: "cosine"
  save_interval: 10

dataset:
  path: "datasets/magicbrush/"
  image_size: 256
  preprocess: true
```

### 评估模型

评估训练好的模型在验证集上的性能：

```bash
python evaluate.py --config configs/bridgepix2pix_config.yaml --model checkpoints/bridgepix2pix_best.pth
```

### 推理生成

根据输入图像和编辑指令生成编辑后的图像：

```bash
python inference.py --config configs/bridgepix2pix_config.yaml --model checkpoints/bridgepix2pix_best.pth --input_image path/to/image.jpg --instruction "将猫换成戴帽子的狗，并将背景改为山脉景观。"
```

## 实验

### 基线模型：InstructPix2Pix

基线模型基于原始的 InstructPix2Pix 架构，未集成 S-DSB 方法。用于对比 BridgePix2Pix 的性能提升。

### 核心模型：BridgePix2Pix（集成 S-DSB）

核心模型通过引入 Simplified Diffusion Schrödinger Bridge 方法，对基线模型进行了优化，旨在提升模型的收敛速度和生成质量。

### 实验设置

- **数据集**：MagicBrush
- **评估指标**：
  - **CLIP 相似度**：衡量生成图像与编辑指令的匹配程度。
  - **FID 分数（Frechet Inception Distance）**：评估生成图像的质量和多样性。
- **超参数设置**：
  - 学习率：1e-4
  - 批量大小：16
  - 优化器：AdamW
  - 训练轮数：100
  - 损失函数权重：`lambda_sdsb` = 0.5，`lambda_clip` = 0.5

## 结果展示

### 训练损失曲线

![训练损失](docs/training_loss.png)

*说明：展示基线模型和 BridgePix2Pix 模型的训练损失随时间的变化情况。BridgePix2Pix 模型表现出更快的收敛速度和更低的最终损失值。*

### CLIP 相似度权衡曲线

![CLIP权衡](docs/clip_tradeoff.png)

*说明：展示在不同 CLIP 相似度损失权重下，CLIP 相似度与图像质量（FID 分数）之间的权衡关系。BridgePix2Pix 在保持较高 CLIP 相似度的同时，图像质量指标表现优异。*

### 复杂指令编辑示例

| 原始图像 | 编辑指令 | BridgePix2Pix 生成结果 |
| -------- | -------- | ---------------------- |
| ![原始图像1](docs/original1.png) | "将猫换成戴帽子的狗，并将背景改为山脉景观。" | ![编辑后图像1](docs/edited1.png) |
| ![原始图像2](docs/original2.png) | "将红色汽车变成蓝色，并添加一个日落背景。" | ![编辑后图像2](docs/edited2.png) |

*说明：展示 BridgePix2Pix 在处理复杂和多方面编辑指令时的生成效果，模型能够准确地按照指令进行图像编辑。*

## 贡献指南

我们欢迎任何形式的贡献！无论是报告问题、建议新功能，还是提交代码，您的参与对我们非常重要。

### 如何贡献

1. **Fork 仓库**

2. **创建新分支**

    ```bash
    git checkout -b feature/YourFeatureName
    ```

3. **进行更改**

4. **提交更改**

    ```bash
    git commit -m "添加您的提交信息"
    ```

5. **推送到您的 Fork**

    ```bash
    git push origin feature/YourFeatureName
    ```

6. **打开 Pull Request**

    - 前往原始仓库，点击 "New pull request" 按钮。

### 代码规范

- **遵循 PEP8 标准**：保持代码风格一致，提升可读性。
- **编写文档**：为新增功能编写相应的文档和注释。
- **测试覆盖**：确保更改经过充分测试，避免引入错误。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 致谢

- 感谢 [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) 提供的基础代码库。
- 感谢 [MagicBrush](https://github.com/OSU-NLP-Group/MagicBrush) 提供的全面数据集。
- 感谢 [Hugging Face Diffusers](https://github.com/huggingface/diffusers) 提供的扩散模型实现。
- 感谢 [OpenAI CLIP](https://github.com/openai/CLIP) 提供的强大文本-图像嵌入能力。
- 感谢《Simplified Diffusion Schrödinger Bridge》论文的作者们，他们的研究为本项目的核心贡献提供了重要启发。

---

*如有任何问题或需要支持，请联系 [your.email@example.com](mailto:your.email@example.com)。*


## 启动训练：
```bash
python main.py --name 0 --base configs/s_dsb_train_magicbrush.yaml --train --gpus "0,1,2,3" --strategy ddp
```

## 启动推理：
```bash
python inference.py --config configs/bridgepix2pix_config.yaml --model checkpoints/bridgepix2pix_best.pth --input_image path/to/image.jpg --instruction "将猫换成戴帽子的狗，并将背景改为山脉景观。"
```