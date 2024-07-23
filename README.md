# Mini ViTs Trainer :octocat:


Welcome to the Mini ViTs Trainer! This repository is designed to be a centralized resource for developers, researchers, and tech enthusiasts to explore and implement the cutting-edge in machine learning technology. We focus on providing high-quality code, pre-trained models, and comprehensive documentation to help you leverage state-of-the-art technologies across various applications. 🚀

## What You Can Do with This Repository

### 1. Train Various ViTs on Small Datasets with Just a single GPU 💻

This repository emphasizes machine learning practices in resource-constrained environments, specifically training various Vision Transformers (ViTs) on small datasets. We understand that not everyone has access to large-scale computational resources, so we provide optimized algorithms and practical advice for effective training with just a single GPU. Our goal is to lower the entry barrier, enabling more people to innovate and research with the latest technologies.

#### Train a Vision Transformer on a 2080Ti in Just 4 Hours 🕓

Efficient training of machine learning models on accessible hardware is a critical need for researchers and developers with limited resources. This repository enables practical training of Vision Transformer (ViT) models from scratch on a single 2080Ti GPU within just four hours 🚀. It's a step towards making advanced AI models more accessible to a broader audience.

#### What You Can Find Here

- **Training Code for Mainstream Small-Scale Datasets** 📚: We provide ready-to-use training scripts for CIFAR-10, CIFAR-100, SVHN, and Tiny ImageNet. These datasets allow for quick experimentation and development, ideal for those looking to work within hardware or time constraints.

- **Variety of ViT Models** 🔍: The repository hosts a selection of over ten ViT variants, including but not limited to ViT, CaiT, CvT, MobileViT, CrossViT, DeepViT, RegionViT, RvT, and T2T. This variety ensures that you can explore different architectures and find the one that suits your project's needs best.

This repository aims to provide a straightforward, no-frills approach to training ViT models 😊. By focusing on small-scale datasets and offering a selection of ViT architectures, we hope to facilitate easier entry into the field of computer vision for those without access to large computational resources.

Whether you're exploring different ViT architectures for academic purposes or developing applications with state-of-the-art AI technology, this repository offers the tools and resources needed to achieve efficient results without extensive computational power 💪.


# Getting Started with Our Repository 🚀

Welcome to our repository! Before diving into training Vision Transformers (ViTs) on a 2080Ti in just 4 hours, let's set up your environment. Follow these steps to ensure you have all the necessary packages and dependencies installed 🛠️.

## Requirements 📋

To get started, you'll need to install the required Python packages. We recommend creating a new `conda` environment and then using `pip` to install the provided `requirements.txt` packages. Follow these steps in your terminal:

```bash
conda create --name myenv python=3.8
conda activate myenv
pip install -r requirements.txt
```

Here are the specific versions of the packages you'll need:

- `einops==0.7.0` `numpy==1.18.0` `ply==3.11` `pyasn1-modules==0.2.8` `PyQt5-sip==12.11.0` `requests-oauthlib==1.3.0` `tensorboard==2.10.0` `timm==0.6.7` `torch==1.10.0` `torchsummary==1.5.1` `torchvision==0.11.1` `tqdm==4.66.1` `colorama==0.4.6` `setuptools==59.5.0` `google-auth==2.6.0` `google-auth-oauthlib==0.4.2` `grpcio==1.48.2` `protobuf==3.16.0` `six==1.16.0`

Ensure that you have Python 3.7 or newer installed on your machine before proceeding with the installation of these packages 🐍. This setup is essential for running the training scripts and utilizing the full potential of our repository 🌟.

## Train & Test

To train any model in our repository, you can use the following command, which allows you to specify the GPU devices, model, dataset, and number of epochs. This flexibility ensures you can tailor the training process to your specific needs and hardware capabilities.

```bash
CUDA_VISIBLE_DEVICES={0-7} python train.py --model {model name} --dataset {dataset name}
```

### Available Model Names

You can specify one of the following model names using the `--model` parameter:

`vit`, `mbv2`, `swin-s`, `swin-t`, `cait-s`, `cait-t`, `t2t-b`, `cvt-b`, `deepvit-t`, `deepvit-s`, `rvt-s`, `rvt-t`, `regionvit-b`, `crossvit-t`, `crossvit-s`, `xcit-s`, `xcit-t`, `twinssvt-b`, `twinssvt-s`

Each model offers unique configurations and capabilities, ranging from standard ViT models to specialized architectures like MobileViT, Swin Transformer, and more.

### Supported Datasets

The repository supports training and testing on the following datasets, which you can specify using the `--dataset` parameter:

- `CIFAR10` - A dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
- `CIFAR100` - Similar to CIFAR10 but with 100 classes.
- `SVHN` - The Street View House Numbers (SVHN) dataset, a real-world image dataset for developing machine learning and object recognition algorithms.
- `T-IMNET` (Tiny ImageNet) - A scaled-down version of the ImageNet dataset, consisting of 200 classes, each with 500 training images.

### Example Usage

For example, to train a vit model on the CIFAR10 dataset for 100 epochs, your command would look like this:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model vit --dataset CIFAR10 --epochs 100
```

### Calculating Model Parameters and Computational Complexity 📊

To understand the efficiency and demands of each model, you can calculate the model's parameters and its computational complexity using the `count.py` script. This script provides valuable insights into the model's size and the computational resources it requires, which is crucial for evaluating its suitability for your specific hardware and use case. 🔍

#### How to Use 🛠️

Run the following command in your terminal, replacing `{model name}` with the name of the model you wish to evaluate:

```bash
python count.py --model {model name}
```

## Adding and Running Your Own Custom Model 🛠️

Integrating and testing your own custom models within our framework is straightforward and can be accomplished in just two steps.

### Step 1: Add Your Custom Model

First, you need to place your custom-defined model into the `models` folder. Your model should be a PyTorch nn.Module. Ensure your model file follows the best practices for defining PyTorch models, including proper initialization and forward pass definitions.

### Step 2: Register Your Model

Next, you will need to import and register your model within the `create_model.py` file. To do this, add an import statement at the top of the file to include your custom model. Then, add the following code snippet to the model creation logic, replacing 'model name' with your model's unique identifier and Your_Model with the class name of your custom model.

```bash
elif args.model == 'model name':
    model = Your_Model(**kargs)
```

# Experimental Results

The table below presents the experimental results of various models on the Tiny-ImageNet dataset, including model accuracy, the number of parameters, and floating-point operations (Flops).

| Model             | Tiny-ImageNet Accuracy (%) | # Parameters | Flops     |
|-------------------|---------------------------|--------------|-----------|
| [T2T-T](https://arxiv.org/abs/2101.11986)        | 53.92                     | 1.07M        | 78.23M    |
| [RvT-T](https://arxiv.org/abs/2104.09864)        | 50.65                     | 1.10M        | 57.61M    |
| [Swin-T](https://arxiv.org/abs/2103.14030)       | 54.93                     | 1.06M        | 38.90M    |
| [CaiT-T](https://arxiv.org/abs/2103.17239)       | 54.76                     | 1.03M        | 61.85M    |
| [XCiT-T](https://arxiv.org/abs/2106.09681)       | 56.78                     | 0.96M        | 51.44M    |
| [ViT-Lite](https://openreview.net/pdf?id=YicbFdNTTy)     | 53.46                     | 1.11M        | 69.64M    |
| [DeepViT-T](https://arxiv.org/abs/2103.11886)    | 34.64                     | 0.99M        | 62.96M    |
| [RegionViT-T](https://arxiv.org/abs/2106.02689)  | 54.32                     | 0.97M        | 29.38M    |
| [CrossViT-T](https://arxiv.org/abs/2103.14899)   | 47.03                     | 1.04M        | 57.59M    |
| [T2T-S](https://arxiv.org/abs/2101.11986)      | 41.25                     | 2.56M        | 52.96M    |
| [RvT-S](https://arxiv.org/abs/2104.09864)      | 55.51                     | 2.72M        | 145.09M   |
| [Swin-S](https://arxiv.org/abs/2103.14030)     | 58.61                     | 2.93M        | 95.55M    |
| [CaiT-S](https://arxiv.org/abs/2103.17239)     | 59.21                     | 2.77M        | 164.46M   |
| [XCiT-S](https://arxiv.org/abs/2106.09681)     | 60.09                     | 2.81M        | 157.54M   |
| [ViT-Small](https://openreview.net/pdf?id=YicbFdNTTy)  | 55.74                     | 2.76M        | 176.06M   |
| [DeepViT-S](https://arxiv.org/abs/2103.11886)  | 44.45                     | 2.54M        | 162.85M   |
| [Twins SVT-S](https://arxiv.org/abs/2104.13840)| 37.13                     | 2.76M        | 197.00M   |
| [RegionViT-S](https://arxiv.org/abs/2106.02689)| 53.96                     | 2.86M        | 53.82M    |
| [CrossViT-S](https://arxiv.org/abs/2103.14899) | 52.70                     | 2.40M        | 126.11M   |
| [T2T-B](https://arxiv.org/abs/2101.11986)      | 58.46                     | 13.45M       | 853.02M   |
| [CvT-B](https://arxiv.org/pdf/your_paper21.pdf)      | 55.88                     | 6.52M        | 102.56M   |
| [MobileViTv2](https://arxiv.org/abs/2206.02680) | 58.28                     | 8.17M        | 189.77M   |
| [Twins SVT-B](https://arxiv.org/abs/2104.13840)| 49.24                     | 9.04M        | 308.74M   |
| [RegionViT-B](https://arxiv.org/abs/2106.02689)| 57.83                     | 12.39M       | 195.02M   |
