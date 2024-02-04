# LKCA Official Repository :octocat:

Welcome to the official GitHub repository for LKCA! This repository is designed to be a centralized resource for developers, researchers, and tech enthusiasts to explore and implement the cutting-edge in machine learning technology. We focus on providing high-quality code, pre-trained models, and comprehensive documentation to help you leverage state-of-the-art technologies across various applications. 🚀

## What You Can Do with This Repository

### 1. Official Repository of LKCA :books:

As the official repository for LKCA, we offer a suite of tools and libraries to support the development of machine learning and artificial intelligence projects. Whether you're looking for the latest research findings or reliable code implementations, this is your go-to starting point.

### 2. Train Various ViTs on Small Datasets with Just a GPU or CPU 💻

This repository emphasizes machine learning practices in resource-constrained environments, specifically training various Vision Transformers (ViTs) on small datasets. We understand that not everyone has access to large-scale computational resources, so we provide optimized algorithms and practical advice for effective training with just a single GPU or CPU. Our goal is to lower the entry barrier, enabling more people to innovate and research with the latest technologies.

### Train a Vision Transformer on a 2080Ti in Just 4 Hours

Efficient training of machine learning models on accessible hardware is a critical need for researchers and developers with limited resources. This repository enables practical training of Vision Transformer (ViT) models from scratch on a single 2080Ti GPU within just four hours. It's a step towards making advanced AI models more accessible to a broader audience.

### Train a Vision Transformer on a 2080Ti in Just 4 Hours 🕓

Efficient training of machine learning models on accessible hardware is a critical need for researchers and developers with limited resources. This repository enables practical training of Vision Transformer (ViT) models from scratch on a single 2080Ti GPU within just four hours 🚀. It's a step towards making advanced AI models more accessible to a broader audience.

#### What You Can Find Here

- **Training Code for Mainstream Small-Scale Datasets** 📚: We provide ready-to-use training scripts for CIFAR-10, CIFAR-100, SVHN, and Tiny ImageNet. These datasets allow for quick experimentation and development, ideal for those looking to work within hardware or time constraints.

- **Variety of ViT Models** 🔍: The repository hosts a selection of over ten ViT variants, including but not limited to ViT, CaiT, CvT, MobileViT, CrossViT, DeepViT, RegionViT, RvT, and T2T. This variety ensures that you can explore different architectures and find the one that suits your project's needs best.

This repository aims to provide a straightforward, no-frills approach to training ViT models 😊. By focusing on small-scale datasets and offering a selection of ViT architectures, we hope to facilitate easier entry into the field of computer vision for those without access to large computational resources.

Whether you're exploring different ViT architectures for academic purposes or developing applications with state-of-the-art AI technology, this repository offers the tools and resources needed to achieve efficient results without extensive computational power 💪.


# Getting Started with Our Repository 🚀

Welcome to our repository! Before diving into training Vision Transformers (ViTs) on a 2080Ti in just 4 hours, let's set up your environment. Follow these steps to ensure you have all the necessary packages and dependencies installed 🛠️.

## Requirements 📋

To get started, you'll need to install the required Python packages. This can be easily done using `pip` and the provided `requirements.txt` file. Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

Here are the specific versions of the packages you'll need:

- `einops==0.7.0`
- `numpy==1.18.0`
- `ply==3.11`
- `pyasn1-modules==0.2.8`
- `PyQt5-sip==12.11.0`
- `requests-oauthlib==1.3.0`
- `tensorboard==2.10.0`
- `timm==0.6.7`
- `torch==1.10.0`
- `torchsummary==1.5.1`
- `torchvision==0.11.1`
- `tqdm==4.66.1`
- `colorama==0.4.6`
- `setuptools==59.5.0`
- `google-auth==2.6.0`
- `google-auth-oauthlib==0.4.2`
- `grpcio==1.48.2`
- `protobuf==3.16.0`
- `six==1.16.0`

Ensure that you have Python 3.7 or newer installed on your machine before proceeding with the installation of these packages 🐍. This setup is essential for running the training scripts and utilizing the full potential of our repository 🌟.

## Train & Test

To train any model in our repository, you can use the following command, which allows you to specify the GPU devices, model, dataset, and number of epochs. This flexibility ensures you can tailor the training process to your specific needs and hardware capabilities.

```bash
CUDA_VISIBLE_DEVICES={0-7} python train.py --model {model name} --dataset {dataset name} --epochs {int}
