# A good teacher learns while teaching: Heterogeneous architectural knowledge distillation for fast MRI reconstruction

Reconstructing high-quality images from under-sampled k-space data is a core problem in fast Magnetic Resonance Imaging. To address this issue, in this study, a super-resolution reconstruction technique based on knowledge distillation across heterogeneous networks.
Employing a corrective strategy to refine the erroneous knowledge within the teacher network, the proposed method enhances the accuracy of the knowledge imparted to the student network by incorporating accurate edge information.
By reconstructing the mathematical model, the proposed method ensures that the convolutional neural network-based student network can effectively learn the remote feature information from the visual transformer-based teacher network, and achieve high-quality image reconstruction using heterogeneous network knowledge transfer.
Experiments on multiple publicly available datasets, in both k-space and image domain, demonstrate that the proposed method achieves performance close to state-of-the-art methods while maintaining a low level of complexity, and can efficiently recover the original information from k-space under-sampled data.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Citation](#citation)

## Introduction

The proposed lightweight image reconstruction network integrates a compact architecture with a knowledge distillation framework to balance efficiency and reconstruction quality. The model employs a dynamic adjustment mechanism to improve the effectiveness of knowledge transfer, ensuring robust performance even in resource-constrained environments.

Key contributions include:
* Proposal for a dynamic distillation strategy that incorporates a knowledge correction mechanism for the teacher network, which allows it to adjust its parameters and structure by feedback received from the student network, simultaneously rectifying any preexisting errors in its knowledge.
* Proposal for a lightweight heterogeneous network distillation framework that leverages the strengths of both CNN and ViT networks.
* Application of a sub-pixel convolution strategy to reconstruct the down-sampling low-dimensional features in the network, effectively using information from the original feature map. Achievement of traditional up-sampling effects without introducing additional zero- or one-padding preserves network simplicity while improving the accuracy of reconstruction of low-level features.
* Propossal for a novel method for fast MRI reconstruction that achieves performance comparable to state-of-the-art techniques while using significantly fewer parameters.

## Features

- **Lightweight Architecture**: Optimized for real-time deployment with minimal computational cost.
- **High-Quality Reconstruction**: Achieves performance comparable to larger networks through robust feature learning.
- **Knowledge Distillation**: Transfers knowledge effectively from teacher to student networks.
- **Dynamic Adjustment Mechanism**: Improves the accuracy and robustness of student network learning.

## Installation

To set up the environment, ensure you have Python installed and execute the following commands:

```bash
git clone https://github.com/kldys/CHPNet
cd CHPNet
```
## Dataset

The experiments were conducted on the following datasets:

- **FastMRI**: A large-scale MRI dataset collected for fast MRI provides both raw k-space data and DICOM images. It includes a single-coil subset and a multi-coil subsets.  In contrast to a single-coil MRI, multi-coil MRI poses a significantly greater challenge due to issues with phase differences and sensitivity mapping, which exceed the scope of the present study.  Here, the single-coil subset was used, which contained 1,594 scans acquired for diagnostic knee MRI, to utilize for fast MRI reconstruction experiments. Division of the training and testing datasets adhered to the guidelines provided by the data source. 
- **Brain tumor MRI**: This dataset consists of 7023 brain MRI images. For the dataset, 400 images were randomly selected for training, and an additional 20 images (five for each brain tumor type: glioma, meningioma, no tumor, and pituitary) were selected for testing.
- **Flickr2K**: This dataset comprises 2650 color 2K high-resolution natural images. In this study, 400 high-resolution natural images were randomly selected from Flickr2K for training to boost the performance of the network. During the training phase, the dataset comprises an equal number of natural and MRI images.
- **Set5**: The dataset consists of five high-resolution natural images and their corresponding low-resolution versions, and is designed to validate the migration performance of the algorithm on different types of images.
- 
## Citation

If you find this repository helpful, please cite our work:

```bibtex
@article{
}

