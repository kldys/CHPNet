# A good teacher learns while teaching: Heterogeneous architectural knowledge distillation for fast MRI reconstruction

This repository contains the implementation of a lightweight image reconstruction network that leverages advanced knowledge distillation strategies to achieve high-quality results with reduced computational overhead. The model is designed for real-time applications and demonstrates robust performance comparable to larger networks.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Experiments](#experiments)
- [Citation](#citation)

## Introduction

The proposed lightweight image reconstruction network integrates a compact architecture with a knowledge distillation framework to balance efficiency and reconstruction quality. The model employs a dynamic adjustment mechanism to improve the effectiveness of knowledge transfer, ensuring robust performance even in resource-constrained environments.

Key contributions include:
1. Proposal for a dynamic distillation strategy that incorporates a knowledge correction mechanism for the teacher network, which allows it to adjust its parameters and structure by feedback received from the student network, simultaneously rectifying any preexisting errors in its knowledge.
2. Proposal for a lightweight heterogeneous network distillation framework that leverages the strengths of both CNN and ViT networks.
3. Application of a sub-pixel convolution strategy to reconstruct the down-sampling low-dimensional features in the network, effectively using information from the original feature map. Achievement of traditional up-sampling effects without introducing additional zero- or one-padding preserves network simplicity while improving the accuracy of reconstruction of low-level features.
4. Propossal for a novel method for fast MRI reconstruction that achieves performance comparable to state-of-the-art techniques while using significantly fewer parameters.

## Features

- **Lightweight Architecture**: Optimized for real-time deployment with minimal computational cost.
- **High-Quality Reconstruction**: Achieves performance comparable to larger networks through robust feature learning.
- **Knowledge Distillation**: Transfers knowledge effectively from teacher to student networks.
- **Dynamic Adjustment Mechanism**: Improves the accuracy and robustness of student network learning.

## Installation

To set up the environment, ensure you have Python installed and execute the following commands:

```bash
git clone https://github.com/kldys/CHPNet
cd CHPNet```

## Experiments

### Dataset

The experiments were conducted on the following datasets:

- **Dataset A**: Description of Dataset A.
- **Dataset B**: Description of Dataset B.

### Configuration

- **Teacher Network**: Model name and architecture.
- **Student Network**: Model name and architecture.
- **Training Settings**: Optimizer, learning rate, number of epochs, etc.


