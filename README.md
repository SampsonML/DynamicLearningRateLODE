# Dynamic learning rate scheduling with latent ODEs
[![arXiv](https://img.shields.io/badge/arXiv-2410.08923-<COLOR>.svg)](https://arxiv.org/abs/2509.23052)
[![Blog](https://img.shields.io/badge/Blog-link-orange.svg)](https://msampson.net/blog/2025/lode-scheduler/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview 
We learn a latent representation of training dynamics, training loss, validation accuracy and learning rate, from the observation of prior runs, which one would expect to have run anyway during a standard hyperparameter sweep. By encoding training loss, validation accuracy, and learning rate into a latent space and evolving these quantities via an ODE, the system can simulate how training would unfold with different parameters, namely the learning rate schedule. 
<img src="/images/model_schematic_.png" height="500">

### Comparison to other schedules
Across Fashion-MNIST, CIFAR-100, ImageNet, and even a Transformer language model, the LODE scheduler performs better than the baselines: cosine, OneCycle, exponential decay, hypergradient descent, schedule-free, and reinforcement learning controllers. Quite surprisingly we see that the best performing learning rate schedules determined by our LODE scheduler often suggests significantly higher learning rates at early times than what we see in the best parametric schedules.

<img src="/images/f2.png" height="250">

Not only did models reach higher accuracy, they also landed in flatter regions of the loss landscape—which hints towards stronger model generalization.

Key highlights:
- Consistently superior test accuracy across CNNs, ResNets, and Transformers.
- Flatter minima (confirmed via Hessian eigenvalue analysis).
- Computational cost only ~25% higher than simple parametric schedules, but cheaper than RL-based methods.

## Installation

> [!IMPORTANT]  
> This repository relies on JAX which is well maintained, but also very fast moving. Please use your favorite environment manager and create a fresh env before running this.
> [uv](https://docs.astral.sh/uv/#projects) is particularly nice 

Clone the repository and install dependencies:

```shell
git clone https://github.com/SampsonML/DynamicLearningRateLODE.git
cd DynamicLearningRateLODE
pip install -e .
# or using uv
uv pip install -e .
```
> **Note:** The requirements installs `jax[cpu]`, to run this model with CUDA support if you want GPU acceleration please install the appropriate jax flavour.
> To do this please visit here (https://docs.jax.dev/en/latest/installation.html) for the latest methods for GPU and TPU compatible JAX installations, noting mainly the version of the CUDA drivers on your machine (i.e. 12.X, 13.X)

## Repository Structure

```text
DynamicLearningRateLODE/
├── src/
│   └── dynamic_lode/              # Core package root
│       ├── core/                  
│       │   ├── __init__.py
│       │   ├── lode.py            # LatentODE architecture
│       │   └── lode_scheduler.py  # Latent-ODE based learning rate scheduler
│       ├── models/                # Model architectures tested
│       │   ├── __init__.py
│       │   ├── CNN.py
│       │   ├── ResNet18.py
│       │   ├── ResNet34.py
│       │   └── ResNetFaMNIST.py
│       └── utils/                 # Utility functions
│           ├── __init__.py
│           ├── hessian.py         # Power-iteration eigenvalue approx
│           └── schedule.py        # JIT-optimized buffer scheduling 
├── experiments/                   # Research runs
│   ├── lode_cifar100.py           # Main dynamic scheduling experiment
│   └── train_cifar100.py          # Baseline training script (create training data)
├── images/                        # Demo images for docs
├── pyproject.toml                 # Project metadata and dependencies
├── LICENSE                        # Apache 2.0
└── README.md                      # Docs
```

### Citation
If you make use of this code please cite:
```bibtex
@article{sampson2025dynamics,
  title={Dynamics of Learning: Generative Schedules from Latent ODEs},
  author={Sampson, Matt L and Melchior, Peter},
  journal={arXiv preprint arXiv:2509.23052},
  year={2025}
}
```
