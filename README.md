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

| Dataset & Model | 2nd Best Schedule (Baseline) | LODE Accuracy | Improvement |
| :--- | :--- | :--- | :--- |
| **Fa-MNIST (CNN)** | 93.6% | **93.8%** | +0.2% |
| **Fa-MNIST (ResNet18)** | 93.7% | **93.9%** | +0.2% |
| **CIFAR-100 (ResNet18)** | 74.0% | **74.9%** | +0.9% |
| **ImageNet (ResNet34)** | 73.9% | **74.5%** | +0.6% |
| **NLP (Transformer)** | 58.5% | **59.8%** | +1.3% |

<img src="/images/f2.png" height="300">

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
uv sync
```
> **Note:** The requirements installs `jax[cpu]`, to run this model with CUDA support if you want GPU acceleration please install the appropriate jax flavour.
> To do this please visit here (https://docs.jax.dev/en/latest/installation.html) for the latest methods for GPU and TPU compatible JAX installations, noting mainly the version of the CUDA drivers on your machine (i.e. 12.X, 13.X)

Run the tests:
```shell
uv run pytest
```

## Training a LODE-scheduler
To train a LODE-scheduler, one needs to collate the learning rate, training loss, and validation accuracy of a set of prior training runs. For example, 
```shell
python experiments/train_cifar100.py --schedule cosine --lr 0.01 --seed 1
```
will train and save these metrics for a ResNet18 on the CIFAR-100 dataset using a cosine-decay schedule, with a peak learning rate of 0.01, for a random seed 1. From experiments, we find using 5 seeds per hyper-parameter configuration to be sufficient for the training dataset.

To train the latent-ODE model, we strongly reccomend to use a path-minimised latent ODE (https://arxiv.org/abs/2410.08923) for which intructions of how to train, save, and use are here: https://github.com/SampsonML/path-minimized-latent-odes. Once a model is trained and saved, the experiment may be ran via,
```shell
python experiments/lode_cifar100.py --schedule cosine --lr 0.01 --seed 1 --path /path/to/latentODE/
```
For questions and issues and access to trained models, please contact matt.sampson@princeton.edu

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
├── tests/                         # Unit tests
│   ├── test_lode_schedule.py      # Scheduler logic verification
│   ├── test_lode_shapes.py        # Tensor shape & architecture checks
│   ├── test_models.py             # Model initialization smoke tests
│   ├── test_schedules.py          # Schedule interpolation tests
│   └── test_utils.py              # Math/Hessian utility verification
├── images/                        # Demo images for docs
├── pyproject.toml                 # Dependencies and project metadata
├── uv.lock                        # Reproducible dependency lockfile
├── LICENSE                        # Apache 2.0
└── README.md                      # Documentation
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
If you make use of the path-minimised latent-ODE of this repository please cite:
```bibtex
@article{sampson2025path,
  title={Path-minimizing latent ODEs for improved extrapolation and inference},
  author={Sampson, Matt L and Melchior, Peter},
  journal={Machine Learning: Science and Technology},
  volume={6},
  number={2},
  pages={025047},
  year={2025},
  publisher={IOP Publishing}
}
```
