# Dynamic learning rate scheduling with latent ODEs
[![arXiv](https://img.shields.io/badge/arXiv-2410.08923-<COLOR>.svg)](https://arxiv.org/abs/2509.23052)
[![Blog](https://img.shields.io/badge/Blog-link-orange.svg)](https://msampson.net/blog/2025/lode-scheduler/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview 

<img src="/images/model_schematic_.png" height="500">


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
