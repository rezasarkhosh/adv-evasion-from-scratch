# adv-evasion-from-scratch

> Implementing classic evasion attacks (FGSM, PGD, C&W) from scratch in PyTorch — no `torchattacks`, no shortcuts, just the math and a small CNN.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-WIP%20v0.1.0-orange.svg)](#)

## Why this repo exists

This is **Sprint 1** of a hands-on AI security curriculum I'm running on myself: every two weeks, pick one attack or defense family, reimplement it from scratch, extend it, and ship a public writeup. Reading papers is cheap. Reimplementing them is where the actual learning happens.

The goal isn't a production-grade attack library — `torchattacks`, `foolbox`, and `cleverhans` already exist for that. The goal is to **understand each attack at the level where I could derive it on a whiteboard**, and to make the tradeoffs (perturbation budget, iterations, transferability) visible in plots anyone can rerun.

## What's implemented

| Attack | Status | Reference |
| --- | --- | --- |
| FGSM (Fast Gradient Sign Method) | ☐ Planned | [Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572) |
| PGD (Projected Gradient Descent) | ☐ Planned | [Madry et al., 2018](https://arxiv.org/abs/1706.06083) |
| C&W L2 (Carlini & Wagner) | ☐ Planned | [Carlini & Wagner, 2017](https://arxiv.org/abs/1608.04644) |

> 🚧 **WIP — v0.1.0.** This README is published before the code is finished, on purpose. Following along: [open issues](../../issues) track progress.

## Quickstart

```bash
git clone https://github.com/<your-username>/adv-evasion-from-scratch.git
cd adv-evasion-from-scratch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Train the baseline CNN on CIFAR-10:

```bash
python train_baseline.py --epochs 20 --out checkpoints/cnn_cifar10.pt
```

Run an attack:

```bash
# FGSM — single-step
python attacks/fgsm.py --model checkpoints/cnn_cifar10.pt --epsilon 0.03

# PGD — multi-step
python attacks/pgd.py --model checkpoints/cnn_cifar10.pt --epsilon 0.03 --steps 20 --alpha 0.007

# C&W — optimization-based
python attacks/cw.py --model checkpoints/cnn_cifar10.pt --confidence 0 --max-iter 1000
```

## Repo structure

```
adv-evasion-from-scratch/
├── attacks/
│   ├── fgsm.py          # Single-step gradient sign attack
│   ├── pgd.py           # Iterative projected gradient descent
│   └── cw.py            # Carlini & Wagner L2 attack
├── models/
│   └── small_cnn.py     # Baseline CIFAR-10 classifier
├── notebooks/
│   └── 01_visualize_attacks.ipynb   # Side-by-side adversarial vs. clean images
├── results/
│   ├── asr_vs_epsilon.png           # Attack success rate as ε grows
│   └── transferability_matrix.png   # Cross-model attack transfer
├── train_baseline.py
├── requirements.txt
└── README.md
```

## What I'm trying to learn (and what's worth replicating)

Each attack answers a different question. I'm tracking what surprised me as I implement each one — those notes will land in the writeup.

**FGSM** — How much can one gradient step buy an attacker? Cheap, but a useful baseline and the first thing every interview will ask about.

**PGD** — What does iterating buy you over a single step, and how does the projection back into the ε-ball actually work in code? PGD is the gold-standard first-order attack; if I can implement it cleanly, the math is internalized.

**C&W L2** — Why does framing the attack as an optimization problem (instead of a gradient step) produce smaller, more imperceptible perturbations? This one is the most algorithmically different from the first two and the one I expect to learn the most from.

## Planned experiments

- [ ] Attack success rate (ASR) vs. perturbation budget ε for each method on CIFAR-10
- [ ] Untargeted vs. targeted comparison across all three methods
- [ ] Transferability matrix: train attack on architecture A, evaluate on architectures B and C
- [ ] One pretrained ImageNet test: how small can ε get before the predicted label flips to something visually unrelated?

## Results

> 📊 Tables and plots will land here as experiments finish. Refreshed at the end of each sprint week.

## Things this repo is *not*

- Not a production attack library — use [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch) or [foolbox](https://github.com/bethgelab/foolbox) for that.
- Not novel research — these are reimplementations of well-known attacks for learning purposes.
- Not a defenses repo — that's a future sprint. Pull requests on attacks welcome; defense suggestions go in issues.

## References

1. Goodfellow, I., Shlens, J., & Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples.* ICLR. [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)
2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR. [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)
3. Carlini, N., & Wagner, D. (2017). *Towards Evaluating the Robustness of Neural Networks.* IEEE S&P. [arXiv:1608.04644](https://arxiv.org/abs/1608.04644)

## About the author

Reza Sarkhosh — MSc candidate in *ICT for Internet and Multimedia: Machine Learning for Healthcare* at the University of Padova, currently visiting TalTech for thesis research on federated learning security. Other relevant work:

- *Trust-Aware Client Scoring in Federated Learning via Contrastive Attention and Representation Clustering* — IEEE COMPSAC 2026 (full paper, 28.5% acceptance).
- Backdoor data-poisoning study on CARLA-based traffic-sign detection.

[GitHub](https://github.com/<your-username>) · [LinkedIn](https://linkedin.com/in/<your-handle>) · [Email](mailto:<your-email>)

## License

MIT — see [LICENSE](LICENSE).
