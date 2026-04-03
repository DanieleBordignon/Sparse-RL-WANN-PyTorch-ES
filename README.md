# Are Weight Agnostic Neural Networks Truly Weight Agnostic?

**An Empirical Analysis via Evolution Strategies**

> This project investigates whether the performance of Weight Agnostic Neural Networks (WANNs) in sparse reward environments can be improved through targeted weight optimization using OpenAI Evolution Strategies (ES). Our findings confirm that explicit weight tuning is unnecessary, the topology itself drives the solution.

📄 **[Read the Full Paper (PDF)]()**

---

## Table of Contents

- [Overview](#overview)
- [Contributions](#contributions)
- [Key Findings](#key-findings)
- [Environments](#environments)
- [Network Topologies](#network-topologies)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Train WANN Topologies](#1-train-wann-topologies)
  - [2. Train Individual Weights via ES](#2-train-individual-weights-via-es)
  - [3. Visualize Network Topologies](#3-visualize-network-topologies)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

[Weight Agnostic Neural Networks (WANNs)](https://arxiv.org/abs/1906.04358) discover network architectures that can solve tasks even when all connections share a single, uniform weight value. This makes them particularly effective in **sparse reward** environments, where traditional RL algorithms (PPO, DQN, Q-Learning) fail due to the lack of dense gradient information.

This project extends the work of [Lingenberg & Reuter (2025)](https://github.com/Tobi-Tob/Sparse-RL-Wann) by:

1. **Reconstructing** the optimal WANN topologies in **PyTorch** (with verified numerical equivalence to the original NumPy implementation).
2. **Applying OpenAI Evolution Strategies (ES)** to optimize individual connection weights of these fixed topologies.
3. **Demonstrating** that targeted weight optimization yields **no performance improvement** over the shared-weight baseline, confirming the weight-agnostic hypothesis.

---

## Contributions

The following scripts represent the main contributions of this project, built on top of the original [Sparse-RL-Wann](https://github.com/Tobi-Tob/Sparse-RL-Wann) codebase:

| Script | Description |
|:---|:---|
| `train_wann_weights.py` | ES weight optimization for WANN topologies. Runs the full training pipeline: shared-weight baseline evaluation, evolutionary weight optimization, statistical testing, and training curve generation. |
| `convert_wann.py` | PyTorch re-implementation of the WANN forward pass. Converts `.out` topology files into differentiable `nn.Module` objects, supporting both shared-weight and individual-weight modes. |
| `verify_consistency.py` | Numerical equivalence check between the original NumPy forward pass and the PyTorch re-implementation, ensuring identical outputs. |
| `print_networks.py` | Network topology visualization. Generates high-resolution PDF renderings of the evolved WANN architectures with labeled inputs, outputs, and activation functions. |

---

## Key Findings

| Configuration | SMC Discrete | SMC Continuous | Lunar Lander |
|:---|:---:|:---:|:---:|
| **Shared Baseline** | **126.2 ± 33.2** | **106.9 ± 13.5** | **67.0 ± 11.4** |
| ES Trained | 128.6 ± 33.3 | 109.8 ± 13.9 | 68.0 ± 11.8 |

> *Average steps ± std. dev. over 200 trials (lower is better).*

✅ **WANNs are genuinely weight agnostic**, topology alone encodes the control policy.\
✅ **Result holds across all complexity levels**, from 4 connections (SMC Discrete) to 71 connections (SMC Continuous).\
✅ **No reward shaping required**, compact, interpretable policies discovered via evolutionary search.

---

## Environments

All three environments use **sparse reward** formulations, where the agent receives non-zero feedback only upon successful task completion:

| Environment | State Dim | Action Space | Reward |
|:---|:---:|:---|:---|
| **Sparse Mountain Car (Discrete)** | 2 | 3 discrete actions (left, none, right) | R_t = 100 · 0.99^t on success, 0 otherwise |
| **Sparse Mountain Car (Continuous)** | 2 | 1 continuous value ∈ [-1, 1] | R_t = 100 · 0.99^t on success, 0 otherwise |
| **Lunar Lander** | 8 | 4 discrete actions (noop, left, main, right) | R_t = 100 · 0.999^t on safe landing, 0 otherwise |

---

## Network Topologies

The optimal WANN topologies discovered by [Lingenberg & Reuter](https://github.com/Tobi-Tob/Sparse-RL-Wann) and reconstructed in PyTorch. Nodes indicate the activation function used; edges represent connections whose weights are assigned a uniform shared value for baseline evaluation.

### Sparse Mountain Car - Discrete (2 hidden nodes, 4 connections)

<p align="center">
  <img src="plots/smc_discrete.png" alt="SMC Discrete network topology" width="80%">
</p>

### Sparse Mountain Car - Continuous (31 nodes, 71 connections)

<p align="center">
  <img src="plots/smc_continuous.png" alt="SMC Continuous network topology" width="95%">
</p>

### Lunar Lander (1 hidden node, 11 connections)

<p align="center">
  <img src="plots/lunar_lander.png" alt="Lunar Lander network topology" width="80%">
</p>

---

## Results

The ES optimization across 100 generations fails to improve upon the shared-weight baseline for any environment. The training curves below show the average steps per episode of the updated network weights (evaluated over 100 independent trials at each generation). The red dashed line indicates the best shared-weight baseline. Performance merely oscillates near the baseline without exhibiting any converging downward trend.

### Sparse Mountain Car - Discrete

<p align="center">
  <img src="plots/smc_discrete_training.png" alt="SMC Discrete training curve" width="70%">
</p>

### Sparse Mountain Car - Continuous

<p align="center">
  <img src="plots/smc_continuous_training.png" alt="SMC Continuous training curve" width="70%">
</p>

### Lunar Lander

<p align="center">
  <img src="plots/lunar_lander_training.png" alt="Lunar Lander training curve" width="70%">
</p>

> Despite the SMC Continuous network having **71 trainable parameters** (vs. just 4 for SMC Discrete), the weight optimization still yields no improvement, confirming that the topology itself encodes the solution regardless of architectural complexity.

---

## Installation

**Prerequisites:** Python 3.8+ with `pip`.

```bash
# Clone the repository
git clone https://github.com/DanieleBordignon/Sparse-RL-WANN-PyTorch-ES.git
cd Sparse-RL-WANN-PyTorch-ES

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Train WANN Topologies

Run the evolutionary algorithm to discover new WANN architectures (from the [original codebase](https://github.com/Tobi-Tob/Sparse-RL-Wann)):

```bash
python wann_train.py
```

Pre-evolved champion networks for all three environments are already provided in `champions/`.

### 2. Train Individual Weights via ES

Run the OpenAI Evolution Strategies weight optimizer on a selected environment:

```bash
python train_wann_weights.py
```

Edit the `ENV_NAME` variable in `main()` to select the target environment:

```python
ENV_NAME = 'smc_discrete'    # Options: 'smc_discrete', 'smc_continuous', 'lunar_lander'
```

**Training configuration:**

| Parameter | Value | Description |
|:---|:---:|:---|
| `POP_SIZE` | 16 | Population size (×2 with antithetic sampling = 32 candidates) |
| `GENERATIONS` | 100 | Number of ES generations |
| `LEARNING_RATE` | 0.05 | Step size for weight updates |
| `SIGMA` | 0.05 | Noise standard deviation for perturbations |
| `N_EVALS_PER_CAND` | 10 | Episodes per candidate evaluation |

**Output:**
- Console log with per-generation statistics (population avg, policy eval, best overall)
- Training curve saved to `plots/<env_name>_training.pdf`
- Welch's t-test comparing baseline vs. trained weights (200-episode evaluation)

### 3. Visualize Network Topologies

Generate high-resolution PDF visualizations of all three WANN architectures:

```bash
python print_networks.py
```

---

## Acknowledgements

This project builds upon:

- **[Sparse-RL-Wann](https://github.com/Tobi-Tob/Sparse-RL-Wann)** by Lingenberg & Reuter, WANN topology evolution and sparse environment implementations.
- **[Weight Agnostic Neural Networks](https://arxiv.org/abs/1906.04358)** (Gaier & Ha, NeurIPS 2019), original WANN framework from [Google Brain Tokyo Workshop](https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease).
- **[Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864)** (Salimans et al., 2017), OpenAI ES algorithm.

---

## License

This project is licensed under the **Apache License 2.0**, see [LICENSE](LICENSE) for details.