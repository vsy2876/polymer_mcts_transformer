# Autoregressive Polymer PSELFIES Generation with MCTS

This repository contains an optimization and sequence generation pipeline for polymer discovery, combining a causal **Autoregressive Transformer Language Model** with a **Monte Carlo Tree Search (MCTS)** exploration loop.

## Architecture & Methodology
Unlike non-autoregressive discrete diffusion architectures, this framework utilizes token-by-token sequence modeling to construct valid polymer building blocks mapped directly onto polymer-adapted SELFIES (PSELFIES).

* **Generative Backbone:** Causal Autoregressive Transformer optimized for handling long-range structural dependencies across discrete token vocabulary transitions.
* **Search Strategy:** Integrated AlphaZero-style Monte Carlo Tree Search framework. The autoregressive network serves as the prior policy network to guide token node selection during tree expansion.
* **Representation Robustness:** Built entirely on top of localized `tokenizer_pselfies.py` regular expressions, removing chemical invalidity traps by leveraging the formal grammatical boundaries of PSELFIES.

---

## Repository Structure

* `model_ar.py` — Core causal transformer token prediction logic and sequence modeling configurations.
* `tokenizer_pselfies.py` — Specialized vocabulary map and regex parser handling polymer-adapted string serialization (`PSELFIES`).
* `pretrain_ar.py` / `pretrain_ar.sh` — Baseline generative script to train the causal language model on structural text grammar before search execution.
* `finetune_mcts_ar.py` / `mcts_ar.sh` — Active search loop performing node exploration, rollout tracking, sequence tree updates, and optimized discovery streams.

---

## Getting Started

### 1. Pre-training Configuration
To run baseline causal pre-training on cluster nodes:
`bash pretrain_ar.sh`

### 2. MCTS Fine-Tuning Optimization
To execute the token search expansion loop, run the primary batch engine script:
`bash mcts_ar.sh`

---

## Research Attribution
This codebase is a component of ongoing graduate research at the Georgia Institute of Technology (School of Materials Science & Engineering).

**Copyright & Licensing** © 2026 Vansh Suresh Yadav. All rights reserved.  
This code is intended exclusively for private research evaluation. Copying, distributing, or modifying these files without explicit authorization is strictly prohibited.
