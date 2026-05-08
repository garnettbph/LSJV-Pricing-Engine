# Advanced Risk Engine: Local Stochastic Jump-Volatility (LSJV)

This project implements a **Particle Method calibration** for a **Bates-based Stochastic Local Volatility** model, utilizing PyTorch's Autograd engine for Algorithmic Adjoint Differentiation (AAD) to compute exact pathwise Greeks.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [The Mathematical Framework](#the-mathematical-framework)
3. [Technology Stack & Architecture](#technology-stack--architecture)
4. [Installation & Usage](#installation--usage)
5. [Interactive Prototyping (The Notebook)](#interactive-prototyping-the-notebook)
6. [Repository Structure](#repository-structure)

---

## Project Overview
Standard derivatives pricing models (like Black-Scholes) assume constant volatility and continuous asset paths. In reality, equity markets exhibit **volatility clustering**, **leverage effects (skew)**, and **discrete jumps** (crashes or overnight gaps). 

To capture this empirical reality, quantitative desks rely on Stochastic Local Volatility (SLV) models. This engine pushes that boundary further by building an **LSJV (Local Stochastic Jump-Volatility)** framework. It blends the Merton Jump-Diffusion model with Heston Stochastic Volatility, perfectly anchored to the market's Local Volatility surface via Gyöngy's Theorem.

By leveraging PyTorch, the engine bypasses slow PDE solvers in favor of a heavily vectorized Monte Carlo Particle Method, executing massive path simulations on the GPU in seconds rather than minutes.

---

## The Mathematical Framework

### 1. Underlying Dynamics (Under Risk-Neutral Measure $\mathbb{Q}$)
The asset price $S_t$ and its variance $v_t$ are governed by the following system of Stochastic Differential Equations (SDEs):

$$dS_t = (r - q - \lambda \bar{\mu}) S_t dt + L(S_t, t)\sqrt{v_t} S_t dW_t^{(1)} + S_{t-}(e^J - 1) dN_t$$

$$dv_t = \kappa(\theta - v_t) dt + \xi \sqrt{v_t} dW_t^{(2)}$$

$$\mathbb{E}[dW_t^{(1)} dW_t^{(2)}] = \rho dt$$

* **$dN_t$**: A Poisson process with intensity $\lambda$ controlling discrete market jumps.
* **$J$**: Jump size, normally distributed $J \sim \mathcal{N}(\mu_J, \sigma_J^2)$.
* **$\bar{\mu}$**: Jump compensator ensuring the process remains a martingale.
* **$L(S_t, t)$**: The deterministic Leverage Function.

### 2. Calibration via Gyöngy's Theorem
To ensure the model perfectly prices vanilla options (matching the market's implied volatility surface), the marginal distribution of the model must match the market. According to Gyöngy's theorem, we must solve for $L(S_t, t)$ such that:

$$\sigma_{LV}^2(S_t, t) = L^2(S_t, t) \mathbb{E}[v_t | S_t]$$

### 3. The Particle Method Engine
Solving the 2D Fokker-Planck PDE for the leverage function is notoriously unstable. Instead, this engine uses the **Particle Method**. 
At each time step, we simulate $N$ paths (particles) and estimate the conditional expectation $\mathbb{E}[v_t | S_t]$ via **Non-Parametric Regression**. To handle this on a GPU without memory exhaustion, we sort the asset paths and apply a vectorized 1D Gaussian Convolution Kernel.

### 4. Algorithmic Adjoint Differentiation (AAD)
Standard risk systems calculate Greeks (Delta, Vega) via "bumping" (finite difference), requiring full re-simulations of the Monte Carlo engine. This framework utilizes **PyTorch's Autograd**, maintaining the computation graph of the simulation to calculate the exact, analytical derivative of the payoff with respect to the initial state variables ($\partial V / \partial S_0$, $\partial V / \partial v_0$) in a single backward pass.

---

## Technology Stack & Architecture
* **PyTorch:** Core mathematical engine. Handles tensorized Monte Carlo simulations, GPU allocation, and AAD (Backpropagation).
* **yFinance:** Live data pipeline. Dynamically pulls historical data to estimate jump parameters and live options chains to extract At-The-Money (ATM) implied variance.
* **Plotly:** Interactive, browser-based data visualization to empirically justify the model's jump and volatility assumptions.

---

## Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ installed. It is highly recommended to run this in an environment with CUDA/GPU support, though it will automatically fall back to CPU if necessary.

```bash
git clone [https://github.com/garnettbph/LSJV-Pricing-Engine.git](https://github.com/garnettbph/LSJV-Pricing-Engine.git)
cd lsjv-pricing-engine
pip install -r requirements.txt
```

### CLI Execution
The engine is packaged as a Command Line Interface (CLI) tool. You can run it on any liquid ticker that has an active options chain.

### Basic Run (Defaults to SPY with 50,000 paths):
```bash
python main.py
```

### (e.g., TSLA, 100,000 paths, finer time steps):
```bash
python main.py --ticker TSLA --paths 100000 --steps 504
```
Upon completion, the engine will output the Option Price, Pathwise Delta, Pathwise Vega, and automatically generate an interactive HTML dashboard ([TICKER]_dynamics_dashboard.html) visualizing the empirical market data.

---

## Interactive Prototyping (The Notebook)
To understand how the math translates into the final PyTorch engine, check out the Jupyter Notebook located in the notebooks/ directory: notebooks/LSJV_Prototyping.ipynbThis notebook serves as the "brainstorming" sandbox for the project. 
It includes:
* Step-by-step visual derivations of the Bates jump components.
* Inline mathematical proofs for the Log-Euler time-stepping scheme.
* Interactive Plotly charts demonstrating the difference between constant volatility (Black-Scholes) and stochastic jumps (LSJV) on simulated paths.
* A playground to manually tweak parameters (like Heston $\kappa$ or Poisson $\lambda$) to see their immediate impact on the implied volatility surface.
If you want to modify the core engine, start by experimenting in this notebook.

## Repository Structure
```text
LSJV-Pricing-Engine/
│
├── lsjv_engine/                # Core Python Package
│   ├── __init__.py             
│   ├── data_pipeline.py        # yfinance integration & parameter estimation
│   ├── core_math.py            # PyTorch Particle Method & SDE execution
│   └── visualizer.py           # Plotly HTML dashboard generation
│
├── notebooks/                  # Interactive Prototyping
│   └── LSJV_Prototyping.ipynb  # Step-by-step model derivation and visual brainstorming
│
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
├── .gitignore                  
└── README.md
```
---
*Disclaimer: This is for educational and portfolio demonstration purposes. Do not use for live trading without extensive surface calibration optimizers.*
