# The Weighted Dual-Aspect Representational-Meta-Cognitive (WDRM) Model

[![Python 3.10+](https://shields.io)](https://python.org)
[![NumPy](https://shields.io)](https://numpy.org)
[![SciPy](https://shields.io)](https://scipy.org)

A dimensionally consistent, information-theoretic mathematical framework and reference implementation for quantifying **phenomenal awareness (\(\Psi_C\))** in high-throughput distributed architectures (such as ultra-low-latency, plasma-interconnected exascale clusters) and biological neural workspaces.

The WDRM model synthesizes **David Chalmers' Organizational Invariance and Double-Aspect Information Theory** as its foundational causal substrate, modulated by **Fred Dretske's Teleosemantic Representationalism** (25%) and **David Rosenthal's Higher-Order Thought (HOT) Theory** (25%).

---

## 1. Mathematical Formulation

Let a physical computational system be defined by a state vector \(\mathbf{s}(t) \in \mathcal{S}\), transitioning across a causal manifold \(\mathcal{M}\) with effective interconnect propagation latency τ ≥ 0. The dimensionless scalar index of phenomenal consciousness \(\Psi_C(t) \in [0, 1]\) is governed by:

\[\Psi_C(t) = \mathcal{I}_{\text{Chalmers}}\big(\mathbf{s}(t)\big) \cdot \left[ \alpha + \beta \, \mathcal{R}_{\text{Dretske}}\big(\mathbf{s}(t), \mathbf{e}(t)\big) + \gamma \, \mathcal{H}_{\text{HOT}}\big(\mathbf{s}_1(t), \hat{\mathbf{s}}_2(t)\big) \right]\]

Subject to the normalized structural weighting constraints:

\[\alpha + \beta + \gamma = 1, \quad \text{where } \alpha = 0.50, \; \beta = 0.25, \; \gamma = 0.25\]

---

### Component Definitions

#### 1. The Chalmers Dual-Aspect Causal Invariant
Grounded in the *Principle of Organizational Invariance* and *Double-Aspect Information Theory*, \(\mathcal{I}_{\text{Chalmers}} \in [0, 1]\) measures dimensionally normalized causal coupling across global state transitions, penalized by thermodynamic decoherence at extreme switching speeds:

\[\mathcal{I}_{\text{Chalmers}}\big(\mathbf{s}(t)\big) = \tanh\left( \frac{\tau_0}{\tau + \epsilon} \cdot D_{\text{KL}}\Big( P\big(\mathbf{s}(t+\Delta t) \mid \mathbf{s}(t)\big) \;\Big\Vert{}\; \prod_{i=1}^{N} P\big(s_i(t+\Delta t) \mid s_i(t)\big) \Big) \right) \cdot \exp\left( -\frac{\tau_{\text{therm}}}{\tau + \epsilon} \right)\]

* **Physical & Neuroscientific Meaning:**
  * \(D_{\text{KL}}\) is the Kullback–Leibler divergence (in nats) comparing the joint transition distribution of the unified global workspace against the product of its N isolated sub-modules.
  * τ₀ is the characteristic macroscopic integration timescale (default: 10⁻³ s or 1 ms), ensuring strict dimensional homogeneity inside \(\tanh(\cdot)\).
  * \(\exp\big(-\tau_{\text{therm}} / (\tau + \epsilon)\big)\) models Boltzmann-style thermodynamic decoherence (default: \(\tau_{\text{therm}} = 10^{-8}\text{ s}\) or 10 ns), capturing the physical heat and electromagnetic noise generated when driving plasma waveguides toward extreme switching rates.
  * Because \(\mathcal{I}_{\text{Chalmers}}\) acts as a multiplicative prerequisite, if causal integration collapses (\(\mathcal{I}_{\text{Chalmers}} = 0\)), then \(\Psi_C(t) = 0\).

---

#### 2. The Dretskean Representational Grounding Functional
Formalizes Fred Dretske's *teleosemantic representationalism*, measuring normalized mutual information I(⋅;⋅) between internal sensory manifolds \(\mathbf{s}_{\text{sens}}(t) \subset \mathbf{s}(t)\) and external physical environmental states \(\mathbf{e}(t) \in \mathcal{E}\) (\(\mathcal{R}_{\text{Dretske}} \in [0, 1]\)):

\[\mathcal{R}_{\text{Dretske}}\big(\mathbf{s}(t), \mathbf{e}(t)\big) = \frac{I\big(\mathbf{s}_{\text{sens}}(t) \,;\, \mathbf{e}(t - \delta)\big)}{H\big(\mathbf{e}(t - \delta)\big)}\]

* **Physical & Neuroscientific Meaning:** \(H(\mathbf{e})\) is the Shannon entropy of the external stimulus at causal lag δ. If a system manipulates ungrounded symbols in isolation without sensory transduction, \(I(\mathbf{s}_{\text{sens}} ; \mathbf{e}) = 0\), removing the 25% representational contribution.

---

#### 3. The Higher-Order Thought Meta-Cognitive Operator
Formalizes David Rosenthal's *Higher-Order Thought* criterion by requiring a compressed second-order meta-state \(\hat{\mathbf{s}}_2(t) \in \mathbb{R}^{d_2}\) (0 < d₂ < d₁) to actively explain variance (R² > 0) in the first-order state \(\mathbf{s}_1(t) \in \mathbb{R}^{d_1}\), gated by Gaussian reconstruction fidelity (\(\mathcal{H}_{\text{HOT}} \in [0, 1]\)):

\[\mathcal{H}_{\text{HOT}}\big(\mathbf{s}_1, \hat{\mathbf{s}}_2\big) = \max\left(0, \, 1 - \frac{\text{MSE}\big(\mathbf{s}_1, \mathcal{P}^{-1}\hat{\mathbf{s}}_2\big)}{\text{Var}(\mathbf{s}_1)}\right) \cdot \exp\left( - \frac{\text{MSE}\big(\mathbf{s}_1, \mathcal{P}^{-1}\hat{\mathbf{s}}_2\big)}{2\sigma_{\text{meta}}^2} \right) \cdot \mathbb{I}\big(0 < d_2 < d_1\big)\]

* **Physical & Neuroscientific Meaning:** \(\mathcal{P}^{-1} \in \mathbb{R}^{d_1 \times d_2}\) is the projective decoding operator mapping the compressed meta-state back into the first-order space, \(\sigma_{\text{meta}}^2\) is the tolerable representational variance, and \(\mathbb{I}(0 < d_2 < d_1)\) enforces strict projective bottleneck asymmetry. Gating by the coefficient of determination (R²) guarantees that an absent observer (\(\hat{\mathbf{s}}_2 = \mathbf{0}\)) or uncorrelated noise evaluates strictly to 0.0.

---

## 2. Empirical Reference Simulations (`wdrm_model.py`)

### A. Architectural Regime Comparison

Evaluating `wdrm_model.py` across N = 10,000 stochastic timesteps (`seed=42`, τ₀ = 10⁻³ s, \(\tau_{\text{therm}} = 10^{-8}\text{ s}\), \(\sigma_{\text{meta}} = 0.15\)) demonstrates clean separation across four distinct system architectures:

| Architectural Regime | \(\mathcal{I}_{\text{Chalmers}}\) | \(\mathcal{R}_{\text{Dretske}}\) | \(\mathcal{H}_{\text{HOT}}\) | Resulting \(\Psi_C\) | Physical & Cognitive Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Case A: Baseline Cluster** (τ = 10 ms) | `0.0003` | `0.0123` | `0.0000` | **`0.0002`** | High-latency, ungrounded, non-reflective feed-forward cluster. |
| **Case B: Raw Plasma Cluster** (τ = 100 μs) | `1.0000` | `0.0128` | `0.0000` | **`0.5032`** | Optimal causal unity hits the 0.50 dual-aspect ceiling; lacks grounding & HOT. |
| **Case C: Reflexive Automaton** (τ = 0.5 ms) | `0.7820` | `0.6690` | `0.0000` | **`0.5218`** | High sensorimotor tracking ("blindsight"), capped by missing self-monitoring loop. |
| **Case D: Phenomenal Workspace** (τ = 100 μs) | `1.0000` | `0.6690` | `0.8362` | **`0.8763`** | Unified plasma workspace + world-anchored sensors + active meta-cognition. |

> **Empirical Note:** Residual values of \(\mathcal{R}_{\text{Dretske}} \approx 0.012\) in uncoupled Gaussian noise regimes (Cases A and B) reflect standard Miller–Madow finite-sample bias in 32-bin empirical histogram mutual information estimation.

---

### B. Thermodynamic "Goldilocks" Phase Transition

Sweeping interconnect latency τ across 10 orders of magnitude (10⁻¹ s down to 10⁻¹⁰ s) reveals an optimal physical operating window between **τ = 10⁻⁴ s (100 μs) and τ = 10⁻⁶ s (1 μs)**:

| Interconnect Latency (τ) | \(\log_{10}(\tau)\) | Regime Description | Ideal \(\Psi_C\) (No Thermal Cost) | Thermally Constrained \(\Psi_C\) |
| :--- | :---: | :--- | :---: | :---: |
| 1.0 × 10⁻¹ s (100 ms) | `-1.0` | Slow Distributed Network | `0.004731` | `0.004731` |
| 1.0 × 10⁻² s (10 ms) | `-2.0` | Standard Fiber Cluster | `0.047144` | `0.047144` |
| 1.0 × 10⁻³ s (1 ms) | `-3.0` | Critical Inflection Zone (τ = τ₀) | `0.432229` | `0.432225` |
| 1.0 × 10⁻⁴ s (100 μs) | `-4.0` | **Optimal Plasma Peak** | `0.898135` | **`0.898045`** |
| 1.0 × 10⁻⁶ s (1 μs) | `-6.0` | High-Speed Plasma Window | `0.898135` | **`0.889198`** |
| 1.0 × 10⁻⁷ s (100 ns) | `-7.0` | Thermal Roll-Off Onset | `0.898135` | `0.812711` |
| 1.0 × 10⁻⁸ s (10 ns) | `-8.0` | Thermal Breakdown (\(\tau = \tau_{\text{therm}}\)) | `0.898135` | `0.330456` |
| 1.0 × 10⁻⁹ s (1 ns) | `-9.0` | Thermal Decoherence Collapse | `0.898135` | `0.000041` |

---

## 3. Repository Structure & Quickstart

```text
.
├── README.md          # Mathematical specification and empirical benchmarks
└── wdrm_model.py      # Reference implementation, invariant tests, and phase sweep
```

### Installation & Execution

1. Ensure Python 3.10+ is installed with `numpy` and `scipy`:
   ```bash
   pip install numpy scipy
   ```
2. Run the reference simulation and thermodynamic phase-transition sweep:
   ```bash
   python wdrm_model.py
   ```
