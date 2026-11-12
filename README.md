# WDRM 2.0 — Weighted Dual-Aspect Representational-Meta-Cognitive Model

**An exploratory reference model combining causal transfer, predictive sensorimotor closure, and held-out error monitoring.** Version 2 replaces the additive floor with three necessary computational components.

These are operational proxies. The model does not establish a consciousness threshold, validate a philosophical theory, or classify dreaming, plants, AI systems, or other organisms as conscious.

## Install and run

Python 3.10+ and NumPy are required. SciPy optionally accelerates nearest-neighbor estimation.

```bash
python -m pip install -r requirements.txt
python wdrm.py
python -m unittest -v
```

Optional tree backend:

```bash
python -m pip install -r requirements-optional.txt
```

| File | Purpose |
| --- | --- |
| `wdrm.py` | WDRM 2.0 implementation and seeded synthetic demonstration |
| `test_wdrm.py` | v2 numerical and counterexample tests |
| `requirements.txt` | Required dependency |
| `requirements-optional.txt` | Dependencies including the optional SciPy accelerator |
| `wdrm_v1.py` / `README_v1.md` | Preserved previous implementation and documentation |
| `test_v1.py` | Preserved regression suite, importing `wdrm_v1` |

## 1. Master functional: every pillar is necessary

```math
\Psi_C =
\left(I_{\mathrm{int}}\right)^{w_1}
\left(R_{\mathrm{ground}}\right)^{w_2}
\left(H_{\mathrm{meta}}\right)^{w_3},
\qquad
w_1,w_2,w_3>0,
\qquad
w_1+w_2+w_3=1.
```

This is a **weighted geometric mean**, not a harmonic mean. Defaults retain v1's relative emphasis: $w_1=0.50$, $w_2=0.25$, and $w_3=0.25$.

```math
0\leq I_{\mathrm{int}},R_{\mathrm{ground}},H_{\mathrm{meta}}\leq1
\quad\Longrightarrow\quad
0\leq\Psi_C\leq1.
```

Any exactly zero component gives exactly zero composite score. There is no additive baseline. Positive values are combined in log space:

```math
\ln\Psi_C =
w_1\ln I_{\mathrm{int}}+
w_2\ln R_{\mathrm{ground}}+
w_3\ln H_{\mathrm{meta}}.
```

Weights must be strictly positive and sum to one within absolute tolerance `1e-12`. Component inputs must be finite scalars in the unit interval. Small positive values are not hard zeros; the geometric mean can exceed its weakest component.

## 2. Corrections to the proposed v2 specification

The reviewer proposed useful experimental requirements alongside several formulas or guarantees that do not follow mathematically. This implementation makes the corrections explicit.

| Proposed claim or formula | Implemented treatment |
| --- | --- |
| Sum of two information transfers divided by the smaller target entropy is bounded by one | A deterministic two-bit swap already gives two. Normalize each direction separately and average. |
| A further unspecified factor $K(\mathcal B)$ establishes the bound | No undefined factor is used; the bounded directional normalization defines the cut objective. |
| Continuous MI divided by continuous conditional entropy lies in $[0,1]$ | Differential entropy can be zero or negative and changes under unit rescaling. Continuous data use a bounded information transform; discrete data retain conditional-entropy ratios. |
| Exact MIP has pairwise polynomial cost | Exact search examines $2^{m-1}-1$ cuts. Explicit candidate cuts produce a labeled restricted-search upper bound. |
| MIP eliminates all analyst boundary choices | It searches within supplied units. Units, system boundary, intervention distribution, temporal grain, and representation still matter. |
| Every replay or lookup table has zero causal sensitivity | An unresponsive replay may score zero. A responsive lookup table implementing a causal transition can score positively. |
| A two-standard-deviation surrogate floor always zeros white noise | This is a heuristic threshold with finite-sample false positives and false negatives. |
| Action-conditioned transfer is empowerment | This measures policy-dependent loop closure. Empowerment is channel capacity, requiring optimization over action distributions. |
| Spearman correlation establishes calibration and excludes passive compressors | It measures error-ranking sensitivity. Passive representations may encode difficulty or uncertainty; endogenous use needs separate testing. |

The resulting cut score is a **WDRM-specific causal-transfer proxy**, not an implementation of Barrett–Seth integrated information or IIT 3.0. Those works motivate this investigation but define different measures [1–3].

## 3. Causal integration

### Interventional data contract

Supply aligned `(n_trials, m_units)` arrays and provenance:

- `current`: independently randomized current-state assignments.
- `next_state`: resulting next-state observations at a declared lag.
- `protocol`: a description of interventions, assignment distributions, timing, and controls.
- `independently_randomized=True`: explicit caller attestation.

The intended distribution is:

```math
P_{\mathrm{do}}(x_t,x_{t+1})
=
q(x_t)\,P\!\left(x_{t+1}\mid\operatorname{do}(X_t=x_t)\right),
\qquad
q(x_t)=\prod_jq_j(x_t^{(j)}).
```

The API cannot verify experimental provenance or turn observational recordings into interventions. It rejects missing declarations and returns `caller_attested_not_verified`. Randomization does not remove the need to justify the measurement model and uncontrolled external influences. Observational identification and nonrandomized policies require a separate causal design; this implementation does not silently accept them as interventions.

### Directional transfer and normalization

For a cut $\mathcal B=(A,B)$:

```math
J_{A\rightarrow B}=I_{\mathrm{do}}(A_t;B_{t+1}\mid B_t),
\qquad
J_{B\rightarrow A}=I_{\mathrm{do}}(B_t;A_{t+1}\mid A_t).
```

Conditioning on the recipient's current state is an explicit refinement of the reviewer's expression. It separates cross-cut predictive influence from recipient persistence and can retain conditional interactions such as XOR.

Let $J^\star$ denote the nonnegative, optionally surrogate-thresholded estimate. The same normalization operator is used for integration and grounding.

For **discrete data**:

```math
\mathcal N_{\mathrm{disc}}(X;Y\mid Z)=
\begin{cases}
\operatorname{clip}\!\left(
\dfrac{J^\star(X;Y\mid Z)}{\widehat H(Y\mid Z)},0,1
\right), & \widehat H(Y\mid Z)>10^{-12},\\
0, & \text{otherwise}.
\end{cases}
```

Discrete joint entropies count observed tuples without allocating a dense state table. This plug-in estimate remains sample-limited. Use `kind="discrete"` for categorical/discrete variables encoded numerically, not continuous values treated as unique categories.

For **continuous data**:

```math
\mathcal N_{\mathrm{cont}}(X;Y\mid Z)
=
1-\exp\!\left(-\frac{2J^\star(X;Y\mid Z)}{d_Y}\right),
```

where $d_Y$ counts nonconstant target coordinates, with a minimum of one in code. This is a bounded monotone transform, not an entropy ratio. Per-coordinate normalization reduces extensive growth but is not invariant to redundant coordinates, representation choices, or changes of dimensionality. Discrete and continuous scores are not interchangeable calibrations.

The cut score and minimum are:

```math
\phi(\mathcal B)=
\frac{
\mathcal N(A_t;B_{t+1}\mid B_t)+
\mathcal N(B_t;A_{t+1}\mid A_t)
}{2},
\qquad
\Phi_{\mathrm{norm}}=\min_{\mathcal B}\phi(\mathcal B)\in[0,1].
```

Both directions are evaluated, but averaging does **not** require both to be nonzero. A perfectly informative one-way cut can score $0.5$. Requiring reciprocal nonzero influence would need a different objective, such as a geometric mean of the directional terms.

### Exact versus restricted partitions

Canonical enumeration fixes unit zero in $A$, excludes the all-unit cut, and includes each unique nontrivial bipartition once.

- Default: exhaustive enumeration up to `max_exact_units=8`.
- Above the limit: raise an error rather than silently approximate.
- With `candidate_partitions`: canonicalize and deduplicate supplied cuts; report `restricted_upper_bound` unless all unique cuts are covered.

The restricted minimum bounds the exhaustive minimum **of the same estimated objective on the same data** from above. It is not a confidence bound on a population quantity. `exact` describes cut coverage, not exact information estimation. Exhaustive search remains exponential, and multivariate k-NN still encounters high-dimensional sparsity [3].

### Intrinsic timescale and relative latency

```math
I_{\mathrm{int}}(\tau)=
\Phi_{\mathrm{norm}}
\tanh\!\left(\frac{\tau_{\mathrm{auto}}}{\tau+\epsilon_\tau}\right)
\exp\!\left(-\kappa\frac{\tau_{\mathrm{auto}}}{\tau+\epsilon_\tau}\right).
```

`estimate_intrinsic_timescale()` uses a **separate, ordered, unperturbed trace**. It centers nonconstant channels, computes biased autocorrelations normalized by zero-lag energy, averages across channels, and interpolates the first $1/e$ crossing. The default maximum lag is half the recording.

Constant traces and recordings with no observed crossing raise an error. No fixed biological or silicon timescale is substituted. Oscillatory or nonstationary data require care: first crossing is an operational convention, not a unique intrinsic timescale.

Time quantities use the same units; seconds are recommended. $\kappa$ is dimensionless. Scaling all times, **including** $\epsilon_\tau$, preserves the response. Otherwise finite regularization slightly breaks exact unit invariance near zero latency.

The short-latency factor remains phenomenological, not a thermodynamic derivation. Writing $r=\tau_{\mathrm{auto}}/(\tau+\epsilon_\tau)$ gives the unconstrained response peak:

```math
r_\ast=\frac12\operatorname{arsinh}\!\left(\frac{2}{\kappa}\right).
```

The accessible maximum may lie at zero latency. `include_penalty=False` provides an ablation. Relative-speed normalization does not establish substrate equivalence or measure consciousness by itself.

## 4. Active sensorimotor grounding

Supply aligned observations of $S_t$, $E_{t+\Delta t}$, $A_t$, and $S_{t+\Delta t}$. The API does not construct lags or choose $\Delta t$.

```math
R_{\mathrm{ground}}=
\mathcal N(S_t;E_{t+\Delta t}\mid A_t)
\cdot
\mathcal N(A_t;S_{t+\Delta t}\mid S_t).
```

The factors measure predictive relevance and action-conditioned sensorimotor closure.

Missing actions (`None`) or actions constant in every coordinate return exactly zero. Varying actions require `actions_randomized=True` and an `action_protocol`. A naturally observed action stream does not become causal merely by conditioning on $S_t$.

A passive transducer with no action channel therefore scores zero. A thermostat with an actuator can have closure; it is not universally excluded. Predictive filtering may improve relevance, but no guarantee eliminates noise, sampling error, or consequences of discarded predictive information.

**Offline states:** `mode="internal"` applies the same calculation to an identified internal generative environment and internal interventions/actions. The label gives no score bonus. Observable predictive loop closure remains necessary. This permits studying offline dynamics without establishing biological dreaming or guaranteeing zero for every possible oscillator model.

This is policy-dependent action transfer, not optimized empowerment channel capacity [6].

## 5. k-NN estimation and surrogate controls

Continuous inputs undergo average-rank marginal copula transforms; constant coordinates are removed. The implementation uses strict Chebyshev neighborhoods, KSG-1 MI, and the Frenzel–Pompe conditional extension [4,5]:

```math
\widehat I(X;Y\mid Z)=
\psi(k)+\frac1N\sum_i
\left[
\psi(n_z(i)+1)-
\psi(n_{xz}(i)+1)-
\psi(n_{yz}(i)+1)
\right].
```

Counts exclude each sample itself. A Theiler window can exclude temporally adjacent rows; empty-conditioning counts adapt to eligible neighbors. Rank transforms improve marginal scale robustness, but ties and dimensionality still matter. Zero-radius neighborhoods raise an error; discrete data are not silently jittered.

The requested correction is:

```math
J^\star=
\max\!\left(0,\widehat I-\mu_{\mathrm{shift}}
-z_{\mathrm{shift}}\sigma_{\mathrm{shift}}\right),
\qquad z_{\mathrm{shift}}=2\ \text{by default}.
```

The source block $X$ is circularly shifted against fixed $(Y,Z)$. Distinct offsets are selected reproducibly from `min_shift` through `N-min_shift`. Diagnostics include the raw estimate, null mean, sample standard deviation, offsets, corrected amount, kind, and backend. `correct=False` returns the nonnegative raw estimate and records that correction was disabled.

Circular shifts approximately preserve within-block serial structure, assuming suitable stationarity and lag separation. For CMI they may also disrupt the $X$–$Z$ relationship. This is a **heuristic baseline**, not an exact conditional-randomization test, a guaranteed zero under independence, or a calibrated p-value. Use stronger conditional surrogates or repeated independent interventions when needed. Cut selection, hyperparameter tuning, periodicity, and nonstationarity require further uncertainty analysis.

### Computational cost

| Operation | Cost / constraint |
| --- | --- |
| SciPy k-d tree | Often efficient at low dimension; no universal $O(N\log N)$ query guarantee in high dimension |
| NumPy fallback | Chunked $O(N^2d)$ time; $O(bN)$ distance workspace, $b=128$; capped at 4,000 rows |
| Exact MIP | $2^{m-1}-1$ cuts, two directional estimates per cut, multiplied by surrogate work |
| Discrete estimator | Counts observed tuples; avoids dense $k^m$ allocation, not sample-complexity limits |
| RBF kernel ridge | Dense $O(n^2)$ training storage and generally $O(n^3)$ solve; default 2,000 training-row limit |

No pairwise approximation is presented as an exact bipartition calculation. Pairwise estimates can miss synergy.

## 6. Metacognitive self-monitoring proxy

```math
H_{\mathrm{meta}}=F_{\mathrm{abs}}F_{\mathrm{cal}}.
```

### Nonlinear abstraction and measurement floor

An RBF kernel ridge probe predicts first-order targets:

```math
\widehat S_1=g_\theta(S_2).
```

Input/output scaling uses training data only. The kernel diagonal gets `regularization * n_train`. Kernel width defaults to $1/d_2$ in standardized code coordinates and can be specified explicitly.

For held-out rows:

```math
V=\frac{1}{nd_1}\sum_{i,j}(S_{1,ij}-\overline S_{1,\cdot j})^2,
\qquad
\operatorname{NMSE}=\frac{\operatorname{MSE}}{V},
\qquad R^2=1-\operatorname{NMSE}.
```

```math
F_{\mathrm{abs}}=
\mathbf 1[0<d_2<d_1]\,
\mathbf 1[V>\eta_{\mathrm{snr}}\sigma_{\mathrm{noise}}^2]\,
\max(0,R^2)
\exp\!\left(-\frac{\operatorname{NMSE}}{2\sigma_{\mathrm{meta}}^2}\right).
```

`noise_variance` is **required and strictly positive**, in squared target units. It represents independently characterized pooled per-coordinate measurement variance; there is no default sensor noise. Choosing an arbitrarily small floor can defeat SNR protection. Reconcile mixed feature units and transform measurement variance consistently with any target normalization.

### Error sensitivity on a third split

```math
e_i=\left\|S_{1,i}-g_\theta(S_{2,i})\right\|_2^2,
\qquad
\widehat e_i=\max(0,h_\phi(S_{2,i})),
\qquad
F_{\mathrm{cal}}=
\max\!\left(0,\rho_{\mathrm{Spearman}}(\widehat e,e)\right).
```

The second RBF probe $h_\phi$ predicts error from the code. Use three separate splits:

1. Fit $g_\theta$ on the **training split**.
2. Freeze it, compute errors on the **error-probe calibration split**, and fit $h_\phi$ there.
3. Freeze both probes and evaluate abstraction and error sensitivity on the **test split**.

`fit_meta_probe()` implements stages 1–2; `compute_metacognition()` implements stage 3. Each probe learns its scaling within its own training split. The API cannot detect overlapping recordings or prevent caller leakage. For time series, use nonoverlapping blocked splits with temporal gaps; tune hyperparameters using additional training-side validation.

Constant errors or constant error predictions return zero correlation. A relative floating-point tolerance handles numerical jitter around constants; the separate measurement SNR gate handles near-constant targets.

The requested symbol $F_{\mathrm{cal}}$ denotes **rank error sensitivity**, not absolute calibration, AUROC2, meta-$d'$, or demonstrated endogenous self-monitoring [7]. A passive code can encode difficulty or uncertainty that an external probe can exploit. PCA, shadows, and bottlenecks are not universally guaranteed to fail. Interventions on an endogenous reliability signal and evidence of its downstream control role would provide stronger support.

## 7. API examples

### Geometric gate and discrete interventions

```python
import numpy as np
from wdrm import WDRMFunctional, WDRMParameters, InterventionalData

model = WDRMFunctional(WDRMParameters())
assert model.evaluate(1.0, 0.0, 1.0)[0] == 0.0

current = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
data = InterventionalData(
    current=current,
    next_state=current[:, ::-1],
    protocol="Independent fair-bit resets; subsequent units exchange values.",
    independently_randomized=True,
)
integration, details = model.compute_causal_integration(
    data,
    latency_tau=0.1,
    tau_auto=1.0,  # Illustrative: estimate from a separate trace in a real study.
    kind="discrete",
    correct=False,  # Full toy distribution, not an estimated continuous density.
)
print(details["Phi_norm"], integration, details["search"])
```

### Intrinsic timescale, grounding, and held-out metacognition

```python
import numpy as np
from wdrm import WDRMFunctional, WDRMParameters

rng = np.random.default_rng(42)
model = WDRMFunctional(
    WDRMParameters(n_surrogates=7, min_shift=10, sigma_meta=0.5)
)

trace = np.zeros((2000, 2))
for t in range(1, len(trace)):
    trace[t] = 0.8 * trace[t - 1] + rng.normal(size=2)
tau_auto, acf_details = model.estimate_intrinsic_timescale(trace, sample_interval=0.01)

s, action = rng.normal(size=(2, 350))
future_environment = s + 0.3 * action + rng.normal(scale=0.3, size=len(s))
future_sensory = 0.5 * s + action + rng.normal(scale=0.3, size=len(s))
grounding, ground_details = model.compute_grounding(
    s, future_environment, action, future_sensory,
    action_protocol="Synthetic actions independently randomized across trials.",
    actions_randomized=True,
    mode="external",
)

z, uncertainty = rng.uniform(-2, 2, 900), rng.uniform(-1, 1, 900)
code = np.column_stack((z, uncertainty))
target = np.column_stack((
    np.sin(z), np.cos(z), z, z*z, np.sin(2*z), np.cos(2*z)
))
target += rng.normal(size=target.shape) * (0.02 + 0.05 * (uncertainty[:, None] + 1))

probe = model.fit_meta_probe(
    target[:300], code[:300],
    target[300:600], code[300:600],
    regularization=1e-5,
)
meta, meta_details = model.compute_metacognition(
    target[600:], code[600:], probe, noise_variance=1e-4
)
print(tau_auto, grounding, meta, meta_details["F_cal"])
```

These examples use separate synthetic datasets. Combining their numbers does not make them measurements of a single system. Empirical studies need one defined system, compatible windows and lags, and a justified protocol spanning all pillars.

## 8. Defaults and migration

| Parameter | Default | Meaning |
| --- | --- | --- |
| `w1, w2, w3` | `0.50, 0.25, 0.25` | Strictly positive geometric weights |
| `kappa` | `1e-3` | Dimensionless short-latency shape |
| `epsilon_tau` | `1e-12` | Time regularizer; seconds if all times are seconds |
| `sigma_meta` | `0.15` | Relative reconstruction tolerance |
| `eta_snr` | `3.0` | Variance-to-noise threshold |
| `knn_k` | `5` | Neighbor count |
| `n_surrogates` | `19` | Distinct circular shifts |
| `surrogate_z` | `2.0` | Null standard-deviation multiplier |
| `min_shift` | `20` | Minimum offset, in rows |
| `theiler_window` | `0` | Temporal neighbor exclusion, in rows |
| `max_exact_units` | `8` | Exhaustive-search unit limit |
| `seed` | `42` | Surrogate RNG seed |

`min_shift` must exceed the Theiler window; sample count must support the requested neighbors and offsets. `tau_auto` and `noise_variance` are supplied or measured for each system.

| v1 concern | v2 implementation | Remaining limitation |
| --- | --- | --- |
| Additive floor | Geometric gate | Small positive estimation artifacts still propagate |
| Observational correlation | Intervention-declared conditional transfer | Causal provenance needs experimental support |
| Arbitrary selected cut | Canonical exhaustive search for small systems | Boundaries remain chosen; exhaustive cost is exponential |
| Extensive information growth | Bounded directional normalization | Representation and dimension still affect scaling |
| Fixed integration time | Measured autocorrelation timescale | Recording and ACF convention matter |
| Passive correlation | Future prediction multiplied by active closure | Controllers can pass; this is not channel capacity |
| Histogram artifacts | Rank-copula k-NN and shift controls | Bias, dimensionality, ties, and null validity remain |
| Linear reconstruction alone | Nonlinear abstraction and error sensitivity | External decodability does not establish endogenous monitoring |
| Constant-target jitter | Required measurement-based SNR floor | Noise floor must be independently justified |

This is a breaking v2 API:

- `alpha/beta/gamma` become `w1/w2/w3`, changing from additive weights to exponents.
- `tau_0/tau_short` become per-system `tau_auto` and dimensionless `kappa`.
- `compute_causal_integration()` replaces flattened-joint dependence scoring.
- `compute_grounding()` requires actions and future-state alignment.
- `fit_meta_probe()` / `compute_metacognition()` replace linear reconstruction.
- `evaluate()` accepts `i_int, r_ground, h_meta` and returns v2 diagnostic keys.
- `latency_phase_sweep()` takes estimated `phi_norm` and does not repeat the MIP search for every latency.

Import `wdrm_v1` explicitly to reproduce the previous API. There are no aliases that silently substitute v2 semantics.

## 9. Validation

Validated on Python 3.13.3 and NumPy 2.2.5: **50 tests passed; one optional SciPy parity test skipped because SciPy was unavailable.** This includes 20 preserved v1 tests and 30 executed v2 tests.

Coverage includes necessary-condition gates, intervention declarations, independent identity dynamics, disconnected units, swaps, responsive lookup tables, XOR conditional information, exact/restricted cuts, AR(1) timescales, unit rescaling, Gaussian MI, conditional independence, reproducible surrogates, Theiler exclusion, passive/active/internal grounding, nonlinear held-out decoding, error sensitivity, constant predictions, SNR gating, and input immutability.

The optional accelerated path was not executed here. Tests validate implementation behavior, not construct validity or universal counterexample exclusion.

## References and GitHub rendering

1. Barrett & Seth (2011), [Practical Measures of Integrated Information for Time-Series Data](https://doi.org/10.1371/journal.pcbi.1001052).
2. Oizumi, Albantakis & Tononi (2014), [From the Phenomenology to the Mechanisms of Consciousness: Integrated Information Theory 3.0](https://doi.org/10.1371/journal.pcbi.1003588).
3. Kitazono, Kanai & Oizumi (2018), [Efficient Algorithms for Searching the Minimum Information Partition in Integrated Information Theory](https://doi.org/10.3390/e20030173).
4. Kraskov, Stögbauer & Grassberger (2004), [Estimating Mutual Information](https://doi.org/10.1103/PhysRevE.69.066138).
5. Frenzel & Pompe (2007), [Partial Mutual Information for Coupling Analysis of Multivariate Time Series](https://doi.org/10.1103/PhysRevLett.99.204101). See also the [conditional-estimator documentation](https://carlson.pages.ifisc.uib-csic.es/infomeasure/guide/cond_mi/ksg_cond_mi/).
6. Klyubin, Polani & Nehaniv (2005), [Empowerment: A Universal Agent-Centric Measure of Control](https://doi.org/10.1109/CEC.2005.1554676).
7. Fleming & Lau (2014), [How to Measure Metacognition](https://doi.org/10.3389/fnhum.2014.00443).

Equations use inline dollar delimiters and fenced `math` blocks supported by [GitHub's Markdown math renderer](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions). Keep the file as Markdown without an outer code fence. A live GitHub render has not been performed.

