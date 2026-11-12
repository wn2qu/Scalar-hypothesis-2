### Draft

"""
Weighted Dual-Aspect Representational-Meta-Cognitive (WDRM) Reference Model
===========================================================================
A dimensionally consistent, information-theoretic implementation of the WDRM
scalar consciousness functional (\Psi_C), evaluating causal state-transition
coupling, teleosemantic mutual information, and projective meta-representation.

Dependencies: numpy, scipy
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.special import rel_entr
from scipy.stats import differential_entropy


@dataclass(frozen=True)
class WDRMParameters:
    """Normalized structural weights and physical scaling constants."""
    alpha: float = 0.50          # Chalmers dual-aspect structural weight
    beta: float = 0.25           # Dretske representational weight
    gamma: float = 0.25          # Rosenthal Higher-Order Thought weight
    tau_0: float = 1e-3          # Characteristic integration timescale (seconds)
    epsilon: float = 1e-12       # Numerical regularization floor (seconds)
    sigma_meta: float = 0.15     # Tolerable meta-representational reconstruction std

    def __post_init__(self) -> None:
        if not np.isclose(self.alpha + self.beta + self.gamma, 1.0, atol=1e-6):
            raise ValueError("Structural weights (alpha, beta, gamma) must sum to 1.0.")
        if self.tau_0 <= 0 or self.sigma_meta <= 0:
            raise ValueError("Timescale tau_0 and variance sigma_meta must be positive.")


class WDRMFunctional:
    """Computes the scalar phenomenal consciousness index Psi_C from system dynamics."""

    def __init__(self, params: WDRMParameters = WDRMParameters()) -> None:
        self.params = params

    def compute_chalmers_invariant(
        self,
        joint_transition_probs: np.ndarray,
        marginal_transition_probs: list[np.ndarray],
        latency_tau: float,
    ) -> float:
        """
        Computes I_Chalmers using dimensionally normalized Kullback-Leibler divergence:
            I_Chalmers = tanh( (tau_0 / (tau + epsilon)) * D_KL(P_joint || prod_i P_i) )

        Args:
            joint_transition_probs: Flattened probability distribution P(s_{t+1} | s_t).
            marginal_transition_probs: List of marginal distributions P(s_{i,t+1} | s_{i,t}).
            latency_tau: Effective physical interconnect propagation delay (seconds).
        """
        p_joint = np.asarray(joint_transition_probs, dtype=np.float64)
        p_joint /= np.sum(p_joint)

        # Construct product of independent module transition distributions
        p_indep = np.array([1.0], dtype=np.float64)
        for marginal in marginal_transition_probs:
            m = np.asarray(marginal, dtype=np.float64)
            m /= np.sum(m)
            p_indep = np.kron(p_indep, m)

        if p_joint.shape != p_indep.shape:
            raise ValueError("Joint distribution shape must match tensor product of marginals.")

        # Smooth distributions to guarantee absolute continuity (P_joint << P_indep)
        eps_prob = 1e-12
        p_joint = np.clip(p_joint, eps_prob, 1.0)
        p_indep = np.clip(p_indep, eps_prob, 1.0)
        p_joint /= np.sum(p_joint)
        p_indep /= np.sum(p_indep)

        d_kl_nats = float(np.sum(rel_entr(p_joint, p_indep)))
        dimensionless_gain = self.params.tau_0 / (latency_tau + self.params.epsilon)
        return float(np.tanh(dimensionless_gain * d_kl_nats))

    def compute_dretske_grounding(
        self,
        sensory_states: np.ndarray,
        env_states: np.ndarray,
        bins: int = 32,
    ) -> float:
        """
        Computes R_Dretske = I(S_sens ; E) / H(E) via empirical histogram density estimation.

        Args:
            sensory_states: 1D array of internal sensory manifold activations at time t.
            env_states: 1D array of external physical stimulus states at lag (t - delta).
            bins: Number of discretization bins for mutual information estimation.
        """
        c_xy, _, _ = np.histogram2d(sensory_states, env_states, bins=bins)
        p_xy = c_xy / np.sum(c_xy)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        # Entropy H(E)
        nz_y = p_y > 0
        h_env = -float(np.sum(p_y[nz_y] * np.log2(p_y[nz_y])))
        if h_env <= 1e-12:
            return 0.0

        # Mutual Information I(S_sens ; E)
        p_x_p_y = np.outer(p_x, p_y)
        nz_xy = (p_xy > 0) & (p_x_p_y > 0)
        mi_bits = float(np.sum(p_xy[nz_xy] * np.log2(p_xy[nz_xy] / p_x_p_y[nz_xy])))

        return float(np.clip(mi_bits / h_env, 0.0, 1.0))

    def compute_hot_operator(
        self,
        first_order_state: np.ndarray,
        second_order_state: np.ndarray,
        projection_matrix: np.ndarray,
    ) -> float:
        """
        Computes H_HOT enforcing strict projective dimensionality reduction
        (dim(s_2) < dim(s_1)) and Gaussian reconstruction fidelity.

        Args:
            first_order_state: Vector s_1 in R^d1.
            second_order_state: Compressed meta-state s_2 in R^d2 (where d2 < d1).
            projection_matrix: Linear decoding operator P^{-1} mapping R^d2 -> R^d1.
        """
        d1 = first_order_state.shape[0]
        d2 = second_order_state.shape[0]

        # Enforce strict representational hierarchy (prevents trivial self-identity s_2 = s_1)
        if d2 >= d1 or d2 == 0:
            return 0.0

        reconstructed_s1 = projection_matrix @ second_order_state
        mse = float(np.mean((first_order_state - reconstructed_s1) ** 2))
        fidelity = np.exp(-mse / (2.0 * (self.params.sigma_meta ** 2)))
        return float(np.clip(fidelity, 0.0, 1.0))

    def evaluate(
        self,
        i_chalmers: float,
        r_dretske: float,
        h_hot: float,
    ) -> Tuple[float, dict[str, float]]:
        """Evaluates master scalar functional Psi_C(t)."""
        weighted_bracket = (
            self.params.alpha
            + self.params.beta * r_dretske
            + self.params.gamma * h_hot
        )
        psi_c = float(i_chalmers * weighted_bracket)
        breakdown = {
            "I_Chalmers": round(i_chalmers, 4),
            "R_Dretske": round(r_dretske, 4),
            "H_HOT": round(h_hot, 4),
            "Psi_C": round(psi_c, 4),
        }
        return psi_c, breakdown


# ==============================================================================
# Empirical Regime Simulations
# ==============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    model = WDRMFunctional()

    # Shared simulated environmental time-series (10,000 timesteps)
    env_signal = rng.normal(loc=0.0, scale=1.0, size=10_000)

    # First-order cortical/cluster state vector (64-dimensional) and projective decoder
    s1_state = rng.normal(0.0, 0.5, size=64)
    # Create an 8-dimensional higher-order bottleneck subspace (rank 8 < 64)
    basis, _ = np.linalg.qr(rng.normal(size=(64, 8)))
    s2_meta_accurate = basis.T @ s1_state
    s2_meta_null = np.zeros(8)

    # 1. Case A: Standard Distributed Cluster (High latency, ungrounded, no HOT)
    p_m1 = np.array([0.5, 0.5])
    p_m2 = np.array([0.5, 0.5])
    p_weak_joint = np.array([0.27, 0.23, 0.23, 0.27])
    i_a = model.compute_chalmers_invariant(p_weak_joint, [p_m1, p_m2], latency_tau=1e-2)
    r_a = model.compute_dretske_grounding(rng.normal(size=10_000), env_signal)
    h_a = model.compute_hot_operator(s1_state, s2_meta_null, basis)
    _, res_a = model.evaluate(i_a, r_a, h_a)

    # 2. Case B: Zero-Latency Plasma Super-Cluster (Coupled, ungrounded, no HOT)
    p_strong_joint = np.array([0.48, 0.02, 0.02, 0.48])
    i_b = model.compute_chalmers_invariant(p_strong_joint, [p_m1, p_m2], latency_tau=1e-9)
    r_b = model.compute_dretske_grounding(rng.normal(size=10_000), env_signal)
    h_b = model.compute_hot_operator(s1_state, s2_meta_null, basis)
    _, res_b = model.evaluate(i_b, r_b, h_b)

    # 3. Case C: Embodied Reflexive Automaton (Coupled, grounded, no HOT)
    coupled_sensors = env_signal + rng.normal(0.0, 0.12, size=10_000)
    i_c = model.compute_chalmers_invariant(p_strong_joint, [p_m1, p_m2], latency_tau=5e-4)
    r_c = model.compute_dretske_grounding(coupled_sensors, env_signal)
    h_c = model.compute_hot_operator(s1_state, s2_meta_null, basis)
    _, res_c = model.evaluate(i_c, r_c, h_c)

    # 4. Case D: Full Phenomenal Architecture (Plasma workspace + Grounded + HOT)
    # Construct low-rank structured s1_state so 8D meta-state captures core variance
    structured_s1 = basis @ rng.normal(0.0, 0.5, size=8) + rng.normal(0.0, 0.05, size=64)
    s2_meta_structured = basis.T @ structured_s1
    i_d = model.compute_chalmers_invariant(p_strong_joint, [p_m1, p_m2], latency_tau=1e-9)
    r_d = model.compute_dretske_grounding(coupled_sensors, env_signal)
    h_d = model.compute_hot_operator(structured_s1, s2_meta_structured, basis)
    _, res_d = model.evaluate(i_d, r_d, h_d)

    print("WDRM Empirical Regime Evaluation:")
    print(f"Case A (Baseline Cluster)      : {res_a}")
    print(f"Case B (Raw Plasma Cluster)    : {res_b}")
    print(f"Case C (Reflexive Automaton)   : {res_c}")
    print(f"Case D (Phenomenal Workspace)  : {res_d}")
