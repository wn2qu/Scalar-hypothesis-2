"""
Weighted Dual-Aspect Representational-Meta-Cognitive (WDRM) Reference Model
===========================================================================
A dimensionally consistent, information-theoretic implementation of the WDRM
scalar consciousness functional (Psi_C), evaluating causal state-transition
coupling, teleosemantic mutual information, projective meta-representation,
and interconnect latency phase transitions.

Dependencies: numpy, scipy
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.special import rel_entr


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
        """
        if latency_tau < 0:
            raise ValueError("Propagation latency tau cannot be negative.")

        p_joint = np.asarray(joint_transition_probs, dtype=np.float64)
        p_joint /= np.sum(p_joint)

        p_indep = np.array([1.0], dtype=np.float64)
        for marginal in marginal_transition_probs:
            m = np.asarray(marginal, dtype=np.float64)
            m /= np.sum(m)
            p_indep = np.kron(p_indep, m)

        if p_joint.shape != p_indep.shape:
            raise ValueError("Joint distribution shape must match tensor product of marginals.")

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
        """
        if sensory_states.shape != env_states.shape or sensory_states.ndim != 1:
            raise ValueError("sensory_states and env_states must be 1D arrays of equal length.")

        c_xy, _, _ = np.histogram2d(sensory_states, env_states, bins=bins)
        p_xy = c_xy / np.sum(c_xy)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        nz_y = p_y > 0
        h_env = -float(np.sum(p_y[nz_y] * np.log2(p_y[nz_y])))
        if h_env <= 1e-12:
            return 0.0

        p_x_p_y = np.outer(p_x, p_y)
        nz_xy = (p_xy > 0) & (p_x_p_y > 0)
        mi_bits = float(np.sum(p_xy[nz_xy] * np.log2(p_xy[nz_xy] / p_x_p_y[nz_xy])))

        return float(np.clip(mi_bits / h_env, 0.0, 1.0))

    def compute_hot_operator(
        self,
        first_order_state: np.ndarray,
        second_order_state: np.ndarray,
        decoder_matrix: np.ndarray,
    ) -> float:
        """
        Computes H_HOT enforcing:
          1. Strict projective bottleneck: 0 < dim(s_2) < dim(s_1)
          2. Active meta-state norm: ||s_2||_2 > 0
          3. Positive variance explained (R^2 > 0) gated by Gaussian reconstruction fidelity.
        """
        if first_order_state.ndim != 1 or second_order_state.ndim != 1:
            raise ValueError("States must be 1D vectors.")

        d1 = int(first_order_state.shape[0])
        d2 = int(second_order_state.shape[0])

        if decoder_matrix.shape != (d1, d2):
            raise ValueError(f"decoder_matrix must have shape ({d1}, {d2}).")

        if d2 >= d1 or d2 == 0:
            return 0.0

        if np.linalg.norm(second_order_state) <= 1e-12:
            return 0.0

        reconstructed_s1 = decoder_matrix @ second_order_state
        mse = float(np.mean((first_order_state - reconstructed_s1) ** 2))
        var_s1 = float(np.var(first_order_state))

        if var_s1 <= 1e-12:
            return 0.0

        r_squared = 1.0 - (mse / var_s1)
        if r_squared <= 0.0:
            return 0.0

        gaussian_fidelity = float(np.exp(-mse / (2.0 * (self.params.sigma_meta ** 2))))
        return float(np.clip(r_squared * gaussian_fidelity, 0.0, 1.0))

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

    def latency_phase_sweep(
        self,
        joint_transition_probs: np.ndarray,
        marginal_transition_probs: list[np.ndarray],
        r_dretske: float,
        h_hot: float,
        tau_exponents: np.ndarray,
    ) -> list[dict[str, float]]:
        """
        Sweeps interconnect propagation latency tau = 10**exp across orders of
        magnitude to map the causal saturation phase transition of Psi_C.
        """
        sweep_results = []
        for exp in tau_exponents:
            tau = float(10.0 ** exp)
            i_val = self.compute_chalmers_invariant(
                joint_transition_probs, marginal_transition_probs, latency_tau=tau
            )
            psi_val, _ = self.evaluate(i_val, r_dretske, h_hot)
            sweep_results.append({
                "log10_tau": round(float(exp), 2),
                "tau_sec": tau,
                "I_Chalmers": round(i_val, 6),
                "Psi_C": round(psi_val, 6),
            })
        return sweep_results


def run_simulation_and_tests() -> None:
    """Runs boundary regime simulations, invariant checks, and latency phase sweep."""
    rng = np.random.default_rng(seed=42)
    model = WDRMFunctional()

    env_signal = rng.normal(loc=0.0, scale=1.0, size=10_000)
    s1_state = rng.normal(0.0, 0.5, size=64)
    basis, _ = np.linalg.qr(rng.normal(size=(64, 8)))
    s2_meta_null = np.zeros(8)

    p_m1 = np.array([0.5, 0.5])
    p_m2 = np.array([0.5, 0.5])
    p_weak_joint = np.array([0.27, 0.23, 0.23, 0.27])
    p_strong_joint = np.array([0.48, 0.02, 0.02, 0.48])

    # Case A: Standard Distributed Cluster
    i_a = model.compute_chalmers_invariant(p_weak_joint, [p_m1, p_m2], latency_tau=1e-2)
    r_a = model.compute_dretske_grounding(rng.normal(size=10_000), env_signal)
    h_a = model.compute_hot_operator(s1_state, s2_meta_null, basis)
    _, res_a = model.evaluate(i_a, r_a, h_a)

    # Case B: Zero-Latency Plasma Super-Cluster (Coupled, ungrounded, no HOT)
    i_b = model.compute_chalmers_invariant(p_strong_joint, [p_m1, p_m2], latency_tau=1e-9)
    r_b = model.compute_dretske_grounding(rng.normal(size=10_000), env_signal)
    h_b = model.compute_hot_operator(s1_state, s2_meta_null, basis)
    _, res_b = model.evaluate(i_b, r_b, h_b)

    # Case C: Embodied Reflexive Automaton (Coupled, grounded, no HOT)
    coupled_sensors = env_signal + rng.normal(0.0, 0.12, size=10_000)
    i_c = model.compute_chalmers_invariant(p_strong_joint, [p_m1, p_m2], latency_tau=5e-4)
    r_c = model.compute_dretske_grounding(coupled_sensors, env_signal)
    h_c = model.compute_hot_operator(s1_state, s2_meta_null, basis)
    _, res_c = model.evaluate(i_c, r_c, h_c)

    # Case D: Full Phenomenal Architecture (Plasma workspace + Grounded + HOT)
    structured_s1 = basis @ rng.normal(0.0, 0.5, size=8) + rng.normal(0.0, 0.05, size=64)
    s2_meta_structured = basis.T @ structured_s1
    i_d = model.compute_chalmers_invariant(p_strong_joint, [p_m1, p_m2], latency_tau=1e-9)
    r_d = model.compute_dretske_grounding(coupled_sensors, env_signal)
    h_d = model.compute_hot_operator(structured_s1, s2_meta_structured, basis)
    _, res_d = model.evaluate(i_d, r_d, h_d)

    # Formal invariant assertions
    assert res_a["H_HOT"] == 0.0, "Null observer must yield exact 0.0"
    assert res_a["Psi_C"] < 0.01, "Case A failed lower bound check"
    assert np.isclose(res_b["Psi_C"], 0.50, atol=0.01), "Case B failed 0.50 ceiling check"
    assert res_c["Psi_C"] > res_b["Psi_C"], "Case C failed monotonic progression"
    assert res_d["Psi_C"] > 0.85, "Case D failed phenomenal threshold check"
    assert model.compute_hot_operator(s1_state, s1_state, np.eye(64)) == 0.0, "Identity anti-gaming failed"
    assert model.compute_hot_operator(s1_state, rng.normal(size=8), basis) == 0.0, "Random noise HOT must yield 0.0"

