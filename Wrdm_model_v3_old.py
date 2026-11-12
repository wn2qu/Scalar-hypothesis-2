"""
Weighted Dual-Aspect Representational-Meta-Cognitive (WDRM) Reference Model
===========================================================================
A dimensionally consistent, information-theoretic implementation of the WDRM
scalar consciousness functional (Psi_C), incorporating causal state-transition
coupling, teleosemantic mutual information, projective meta-representation,
and thermodynamic interconnect dissipation limits.

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
    tau_therm: float = 1e-8      # Characteristic thermal decoherence timescale (seconds)
    epsilon: float = 1e-12       # Numerical regularization floor (seconds)
    sigma_meta: float = 0.15     # Tolerable meta-representational reconstruction std

    def __post_init__(self) -> None:
        if not np.isclose(self.alpha + self.beta + self.gamma, 1.0, atol=1e-6):
            raise ValueError("Structural weights (alpha, beta, gamma) must sum to 1.0.")
        if self.tau_0 <= 0 or self.tau_therm <= 0 or self.sigma_meta <= 0:
            raise ValueError("Timescales and variance must be positive.")


class WDRMFunctional:
    """Computes the scalar phenomenal consciousness index Psi_C from system dynamics."""

    def __init__(self, params: WDRMParameters = WDRMParameters()) -> None:
        self.params = params

    def compute_chalmers_invariant(
        self,
        joint_transition_probs: np.ndarray,
        marginal_transition_probs: list[np.ndarray],
        latency_tau: float,
        include_thermal_dissipation: bool = True,
    ) -> float:
        """
        Computes I_Chalmers with optional thermodynamic decoherence penalty:
            I_Chalmers = tanh((tau_0 / (tau + eps)) * D_KL) * exp(-tau_therm / (tau + eps))
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
        raw_coupling = float(np.tanh(dimensionless_gain * d_kl_nats))

        if not include_thermal_dissipation:
            return raw_coupling

        thermal_survival = float(
            np.exp(-self.params.tau_therm / (latency_tau + self.params.epsilon))
        )
        return raw_coupling * thermal_survival

    def compute_dretske_grounding(
        self,
        sensory_states: np.ndarray,
        env_states: np.ndarray,
        bins: int = 32,
    ) -> float:
        """Computes R_Dretske = I(S_sens ; E) / H(E) via empirical histogram estimation."""
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
        """Computes R^2-gated Higher-Order Thought operator H_HOT."""
        if first_order_state.ndim != 1 or second_order_state.ndim != 1:
            raise ValueError("States must be 1D vectors.")

        d1 = int(first_order_state.shape[0])
        d2 = int(second_order_state.shape[0])

        if decoder_matrix.shape != (d1, d2):
            raise ValueError(f"decoder_matrix must have shape ({d1}, {d2}).")

        if d2 >= d1 or d2 == 0 or np.linalg.norm(second_order_state) <= 1e-12:
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
        include_thermal_dissipation: bool = True,
    ) -> list[dict[str, float]]:
        """Sweeps interconnect latency tau = 10**exp to map the Goldilocks window."""
        sweep_results = []
        for exp in tau_exponents:
            tau = float(10.0 ** exp)
            i_val = self.compute_chalmers_invariant(
                joint_transition_probs,
                marginal_transition_probs,
                latency_tau=tau,
                include_thermal_dissipation=include_thermal_dissipation,
            )
            psi_val, _ = self.evaluate(i_val, r_dretske, h_hot)
            sweep_results.append({
                "log10_tau": round(float(exp), 2),
                "tau_sec": tau,
                "I_Chalmers": round(i_val, 6),
                "Psi_C": round(psi_val, 6),
            })
        return sweep_results


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    model = WDRMFunctional()

    env_signal = rng.normal(loc=0.0, scale=1.0, size=10_000)
    basis, _ = np.linalg.qr(rng.normal(size=(64, 8)))
    coupled_sensors = env_signal + rng.normal(0.0, 0.12, size=10_000)
    structured_s1 = basis @ rng.normal(0.0, 0.5, size=8) + rng.normal(0.0, 0.05, size=64)
    s2_meta_structured = basis.T @ structured_s1

    p_m1 = np.array([0.5, 0.5])
    p_m2 = np.array([0.5, 0.5])
    p_strong_joint = np.array([0.48, 0.02, 0.02, 0.48])

    r_d = model.compute_dretske_grounding(coupled_sensors, env_signal)
    h_d = model.compute_hot_operator(structured_s1, s2_meta_structured, basis)

    exponents = np.linspace(-1.0, -10.0, 19)
    ideal = model.latency_phase_sweep(p_strong_joint, [p_m1, p_m2], r_d, h_d, exponents, False)
    thermal = model.latency_phase_sweep(p_strong_joint, [p_m1, p_m2], r_d, h_d, exponents, True)

    print("Thermodynamic Goldilocks Sweep (Ideal vs. Thermal-Dissipation):")
    for id_row, th_row in zip(ideal, thermal):
        print(
            f"log10(tau)={id_row['log10_tau']:>5} | "
            f"Ideal Psi_C={id_row['Psi_C']:.6f} | Thermal Psi_C={th_row['Psi_C']:.6f}"
        )
