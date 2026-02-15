"""
Safety analysis for implant and body-worn antennas.

Computes:
  - SAR (Specific Absorption Rate) estimates
  - Temperature rise estimates
  - Compliance with FCC/IEEE/ICNIRP limits
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..utils.constants import C_0
from ..utils.materials import Material, MUSCLE, SKIN_DRY


# ─── Regulatory limits ───────────────────────────────────────────────────────

SAR_LIMIT_FCC_1G = 1.6    # W/kg, averaged over 1g tissue (FCC)
SAR_LIMIT_IEEE_1G = 2.0   # W/kg, averaged over 1g (IEEE C95.1)
SAR_LIMIT_ICNIRP_10G = 2.0  # W/kg, averaged over 10g (ICNIRP, EU)
SAR_LIMIT_IMPLANT = 2.0   # W/kg, specific limit for implants


@dataclass
class SARResult:
    """SAR analysis result."""
    peak_sar_1g: float          # W/kg, 1g average
    peak_sar_10g: float         # W/kg, 10g average
    input_power_w: float        # Assumed input power
    compliant_fcc: bool
    compliant_icnirp: bool
    max_safe_power_w: float     # Maximum input power for compliance
    notes: str = ""


@dataclass
class ThermalResult:
    """Thermal analysis result."""
    peak_temp_rise_c: float     # °C above ambient
    time_to_steady_state_s: float
    safe_duration_s: float      # Duration before exceeding limit
    temp_limit_c: float = 1.0   # IEEE limit for implants: ΔT < 1°C
    compliant: bool = True


class SARAnalyzer:
    """SAR and thermal safety analysis for implant/body-worn antennas."""

    def __init__(self):
        pass

    def estimate_sar(
        self,
        input_power_w: float,
        antenna_efficiency: float,
        freq_hz: float,
        tissue: Material = MUSCLE,
        distance_to_tissue_m: float = 0.0,
        antenna_area_m2: float = 1e-4,
    ) -> SARResult:
        """
        Estimate peak SAR from antenna parameters.

        This is an analytical estimate using simplified models.
        For accurate SAR, use full-wave simulation with tissue phantoms.

        Model: SAR ≈ σ·|E|² / (2ρ)
        For a small antenna radiating into tissue:
        |E|² ≈ P_rad / (π·r²·σ_tissue·δ_skin) approximately

        Args:
            input_power_w: Input power to antenna
            antenna_efficiency: Radiation efficiency (0-1)
            freq_hz: Operating frequency
            tissue: Target tissue material
            distance_to_tissue_m: Distance from antenna to tissue surface
            antenna_area_m2: Effective antenna area
        """
        omega = 2 * math.pi * freq_hz
        sigma = tissue.sigma + omega * 8.854e-12 * tissue.eps_r * tissue.tan_d
        rho = tissue.density

        if rho <= 0 or sigma <= 0:
            return SARResult(
                peak_sar_1g=0, peak_sar_10g=0,
                input_power_w=input_power_w,
                compliant_fcc=True, compliant_icnirp=True,
                max_safe_power_w=float('inf'),
                notes="Tissue parameters invalid (no loss)",
            )

        # Skin depth in tissue
        from ..utils.constants import MU_0
        delta = math.sqrt(2 / (omega * MU_0 * sigma))

        # Power absorbed in tissue
        P_rad = input_power_w * antenna_efficiency

        # Approximate E-field at tissue surface
        # For a small antenna at distance d, with effective area A:
        # The near-field power density ≈ P_rad / (2π·max(d, δ)²)
        d_eff = max(distance_to_tissue_m, delta, 0.001)
        S = P_rad / (2 * math.pi * d_eff**2)

        # E-field from power density: S = |E|²/(2·η_tissue)
        eta_tissue = math.sqrt(MU_0 / (8.854e-12 * tissue.eps_r))
        E_sq = 2 * S * eta_tissue

        # SAR = σ|E|² / (2ρ)
        sar_point = sigma * E_sq / (2 * rho)

        # 1g average (volume = 1cm³ = 1e-6 m³)
        # Averaging reduces peak by roughly factor of 3-10 for small antennas
        sar_1g = sar_point / 5.0
        sar_10g = sar_point / 15.0

        # Maximum safe power
        if sar_1g > 0:
            max_power_fcc = input_power_w * SAR_LIMIT_FCC_1G / sar_1g
        else:
            max_power_fcc = float('inf')

        if sar_10g > 0:
            max_power_icnirp = input_power_w * SAR_LIMIT_ICNIRP_10G / sar_10g
        else:
            max_power_icnirp = float('inf')

        max_safe = min(max_power_fcc, max_power_icnirp)

        return SARResult(
            peak_sar_1g=sar_1g,
            peak_sar_10g=sar_10g,
            input_power_w=input_power_w,
            compliant_fcc=sar_1g <= SAR_LIMIT_FCC_1G,
            compliant_icnirp=sar_10g <= SAR_LIMIT_ICNIRP_10G,
            max_safe_power_w=max_safe,
            notes=(
                f"Analytical estimate. Tissue: {tissue.name}, "
                f"σ={sigma:.3f} S/m, δ={delta*1e3:.1f} mm. "
                f"Use full-wave SAR simulation for compliance."
            ),
        )

    def estimate_thermal(
        self,
        sar_w_per_kg: float,
        tissue: Material = MUSCLE,
        perfusion_rate: float = 0.5e-3,  # blood perfusion [m³/(kg·s)]
        exposure_time_s: float = 3600,
    ) -> ThermalResult:
        """
        Estimate temperature rise from SAR using Pennes bioheat equation (simplified).

        ΔT_ss ≈ SAR / (ρ_blood · c_blood · w_b)
        where w_b is blood perfusion rate.

        Implant IEEE limit: ΔT < 1°C (for chronic implants).
        """
        rho_blood = 1060  # kg/m³
        c_blood = 3770    # J/(kg·K)
        c_tissue = 3500   # J/(kg·K) (approximate for muscle)

        # Steady-state temperature rise
        if perfusion_rate > 0:
            delta_T_ss = sar_w_per_kg / (rho_blood * c_blood * perfusion_rate)
        else:
            # No perfusion (e.g., bone): heat accumulates
            delta_T_ss = sar_w_per_kg * exposure_time_s / (tissue.density * c_tissue)

        # Time constant for thermal equilibration
        tau_thermal = 1.0 / max(rho_blood * c_blood * perfusion_rate / (tissue.density * c_tissue), 0.001)
        time_to_ss = 3 * tau_thermal  # ~95% of steady state

        # Safe duration (before exceeding 1°C)
        temp_limit = 1.0  # °C for implants
        if delta_T_ss > temp_limit:
            # Time to reach limit: T(t) = T_ss · (1 - exp(-t/τ))
            ratio = temp_limit / delta_T_ss
            if ratio < 1:
                safe_duration = -tau_thermal * math.log(1 - ratio)
            else:
                safe_duration = float('inf')
        else:
            safe_duration = float('inf')

        return ThermalResult(
            peak_temp_rise_c=delta_T_ss,
            time_to_steady_state_s=time_to_ss,
            safe_duration_s=safe_duration,
            temp_limit_c=temp_limit,
            compliant=delta_T_ss <= temp_limit,
        )

    def compliance_report(
        self,
        input_power_w: float,
        antenna_efficiency: float,
        freq_hz: float,
        tissue: Material = MUSCLE,
        distance_m: float = 0.0,
        antenna_area_m2: float = 1e-4,
    ) -> dict:
        """Generate full safety compliance report."""
        sar = self.estimate_sar(
            input_power_w, antenna_efficiency, freq_hz,
            tissue, distance_m, antenna_area_m2,
        )

        thermal = self.estimate_thermal(sar.peak_sar_1g, tissue)

        return {
            "sar": {
                "peak_1g_w_per_kg": sar.peak_sar_1g,
                "peak_10g_w_per_kg": sar.peak_sar_10g,
                "limit_fcc_1g": SAR_LIMIT_FCC_1G,
                "limit_icnirp_10g": SAR_LIMIT_ICNIRP_10G,
                "compliant_fcc": sar.compliant_fcc,
                "compliant_icnirp": sar.compliant_icnirp,
                "max_safe_power_mw": sar.max_safe_power_w * 1e3,
            },
            "thermal": {
                "peak_temp_rise_c": thermal.peak_temp_rise_c,
                "limit_c": thermal.temp_limit_c,
                "compliant": thermal.compliant,
                "safe_duration_min": thermal.safe_duration_s / 60 if thermal.safe_duration_s < 1e6 else None,
            },
            "input_power_mw": input_power_w * 1e3,
            "overall_compliant": sar.compliant_fcc and sar.compliant_icnirp and thermal.compliant,
            "notes": sar.notes,
        }
