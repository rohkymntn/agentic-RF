"""
Model calibration from measurements.

The key insight: simulations lie. Real antennas deviate from simulation due to:
  - Manufacturing tolerances (etch, placement)
  - Material property variation (εr, tan_δ vary with batch/frequency)
  - Environment effects (nearby objects, enclosure, tissue loading)
  - Fixture/cable effects

This module fits "reality correction" parameters by comparing
simulated vs measured data, then updates the simulation model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from ..solver.base import SimulationResult
from ..geometry.base import AntennaTemplate
from ..utils.constants import C_0


@dataclass
class CalibrationResult:
    """Result of model calibration."""
    # Fitted correction factors
    freq_shift_factor: float = 1.0    # multiply resonant freq by this
    er_correction: float = 1.0        # multiply εr by this
    loss_correction: float = 1.0      # multiply losses by this
    impedance_scale: float = 1.0      # scale impedance magnitude

    # Goodness of fit
    rms_error_db: float = 0.0         # RMS error in S11 (dB)
    max_error_db: float = 0.0
    correlation: float = 0.0          # correlation coefficient

    # Diagnostic info
    measured_resonance_hz: float = 0.0
    simulated_resonance_hz: float = 0.0
    freq_error_pct: float = 0.0

    notes: str = ""


@dataclass
class RealityModel:
    """
    Updated model parameters that account for real-world deviations.

    Apply these corrections when re-simulating to get more accurate results.
    """
    effective_er: float = 4.4        # calibrated effective permittivity
    effective_tan_d: float = 0.02    # calibrated loss tangent
    freq_offset_hz: float = 0.0     # frequency offset to apply
    z_scale: float = 1.0            # impedance scaling
    additional_loss_ohm: float = 0.0  # additional series resistance

    # Environmental detuning
    detuning_er_factor: float = 1.0  # εr change from environment
    detuning_notes: str = ""

    def apply_to_params(self, params: dict[str, float]) -> dict[str, float]:
        """Apply calibration corrections to design parameters."""
        corrected = params.copy()
        if "substrate_er" in corrected:
            corrected["substrate_er"] *= self.detuning_er_factor
        return corrected


class ModelCalibrator:
    """
    Calibrates simulation models against measurement data.

    Workflow:
    1. Simulate the designed antenna
    2. Measure the fabricated antenna (VNA → S1P file)
    3. Run calibration to find correction factors
    4. Re-simulate with corrections → much closer to reality
    5. Re-optimize if needed
    """

    def __init__(self, template: AntennaTemplate | None = None):
        self._template = template

    def calibrate(
        self,
        simulated: SimulationResult,
        measured: SimulationResult,
        params: dict[str, float] | None = None,
    ) -> CalibrationResult:
        """
        Compare simulation to measurement and find correction factors.

        Args:
            simulated: Simulation result
            measured: Measurement result (from VNA/Touchstone)
            params: Design parameters (optional, for parameter-level correction)

        Returns:
            CalibrationResult with correction factors
        """
        result = CalibrationResult()

        # Find resonance frequencies
        sim_f0 = simulated.resonance_freq()
        meas_f0 = measured.resonance_freq()

        result.simulated_resonance_hz = sim_f0
        result.measured_resonance_hz = meas_f0

        if sim_f0 > 0 and meas_f0 > 0:
            result.freq_shift_factor = meas_f0 / sim_f0
            result.freq_error_pct = (meas_f0 - sim_f0) / sim_f0 * 100

            # εr correction: f ∝ 1/√εr, so εr_corr = (f_sim/f_meas)²
            result.er_correction = (sim_f0 / meas_f0) ** 2

        # Interpolate to common frequency grid for comparison
        f_common = np.linspace(
            max(simulated.frequencies_hz[0], measured.frequencies_hz[0]),
            min(simulated.frequencies_hz[-1], measured.frequencies_hz[-1]),
            min(len(simulated.frequencies_hz), len(measured.frequencies_hz)),
        )

        if len(f_common) < 3:
            result.notes = "Insufficient frequency overlap for detailed calibration"
            return result

        sim_s11 = np.interp(f_common, simulated.frequencies_hz, simulated.s11)
        meas_s11 = np.interp(f_common, measured.frequencies_hz, measured.s11)

        # Error metrics
        error = sim_s11 - meas_s11
        result.rms_error_db = float(np.sqrt(np.mean(error**2)))
        result.max_error_db = float(np.max(np.abs(error)))

        # Correlation
        if np.std(sim_s11) > 0 and np.std(meas_s11) > 0:
            result.correlation = float(np.corrcoef(sim_s11, meas_s11)[0, 1])

        # Loss correction: compare S11 depths
        sim_min_s11 = float(np.min(sim_s11))
        meas_min_s11 = float(np.min(meas_s11))
        if sim_min_s11 < -3 and meas_min_s11 < -3:
            # If measured S11 is less deep, antenna has more loss
            result.loss_correction = 10 ** ((meas_min_s11 - sim_min_s11) / 20)

        # Impedance scaling from S11 magnitude
        sim_s11_mag = np.abs(np.interp(f_common, simulated.frequencies_hz,
                                        simulated.s11_complex))
        meas_s11_mag = np.abs(np.interp(f_common, measured.frequencies_hz,
                                         measured.s11_complex))
        ratio = np.median(meas_s11_mag / np.maximum(sim_s11_mag, 1e-6))
        result.impedance_scale = float(np.clip(ratio, 0.5, 2.0))

        result.notes = (
            f"Freq error: {result.freq_error_pct:+.1f}%, "
            f"εr correction: ×{result.er_correction:.3f}, "
            f"Loss correction: ×{result.loss_correction:.3f}, "
            f"RMS error: {result.rms_error_db:.2f} dB"
        )

        return result

    def build_reality_model(
        self,
        calibration: CalibrationResult,
        original_params: dict[str, float],
    ) -> RealityModel:
        """
        Build a corrected reality model from calibration results.

        This model can be used to re-simulate with higher accuracy.
        """
        model = RealityModel()

        # Update effective permittivity
        if "substrate_er" in original_params:
            model.effective_er = original_params["substrate_er"] * calibration.er_correction
        else:
            model.effective_er = 4.4 * calibration.er_correction

        # Update loss
        model.effective_tan_d = 0.02 * calibration.loss_correction

        # Frequency offset
        if calibration.simulated_resonance_hz > 0:
            model.freq_offset_hz = (calibration.measured_resonance_hz -
                                     calibration.simulated_resonance_hz)

        model.z_scale = calibration.impedance_scale
        model.detuning_er_factor = calibration.er_correction
        model.detuning_notes = calibration.notes

        return model

    def fit_environment_model(
        self,
        free_space_measurement: SimulationResult,
        loaded_measurement: SimulationResult,
    ) -> dict:
        """
        Determine environmental loading from two measurements.

        Compare free-space (or phantom-less) measurement to
        loaded (in-enclosure, on-body, in-phantom) measurement.

        Returns estimated environmental detuning parameters.
        """
        f0_free = free_space_measurement.resonance_freq()
        f0_loaded = loaded_measurement.resonance_freq()

        freq_shift = f0_loaded - f0_free
        freq_shift_pct = freq_shift / f0_free * 100 if f0_free > 0 else 0

        # εr change from frequency shift
        er_change = (f0_free / max(f0_loaded, 1)) ** 2

        # Q change from bandwidth
        bw_free = free_space_measurement.bandwidth_hz(-10)
        bw_loaded = loaded_measurement.bandwidth_hz(-10)
        q_free = f0_free / max(bw_free, 1)
        q_loaded = f0_loaded / max(bw_loaded, 1)
        q_ratio = q_loaded / max(q_free, 1)

        # S11 depth change (indicates loss)
        s11_free = free_space_measurement.s11_at_freq(f0_free)
        s11_loaded = loaded_measurement.s11_at_freq(f0_loaded)

        return {
            "freq_shift_hz": freq_shift,
            "freq_shift_pct": freq_shift_pct,
            "er_loading_factor": er_change,
            "q_ratio": q_ratio,
            "additional_loss_indicated": s11_loaded > s11_free,
            "s11_change_db": s11_loaded - s11_free,
            "recommendation": (
                f"Environment shifts resonance by {freq_shift_pct:+.1f}%. "
                f"Apply εr×{er_change:.3f} correction. "
                f"Q changed by x{q_ratio:.2f} ({'more' if q_ratio < 1 else 'less'} lossy)."
            ),
        }

    def suggest_redesign(
        self,
        calibration: CalibrationResult,
        original_params: dict[str, float],
        target_freq_hz: float,
    ) -> dict[str, float]:
        """
        Suggest parameter modifications to hit the target after calibration.

        Simple approach: if measured resonance is too low, reduce lengths;
        if too high, increase lengths.
        """
        if self._template is None:
            return original_params

        corrected = original_params.copy()

        # Frequency correction: adjust length-related parameters
        freq_ratio = target_freq_hz / max(calibration.measured_resonance_hz, 1)

        # For most antennas, f ∝ 1/L, so L_new = L_old / freq_ratio
        length_params = [p for p in self._template.parameters
                         if 'length' in p.name.lower() or 'radius' in p.name.lower()
                         or 'width' in p.name.lower()]

        for p in length_params:
            if p.name in corrected:
                corrected[p.name] = p.validate(corrected[p.name] / freq_ratio)

        return corrected
