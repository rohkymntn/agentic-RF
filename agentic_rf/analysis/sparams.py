"""
S-parameter analysis utilities.

Works with both simulation results and measured data (Touchstone files).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..solver.base import SimulationResult
from ..utils.units import mag_to_db, s11_to_vswr, s11_to_return_loss, s11_to_mismatch_loss


@dataclass
class BandwidthResult:
    """Result of bandwidth analysis."""
    center_freq_hz: float
    lower_freq_hz: float
    upper_freq_hz: float
    bandwidth_hz: float
    fractional_bandwidth: float
    best_s11_db: float
    best_s11_freq_hz: float


class SParameterAnalyzer:
    """Comprehensive S-parameter analysis."""

    def __init__(self, result: SimulationResult | None = None):
        self._result = result

    def set_result(self, result: SimulationResult):
        self._result = result

    def load_from_arrays(self, frequencies_hz: np.ndarray, s11_complex: np.ndarray):
        """Load from raw arrays (e.g., from VNA measurement)."""
        n = len(frequencies_hz)
        s_params = s11_complex.reshape(n, 1, 1)
        self._result = SimulationResult(
            frequencies_hz=frequencies_hz,
            s_parameters=s_params,
        )

    # ─── Basic metrics ───────────────────────────────────────────────────

    def s11_db(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (frequencies, S11 in dB)."""
        r = self._result
        return r.frequencies_hz, r.s11

    def vswr(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (frequencies, VSWR)."""
        r = self._result
        return r.frequencies_hz, r.vswr

    def return_loss(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (frequencies, return loss in dB) — positive values."""
        freqs, s11 = self.s11_db()
        return freqs, -s11

    def mismatch_loss(self) -> tuple[np.ndarray, np.ndarray]:
        """Mismatch loss in dB."""
        r = self._result
        s11_mag = np.abs(r.s11_complex)
        ml = -10 * np.log10(np.maximum(1 - s11_mag**2, 1e-12))
        return r.frequencies_hz, ml

    # ─── Bandwidth analysis ──────────────────────────────────────────────

    def find_resonances(self, threshold_db: float = -6.0) -> list[BandwidthResult]:
        """
        Find all resonance bands where S11 < threshold.

        Returns list of BandwidthResult for each distinct resonance.
        """
        r = self._result
        s11 = r.s11
        freqs = r.frequencies_hz

        results = []
        below = s11 < threshold_db
        if not np.any(below):
            return results

        # Find contiguous regions
        changes = np.diff(below.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        # Handle edge cases
        if below[0]:
            starts = np.insert(starts, 0, 0)
        if below[-1]:
            ends = np.append(ends, len(s11))

        for s_idx, e_idx in zip(starts, ends):
            band_freqs = freqs[s_idx:e_idx]
            band_s11 = s11[s_idx:e_idx]

            best_idx = np.argmin(band_s11)
            f_low = freqs[s_idx]
            f_high = freqs[min(e_idx, len(freqs) - 1)]
            f_center = (f_low + f_high) / 2
            bw = f_high - f_low

            results.append(BandwidthResult(
                center_freq_hz=f_center,
                lower_freq_hz=f_low,
                upper_freq_hz=f_high,
                bandwidth_hz=bw,
                fractional_bandwidth=bw / f_center if f_center > 0 else 0,
                best_s11_db=float(band_s11[best_idx]),
                best_s11_freq_hz=float(band_freqs[best_idx]),
            ))

        return results

    def bandwidth_at_threshold(self, threshold_db: float = -10.0) -> float:
        """Total bandwidth in Hz where S11 < threshold."""
        resonances = self.find_resonances(threshold_db)
        return sum(r.bandwidth_hz for r in resonances)

    # ─── Band compliance ─────────────────────────────────────────────────

    def check_band_coverage(self, band_start_hz: float, band_stop_hz: float,
                             threshold_db: float = -10.0) -> dict:
        """
        Check if antenna covers a specific frequency band.

        Returns dict with coverage metrics.
        """
        r = self._result
        mask = (r.frequencies_hz >= band_start_hz) & (r.frequencies_hz <= band_stop_hz)
        if not np.any(mask):
            return {"covered": False, "reason": "Band not in simulation range"}

        s11_in_band = r.s11[mask]
        freqs_in_band = r.frequencies_hz[mask]

        below_threshold = s11_in_band < threshold_db
        coverage_pct = np.mean(below_threshold) * 100

        return {
            "covered": coverage_pct > 90,
            "coverage_pct": coverage_pct,
            "worst_s11_db": float(np.max(s11_in_band)),
            "best_s11_db": float(np.min(s11_in_band)),
            "mean_s11_db": float(np.mean(s11_in_band)),
            "band_start_mhz": band_start_hz / 1e6,
            "band_stop_mhz": band_stop_hz / 1e6,
        }

    # ─── Stability / detuning analysis ───────────────────────────────────

    def detuning_sensitivity(self, freq_hz: float, delta_pct: float = 5.0) -> dict:
        """
        Analyze sensitivity to frequency detuning.

        Checks how much S11 degrades if resonance shifts by delta_pct.
        """
        r = self._result
        s11_at_f = r.s11_at_freq(freq_hz)

        f_shift = freq_hz * delta_pct / 100
        s11_shifted_up = r.s11_at_freq(freq_hz + f_shift)
        s11_shifted_down = r.s11_at_freq(freq_hz - f_shift)

        worst_shifted = max(s11_shifted_up, s11_shifted_down)

        return {
            "nominal_s11_db": s11_at_f,
            "shifted_up_s11_db": s11_shifted_up,
            "shifted_down_s11_db": s11_shifted_down,
            "worst_case_s11_db": worst_shifted,
            "margin_db": s11_at_f - worst_shifted,
            "robust": worst_shifted < -6.0,
        }

    # ─── Group delay ─────────────────────────────────────────────────────

    def group_delay(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute group delay from S11 phase."""
        r = self._result
        phase = np.unwrap(np.angle(r.s11_complex))
        freqs = r.frequencies_hz

        if len(freqs) < 2:
            return freqs, np.zeros_like(freqs)

        omega = 2 * np.pi * freqs
        d_phase = np.gradient(phase, omega)
        tau = -d_phase

        return freqs, tau

    # ─── Summary ─────────────────────────────────────────────────────────

    def full_report(self, target_freq_hz: float | None = None) -> dict:
        """Generate comprehensive analysis report."""
        r = self._result
        resonances = self.find_resonances(-10.0)

        report = {
            "n_resonances": len(resonances),
            "resonances": [],
            "total_bandwidth_mhz_10db": self.bandwidth_at_threshold(-10.0) / 1e6,
            "total_bandwidth_mhz_6db": self.bandwidth_at_threshold(-6.0) / 1e6,
        }

        for res in resonances:
            report["resonances"].append({
                "freq_mhz": res.best_s11_freq_hz / 1e6,
                "s11_db": res.best_s11_db,
                "bw_mhz": res.bandwidth_hz / 1e6,
                "fbw_pct": res.fractional_bandwidth * 100,
            })

        if target_freq_hz:
            report["target_freq_mhz"] = target_freq_hz / 1e6
            report["s11_at_target_db"] = r.s11_at_freq(target_freq_hz)
            report["detuning"] = self.detuning_sensitivity(target_freq_hz)

        return report
