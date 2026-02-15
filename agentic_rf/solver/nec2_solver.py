"""
NEC2 Method of Moments solver interface.

NEC2 is excellent for wire antennas (dipoles, loops, helices, Yagi-Uda).
Uses PyNEC Python bindings.

Install: pip install PyNEC
"""

from __future__ import annotations

import time
import math

import numpy as np

from .base import SolverBase, SimulationResult
from ..geometry.base import AntennaGeometry


class NEC2Solver(SolverBase):
    """
    NEC2 Method of Moments solver via PyNEC.

    Best for: wire antennas, loops, helices, Yagi arrays.
    Not suitable for: patch antennas, planar structures (use OpenEMS instead).
    """

    def __init__(self, wire_segments_per_wavelength: int = 20):
        self._segs_per_lam = wire_segments_per_wavelength

    @property
    def name(self) -> str:
        return "NEC2 (MoM)"

    def is_available(self) -> bool:
        try:
            import PyNEC
            return True
        except ImportError:
            return False

    def simulate(
        self,
        geometry: AntennaGeometry,
        freq_start_hz: float,
        freq_stop_hz: float,
        n_freq: int = 101,
        z0: float = 50.0,
        ground_type: str = "free_space",
        **kwargs,
    ) -> SimulationResult:
        """
        Run NEC2 simulation.

        Args:
            geometry: Antenna geometry (traces interpreted as wires)
            freq_start_hz: Start frequency
            freq_stop_hz: Stop frequency
            n_freq: Number of frequency points
            z0: Reference impedance
            ground_type: "free_space", "perfect_ground", or "real_ground"
        """
        if not self.is_available():
            raise RuntimeError("PyNEC not installed. pip install PyNEC")

        import PyNEC

        t_start = time.time()

        freqs = np.linspace(freq_start_hz, freq_stop_hz, n_freq)

        # Create NEC context
        nec = PyNEC.nec_context()
        geo = nec.get_geometry()

        # Convert traces to NEC wires
        lam_min = 3e8 / freq_stop_hz
        wire_radius = 0.001  # default 1mm radius

        for i, trace in enumerate(geometry.traces):
            # Number of segments
            length = math.sqrt(
                (trace.end.x - trace.start.x) ** 2 +
                (trace.end.y - trace.start.y) ** 2 +
                (trace.end.z - trace.start.z) ** 2
            )
            n_segs = max(1, int(length / lam_min * self._segs_per_lam))
            wire_radius = trace.width / 4  # approximate

            geo.wire(
                i + 1,  # tag
                n_segs,
                trace.start.x, trace.start.y, trace.start.z,
                trace.end.x, trace.end.y, trace.end.z,
                wire_radius,
                1.0, 1.0,  # ratio (uniform segments)
            )

        geo.geometry_complete(0)

        # Ground
        if ground_type == "perfect_ground":
            nec.gn_card(1, 0, 0, 0, 0, 0, 0, 0)
        elif ground_type == "real_ground":
            nec.gn_card(0, 0, 13, 0.005, 0, 0, 0, 0)  # average ground

        # Excitation at first port
        if geometry.ports:
            port = geometry.ports[0]
            # Find the wire/segment closest to port
            nec.ex_card(0, 1, 1, 0, 1.0, 0, 0, 0, 0, 0)

        # Loading (if needed)
        nec.ld_card(5, 0, 0, 0, 5.8e7, 0, 0)  # copper conductivity

        # Radiation pattern request
        nec.rp_card(0, 37, 73, 0, 0, 0, 0, 0, 5, 5, 0, 0)

        # Frequency sweep
        s_params = np.zeros((n_freq, 1, 1), dtype=complex)
        z_input = np.zeros((n_freq, 1), dtype=complex)
        peak_gain = np.zeros(n_freq)
        peak_dir = np.zeros(n_freq)
        rad_eff = np.zeros(n_freq)

        for i, f in enumerate(freqs):
            nec.fr_card(0, 1, f / 1e6, 0)  # NEC uses MHz
            nec.xq_card(0)

            # Get impedance
            ipt = nec.get_input_parameters(0)
            z = complex(ipt.get_impedance_real(), ipt.get_impedance_imag())
            z_input[i, 0] = z

            # S11
            gamma = (z - z0) / (z + z0)
            s_params[i, 0, 0] = gamma

            # Get gain
            rp = nec.get_radiation_pattern(0)
            gains = rp.get_gain()
            peak_gain[i] = np.max(gains)
            peak_dir[i] = peak_gain[i]  # NEC gain includes efficiency

            # Efficiency
            rad_eff[i] = rp.get_power_gain() / max(rp.get_directivity(), 1e-6)

        total_eff = rad_eff * (1 - np.abs(s_params[:, 0, 0]) ** 2)

        solve_time = time.time() - t_start

        return SimulationResult(
            frequencies_hz=freqs,
            s_parameters=s_params,
            z_input=z_input,
            radiation_efficiency=rad_eff,
            total_efficiency=total_eff,
            peak_gain_dbi=peak_gain,
            peak_directivity_dbi=peak_dir,
            solver_name=self.name,
            solve_time_seconds=solve_time,
            converged=True,
        )
