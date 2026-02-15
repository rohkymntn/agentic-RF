"""
Touchstone file I/O (S1P, S2P, SNP).

Handles reading VNA measurement data and simulation exports.
Uses scikit-rf for robust parsing, with fallback manual parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..solver.base import SimulationResult


class TouchstoneIO:
    """
    Read/write Touchstone (SNP) files.

    Supports S1P (1-port) and S2P (2-port) formats.
    """

    @staticmethod
    def read(filepath: str | Path) -> SimulationResult:
        """
        Read a Touchstone file and return a SimulationResult.

        Args:
            filepath: Path to .s1p or .s2p file

        Returns:
            SimulationResult populated with S-parameter data
        """
        filepath = Path(filepath)

        try:
            return TouchstoneIO._read_skrf(filepath)
        except ImportError:
            return TouchstoneIO._read_manual(filepath)

    @staticmethod
    def _read_skrf(filepath: Path) -> SimulationResult:
        """Read using scikit-rf (robust, handles all edge cases)."""
        import skrf

        ntwk = skrf.Network(str(filepath))

        n_ports = ntwk.number_of_ports
        freqs = ntwk.f  # Hz
        s = ntwk.s      # complex S-parameter array [n_freq, n_ports, n_ports]

        # Compute derived quantities
        z = ntwk.z  # impedance matrix

        result = SimulationResult(
            frequencies_hz=freqs,
            s_parameters=s,
            z_input=z[:, 0:1, 0] if z is not None else np.array([]),
            solver_name=f"Measurement ({filepath.name})",
            notes=f"Loaded from {filepath.name}, {n_ports}-port, {len(freqs)} points",
        )

        return result

    @staticmethod
    def _read_manual(filepath: Path) -> SimulationResult:
        """
        Manual Touchstone parser (fallback if skrf unavailable).

        Handles basic S1P and S2P files with RI, MA, or DB formats.
        """
        freqs = []
        s_data = []
        freq_unit_mult = 1e9  # default GHz
        format_type = "RI"    # default Real/Imaginary
        z0 = 50.0

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('!'):
                    continue

                # Option line
                if line.startswith('#'):
                    parts = line[1:].upper().split()
                    for i, p in enumerate(parts):
                        if p in ('HZ', 'KHZ', 'MHZ', 'GHZ'):
                            freq_unit_mult = {
                                'HZ': 1, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9
                            }[p]
                        if p in ('RI', 'MA', 'DB'):
                            format_type = p
                        if p == 'R':
                            try:
                                z0 = float(parts[i + 1])
                            except (IndexError, ValueError):
                                pass
                    continue

                # Data line
                values = [float(x) for x in line.split()]
                freq = values[0] * freq_unit_mult
                freqs.append(freq)

                # Parse S-parameters
                s_vals = values[1:]
                s_complex = []
                for j in range(0, len(s_vals), 2):
                    if j + 1 < len(s_vals):
                        v1, v2 = s_vals[j], s_vals[j + 1]
                        if format_type == "RI":
                            s_complex.append(complex(v1, v2))
                        elif format_type == "MA":
                            s_complex.append(v1 * np.exp(1j * np.radians(v2)))
                        elif format_type == "DB":
                            mag = 10 ** (v1 / 20)
                            s_complex.append(mag * np.exp(1j * np.radians(v2)))

                s_data.append(s_complex)

        freqs = np.array(freqs)
        s_data = np.array(s_data)

        # Determine number of ports
        n_s = s_data.shape[1] if len(s_data.shape) > 1 else 1
        n_ports = int(np.sqrt(n_s))
        s_params = s_data.reshape(len(freqs), n_ports, n_ports)

        # Compute impedance from S11
        if n_ports >= 1:
            s11 = s_params[:, 0, 0]
            z_in = z0 * (1 + s11) / (1 - s11)
        else:
            z_in = np.array([])

        return SimulationResult(
            frequencies_hz=freqs,
            s_parameters=s_params,
            z_input=z_in.reshape(-1, 1) if z_in.size > 0 else np.array([]),
            solver_name=f"Measurement ({filepath.name})",
            notes=f"Loaded from {filepath.name}, Z0={z0}Ω, format={format_type}",
        )

    @staticmethod
    def write_s1p(filepath: str | Path, frequencies_hz: np.ndarray,
                  s11: np.ndarray, z0: float = 50.0,
                  format_type: str = "RI"):
        """
        Write a 1-port Touchstone file.

        Args:
            filepath: Output path
            frequencies_hz: Frequency array
            s11: Complex S11 array
            z0: Reference impedance
            format_type: "RI", "MA", or "DB"
        """
        filepath = Path(filepath)

        with open(filepath, 'w') as f:
            f.write(f"! Agentic RF - S1P export\n")
            f.write(f"! {len(frequencies_hz)} points\n")
            f.write(f"# HZ S {format_type} R {z0}\n")

            for freq, s in zip(frequencies_hz, s11):
                if format_type == "RI":
                    f.write(f"{freq:.6e} {s.real:.8e} {s.imag:.8e}\n")
                elif format_type == "MA":
                    mag = abs(s)
                    ang = np.degrees(np.angle(s))
                    f.write(f"{freq:.6e} {mag:.8e} {ang:.4f}\n")
                elif format_type == "DB":
                    db = 20 * np.log10(max(abs(s), 1e-12))
                    ang = np.degrees(np.angle(s))
                    f.write(f"{freq:.6e} {db:.4f} {ang:.4f}\n")

    @staticmethod
    def write_s2p(filepath: str | Path, frequencies_hz: np.ndarray,
                  s_params: np.ndarray, z0: float = 50.0):
        """Write a 2-port Touchstone file."""
        filepath = Path(filepath)

        with open(filepath, 'w') as f:
            f.write(f"! Agentic RF - S2P export\n")
            f.write(f"# HZ S RI R {z0}\n")

            for i, freq in enumerate(frequencies_hz):
                s = s_params[i]
                f.write(
                    f"{freq:.6e} "
                    f"{s[0,0].real:.8e} {s[0,0].imag:.8e} "
                    f"{s[0,1].real:.8e} {s[0,1].imag:.8e} "
                    f"{s[1,0].real:.8e} {s[1,0].imag:.8e} "
                    f"{s[1,1].real:.8e} {s[1,1].imag:.8e}\n"
                )

    @staticmethod
    def detect_fixture_artifacts(result: SimulationResult,
                                  smoothing_window: int = 5) -> dict:
        """
        Detect cable/fixture artifacts in VNA measurement data.

        Looks for:
          - Ripple (periodic variations from cable reflections)
          - Phase slope (cable delay)
          - Frequency-dependent loss
        """
        s11 = result.s11
        freqs = result.frequencies_hz

        if len(s11) < smoothing_window * 2:
            return {"ripple_detected": False}

        # Detect ripple by looking at high-frequency content
        from scipy.signal import detrend
        s11_detrended = detrend(s11)
        rms_ripple = np.sqrt(np.mean(s11_detrended**2))

        # Phase unwrap and check for linear slope (cable delay)
        phase = np.unwrap(np.angle(result.s11_complex))
        if len(phase) > 1:
            phase_slope = np.polyfit(freqs, phase, 1)
            cable_delay_ns = -phase_slope[0] / (2 * np.pi) * 1e9
        else:
            cable_delay_ns = 0

        return {
            "ripple_detected": rms_ripple > 0.5,
            "ripple_rms_db": rms_ripple,
            "cable_delay_ns": cable_delay_ns,
            "suggestion": (
                "Apply time-domain gating or port extension to remove fixture effects"
                if rms_ripple > 0.5 else "Data looks clean"
            ),
        }
