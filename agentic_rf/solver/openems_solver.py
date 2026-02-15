"""
OpenEMS FDTD solver interface.

OpenEMS is an open-source FDTD electromagnetic field solver.
This module translates AntennaGeometry into OpenEMS simulation scripts
and runs them.

Requires:
  - openEMS installed (via package manager)
  - CSXCAD Python bindings
  - AppCSXCAD for visualization (optional)

Install (Ubuntu/Debian):
  sudo apt install openems
  pip install CSXCAD openEMS

Install (macOS via Homebrew):
  brew install openems
"""

from __future__ import annotations

import os
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from .base import SolverBase, SimulationResult
from ..geometry.base import AntennaGeometry, Point3D, Box3D, TraceSegment, Port


class OpenEMSSolver(SolverBase):
    """
    Full-wave FDTD solver using OpenEMS.

    Generates a CSXCAD geometry, runs openEMS, and parses results.
    """

    def __init__(self, work_dir: str | None = None, num_threads: int = 0,
                 end_criteria: float = 1e-5, max_timesteps: int = 500_000):
        """
        Args:
            work_dir: Working directory for simulation files. None = temp dir.
            num_threads: OpenEMS threads (0 = auto).
            end_criteria: Energy decay convergence threshold.
            max_timesteps: Maximum FDTD time steps.
        """
        self._work_dir = work_dir
        self._num_threads = num_threads
        self._end_criteria = end_criteria
        self._max_timesteps = max_timesteps

    @property
    def name(self) -> str:
        return "OpenEMS (FDTD)"

    def is_available(self) -> bool:
        """Check if OpenEMS Python bindings are importable."""
        try:
            import CSXCAD
            import openEMS
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
        boundary_conditions: str = "MUR",
        mesh_resolution: float = 20.0,
        nf2ff: bool = True,
        **kwargs,
    ) -> SimulationResult:
        """
        Run OpenEMS FDTD simulation.

        Args:
            geometry: Antenna geometry
            freq_start_hz: Start frequency
            freq_stop_hz: Stop frequency
            n_freq: Number of frequency points
            z0: Reference impedance
            boundary_conditions: "MUR", "PML_8", or "PEC"
            mesh_resolution: Target mesh resolution (cells per wavelength)
            nf2ff: Compute near-field to far-field transform
        """
        if not self.is_available():
            raise RuntimeError(
                "OpenEMS is not installed. Install with:\n"
                "  Ubuntu: sudo apt install openems && pip install CSXCAD openEMS\n"
                "  macOS: brew install openems"
            )

        import CSXCAD
        from CSXCAD import ContinuousStructure
        import openEMS
        from openEMS import openEMS as OpenEMS

        t_start = time.time()

        # Create working directory
        if self._work_dir:
            sim_dir = Path(self._work_dir) / "openems_sim"
        else:
            sim_dir = Path(tempfile.mkdtemp(prefix="agentic_rf_"))
        sim_dir.mkdir(parents=True, exist_ok=True)

        f_center = (freq_start_hz + freq_stop_hz) / 2
        f_span = freq_stop_hz - freq_start_hz

        # ── Initialize FDTD ──────────────────────────────────────────────
        FDTD = OpenEMS(
            EndCriteria=self._end_criteria,
            NrTS=self._max_timesteps,
        )
        FDTD.SetGaussExcite(f_center, f_span / 2)

        # Boundary conditions
        bc_map = {
            "MUR": [1, 1, 1, 1, 1, 1],
            "PML_8": [3, 3, 3, 3, 3, 3],
            "PEC": [0, 0, 0, 0, 0, 0],
        }
        FDTD.SetBoundaryCond(bc_map.get(boundary_conditions, [1, 1, 1, 1, 1, 1]))

        CSX = ContinuousStructure()
        FDTD.SetCSX(CSX)

        # ── Translate geometry ───────────────────────────────────────────
        # Mesh lines collectors
        mesh_x, mesh_y, mesh_z = [], [], []

        # Add boxes (substrates, ground planes)
        for i, box in enumerate(geometry.boxes):
            mat = CSX.AddMaterial(f"mat_{i}")
            if box.material.is_conductor:
                mat = CSX.AddMetal(f"metal_{i}")
            else:
                mat.SetEpsilon(box.material.eps_r)
                if box.material.tan_d > 0:
                    mat.SetKappa(
                        box.material.effective_conductivity(f_center)
                    )

            p1 = box.origin
            p2 = Point3D(
                p1.x + box.size.x,
                p1.y + box.size.y,
                p1.z + box.size.z,
            )
            mat.AddBox(
                [p1.x * 1e3, p1.y * 1e3, p1.z * 1e3],
                [p2.x * 1e3, p2.y * 1e3, p2.z * 1e3],
                priority=10 + i,
            )
            mesh_x.extend([p1.x * 1e3, p2.x * 1e3])
            mesh_y.extend([p1.y * 1e3, p2.y * 1e3])
            mesh_z.extend([p1.z * 1e3, p2.z * 1e3])

        # Add traces
        trace_metal = CSX.AddMetal("trace_copper")
        for i, trace in enumerate(geometry.traces):
            # Approximate trace as thin box
            w2 = trace.width / 2 * 1e3
            t2 = trace.thickness / 2 * 1e3
            p1 = trace.start
            p2 = trace.end
            trace_metal.AddBox(
                [p1.x * 1e3 - w2, p1.y * 1e3 - w2, p1.z * 1e3 - t2],
                [p2.x * 1e3 + w2, p2.y * 1e3 + w2, p2.z * 1e3 + t2],
                priority=100,
            )
            mesh_x.extend([p1.x * 1e3, p2.x * 1e3])
            mesh_y.extend([p1.y * 1e3, p2.y * 1e3])
            mesh_z.extend([p1.z * 1e3, p2.z * 1e3])

        # Add ports
        for port_def in geometry.ports:
            port = FDTD.AddLumpedPort(
                port_nr=port_def.port_number,
                R=port_def.impedance,
                start=[port_def.start.x * 1e3, port_def.start.y * 1e3, port_def.start.z * 1e3],
                stop=[port_def.end.x * 1e3, port_def.end.y * 1e3, port_def.end.z * 1e3],
                p_dir="z",  # TODO: auto-detect direction
                excite=1.0 if port_def.port_number == 1 else 0,
            )

        # ── Generate mesh ────────────────────────────────────────────────
        lam_min = 3e8 / freq_stop_hz * 1e3  # in mm
        max_cell = lam_min / mesh_resolution

        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(1e-3)  # mm

        # Add geometry-based mesh lines with smooth meshing
        mesh.AddLine('x', sorted(set(mesh_x)))
        mesh.AddLine('y', sorted(set(mesh_y)))
        mesh.AddLine('z', sorted(set(mesh_z)))
        mesh.SmoothMeshLines('x', max_cell)
        mesh.SmoothMeshLines('y', max_cell)
        mesh.SmoothMeshLines('z', max_cell)

        # ── NF2FF box ────────────────────────────────────────────────────
        nf2ff_box = None
        if nf2ff:
            nf2ff_box = FDTD.CreateNF2FFBox()

        # ── Run simulation ───────────────────────────────────────────────
        FDTD.Run(str(sim_dir), verbose=0, numThreads=self._num_threads)

        # ── Post-process ─────────────────────────────────────────────────
        freqs = np.linspace(freq_start_hz, freq_stop_hz, n_freq)

        # Get port data
        port = FDTD.GetLumpedPort(1)
        port.CalcPort(str(sim_dir), freqs)

        s11 = port.uf_ref / port.uf_inc
        z_in = port.uf_tot / port.if_tot

        s_params = s11.reshape(n_freq, 1, 1)
        z_input = z_in.reshape(n_freq, 1)

        # Far-field (if computed)
        peak_gain = np.zeros(n_freq)
        peak_dir = np.zeros(n_freq)
        pattern_data = {}

        if nf2ff and nf2ff_box is not None:
            theta = np.linspace(0, np.pi, 37)
            phi = np.linspace(0, 2 * np.pi, 73)

            gain_pattern = np.zeros((n_freq, len(theta), len(phi)))

            for i, f in enumerate(freqs[::max(1, n_freq // 10)]):
                nf2ff_res = nf2ff_box.CalcNF2FF(
                    str(sim_dir), f, theta, phi,
                )
                gain_pattern[i] = nf2ff_res.Dmax * nf2ff_res.E_norm[0]
                peak_dir[i] = 10 * np.log10(nf2ff_res.Dmax)

            pattern_data = {
                "gain_pattern": gain_pattern,
                "theta_angles": theta,
                "phi_angles": phi,
            }

        # Efficiency from accepted / radiated power
        P_acc = port.P_acc
        P_inc = port.P_inc
        mismatch_eff = P_acc / np.maximum(P_inc, 1e-12)

        solve_time = time.time() - t_start

        result = SimulationResult(
            frequencies_hz=freqs,
            s_parameters=s_params,
            z_input=z_input,
            radiation_efficiency=np.ones(n_freq) * 0.9,  # placeholder
            total_efficiency=mismatch_eff * 0.9,
            peak_gain_dbi=peak_gain,
            peak_directivity_dbi=peak_dir,
            solver_name=self.name,
            solve_time_seconds=solve_time,
            converged=True,
            **pattern_data,
        )

        return result

    def generate_script(self, geometry: AntennaGeometry,
                        freq_start_hz: float, freq_stop_hz: float,
                        output_path: str) -> str:
        """
        Generate a standalone OpenEMS Python script (no dependencies on agentic_rf).

        Useful for debugging, sharing, or running on a cluster.
        """
        script_lines = [
            "#!/usr/bin/env python3",
            '"""Auto-generated OpenEMS simulation script by Agentic RF."""',
            "",
            "import os",
            "import numpy as np",
            "import CSXCAD",
            "from CSXCAD import ContinuousStructure",
            "from openEMS import openEMS",
            "",
            f"# Frequency range",
            f"f_start = {freq_start_hz}",
            f"f_stop = {freq_stop_hz}",
            f"f_center = (f_start + f_stop) / 2",
            f"f_span = f_stop - f_start",
            "",
            "# Setup FDTD",
            f"FDTD = openEMS(EndCriteria={self._end_criteria}, NrTS={self._max_timesteps})",
            "FDTD.SetGaussExcite(f_center, f_span / 2)",
            "FDTD.SetBoundaryCond(['MUR'] * 6)",
            "",
            "CSX = ContinuousStructure()",
            "FDTD.SetCSX(CSX)",
            "",
        ]

        # Add geometry generation code
        for i, box in enumerate(geometry.boxes):
            p1 = box.origin
            sz = box.size
            if box.material.is_conductor:
                script_lines.append(f"metal_{i} = CSX.AddMetal('metal_{i}')")
            else:
                script_lines.append(f"mat_{i} = CSX.AddMaterial('mat_{i}')")
                script_lines.append(f"mat_{i}.SetEpsilon({box.material.eps_r})")
            mat_name = f"metal_{i}" if box.material.is_conductor else f"mat_{i}"
            script_lines.append(
                f"{mat_name}.AddBox("
                f"[{p1.x*1e3}, {p1.y*1e3}, {p1.z*1e3}], "
                f"[{(p1.x+sz.x)*1e3}, {(p1.y+sz.y)*1e3}, {(p1.z+sz.z)*1e3}], "
                f"priority={10+i})"
            )

        script_lines.extend([
            "",
            "# Run",
            f"sim_dir = os.path.join(os.path.dirname(__file__), 'sim_output')",
            "os.makedirs(sim_dir, exist_ok=True)",
            "FDTD.Run(sim_dir, verbose=1)",
            "",
            "print('Simulation complete!')",
        ])

        script = "\n".join(script_lines)

        with open(output_path, 'w') as f:
            f.write(script)

        return script
