"""
Near-field and radiation field computation for visualization.

Provides approximate E-field and H/B-field distributions for each
antenna type. These are analytical approximations suitable for
visualization — not replacements for full-wave solvers.

Field models:
  - Magnetic dipole (loops, spirals, small helices)
  - Electric dipole (meanders, MIFA, short monopoles)
  - Patch cavity model (patch antennas)
  - Hertzian dipole (PIFA feed pin)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..utils.constants import C_0, MU_0, EPS_0, ETA_0


def compute_near_fields(
    antenna_type: str,
    freq_hz: float,
    params: dict[str, float],
    plane: str = "xz",
    extent_mm: float = 100.0,
    n_points: int = 60,
    z_offset_mm: float = 0.0,
) -> dict:
    """
    Compute approximate near-field distribution on a 2D plane.

    Args:
        antenna_type: "loop", "dipole", "patch", "helix" (selects field model)
        freq_hz: Operating frequency
        params: Antenna parameters
        plane: "xz", "xy", or "yz" — the observation plane
        extent_mm: Half-extent of the observation plane in mm
        n_points: Grid resolution per axis
        z_offset_mm: Offset of the observation plane

    Returns:
        dict with:
            coord1, coord2: 2D meshgrid arrays (mm)
            Ex, Ey, Ez: complex E-field components (V/m)
            Hx, Hy, Hz: complex H-field components (A/m)
            Bx, By, Bz: B-field components (T)
            E_mag, H_mag, B_mag: field magnitudes
            labels: axis labels
    """
    mm = 1e-3
    extent = extent_mm * mm
    offset = z_offset_mm * mm

    # Create observation grid
    c1 = np.linspace(-extent, extent, n_points)
    c2 = np.linspace(-extent, extent, n_points)
    C1, C2 = np.meshgrid(c1, c2)

    # Map to 3D coordinates
    if plane == "xz":
        X, Y, Z = C1, np.full_like(C1, offset), C2
        labels = ("X (mm)", "Z (mm)")
    elif plane == "xy":
        X, Y, Z = C1, C2, np.full_like(C1, offset)
        labels = ("X (mm)", "Y (mm)")
    elif plane == "yz":
        X, Y, Z = np.full_like(C1, offset), C1, C2
        labels = ("Y (mm)", "Z (mm)")
    else:
        raise ValueError(f"Unknown plane: {plane}")

    omega = 2 * math.pi * freq_hz
    k = omega / C_0
    lam = C_0 / freq_hz

    # Select field model
    if antenna_type in ("loop", "spiral"):
        Ex, Ey, Ez, Hx, Hy, Hz = _magnetic_dipole_fields(
            X, Y, Z, freq_hz, params
        )
    elif antenna_type in ("meander", "mifa", "dipole"):
        Ex, Ey, Ez, Hx, Hy, Hz = _electric_dipole_fields(
            X, Y, Z, freq_hz, params
        )
    elif antenna_type == "patch":
        Ex, Ey, Ez, Hx, Hy, Hz = _patch_cavity_fields(
            X, Y, Z, freq_hz, params
        )
    elif antenna_type == "helix":
        # Small helix ≈ magnetic dipole + electric dipole
        Ex1, Ey1, Ez1, Hx1, Hy1, Hz1 = _magnetic_dipole_fields(X, Y, Z, freq_hz, params)
        Ex2, Ey2, Ez2, Hx2, Hy2, Hz2 = _electric_dipole_fields(X, Y, Z, freq_hz, params)
        Ex = Ex1 + 0.3 * Ex2
        Ey = Ey1 + 0.3 * Ey2
        Ez = Ez1 + 0.3 * Ez2
        Hx = Hx1 + 0.3 * Hx2
        Hy = Hy1 + 0.3 * Hy2
        Hz = Hz1 + 0.3 * Hz2
    else:
        # Default: electric dipole
        Ex, Ey, Ez, Hx, Hy, Hz = _electric_dipole_fields(X, Y, Z, freq_hz, params)

    # B = μ₀ H
    Bx = MU_0 * Hx
    By = MU_0 * Hy
    Bz = MU_0 * Hz

    # Magnitudes
    E_mag = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
    H_mag = np.sqrt(np.abs(Hx)**2 + np.abs(Hy)**2 + np.abs(Hz)**2)
    B_mag = np.sqrt(np.abs(Bx)**2 + np.abs(By)**2 + np.abs(Bz)**2)

    return {
        "coord1": C1 / mm,  # back to mm for display
        "coord2": C2 / mm,
        "Ex": Ex, "Ey": Ey, "Ez": Ez,
        "Hx": Hx, "Hy": Hy, "Hz": Hz,
        "Bx": Bx, "By": By, "Bz": Bz,
        "E_mag": E_mag,
        "H_mag": H_mag,
        "B_mag": B_mag,
        "labels": labels,
        "plane": plane,
        "freq_hz": freq_hz,
    }


def _magnetic_dipole_fields(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
    freq_hz: float, params: dict,
) -> tuple:
    """
    Fields of a z-directed magnetic dipole (small loop / spiral).

    The magnetic dipole moment: m = N·I·A·ẑ
    In the near field (kr << 1):
      Hᵣ = m·cos(θ)/(2π·r³)
      Hθ = m·sin(θ)/(4π·r³)
      Eφ = -jωμ₀·m·sin(θ)/(4π·r²)
    """
    omega = 2 * math.pi * freq_hz
    k = omega / C_0

    # Estimate magnetic moment from params
    radius = params.get("radius_mm", params.get("outer_radius_mm", 10.0)) * 1e-3
    n_turns = int(params.get("num_turns", 1))
    area = math.pi * radius**2
    I = 1.0  # 1A excitation (normalized)
    m = n_turns * I * area

    # Distance and angles
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r = np.maximum(r, 1e-6)  # avoid singularity

    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(rho, Z)
    phi = np.arctan2(Y, X)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Full-wave magnetic dipole fields (near + far)
    kr = k * r
    exp_jkr = np.exp(-1j * kr)

    # H_r
    Hr = (m / (4 * math.pi)) * cos_theta * (
        2.0 / r**3 + 2j * k / r**2
    ) * exp_jkr

    # H_theta
    Ht = (m / (4 * math.pi)) * sin_theta * (
        1.0 / r**3 + 1j * k / r**2 - k**2 / r
    ) * exp_jkr

    # E_phi
    Ep = -(1j * omega * MU_0 * m / (4 * math.pi)) * sin_theta * (
        1.0 / r**2 + 1j * k / r
    ) * exp_jkr * (-1)

    # Convert spherical to Cartesian
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    Hx = Hr * sin_theta * cos_phi + Ht * cos_theta * cos_phi
    Hy = Hr * sin_theta * sin_phi + Ht * cos_theta * sin_phi
    Hz_field = Hr * cos_theta - Ht * sin_theta

    Ex = -Ep * sin_phi
    Ey = Ep * cos_phi
    Ez = np.zeros_like(Ex)

    return Ex, Ey, Ez, Hx, Hy, Hz_field


def _electric_dipole_fields(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
    freq_hz: float, params: dict,
) -> tuple:
    """
    Fields of a z-directed electric (Hertzian) dipole.

    Models meander, MIFA, and small monopole/dipole antennas.
    The effective dipole length is estimated from the antenna parameters.
    """
    omega = 2 * math.pi * freq_hz
    k = omega / C_0

    # Estimate effective dipole length
    if "total_length_mm" in params:
        dl = min(params["total_length_mm"], 50) * 1e-3
    elif "section_length_mm" in params:
        dl = params["section_length_mm"] * 1e-3
    elif "height_mm" in params:
        dl = params["height_mm"] * 1e-3
    else:
        dl = 10e-3

    I = 1.0  # 1A excitation
    p = I * dl  # electric dipole moment

    r = np.sqrt(X**2 + Y**2 + Z**2)
    r = np.maximum(r, 1e-6)

    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(rho, Z)
    phi = np.arctan2(Y, X)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    kr = k * r
    exp_jkr = np.exp(-1j * kr)

    # E_r
    Er = (p / (4 * math.pi * EPS_0)) * cos_theta * (
        2.0 / r**3 + 2j * k / r**2
    ) * exp_jkr / (1j * omega)

    # E_theta
    Et = (p / (4 * math.pi * EPS_0)) * sin_theta * (
        1.0 / r**3 + 1j * k / r**2 - k**2 / r
    ) * exp_jkr / (1j * omega)

    # H_phi
    Hp = (1j * omega * p / (4 * math.pi)) * sin_theta * (
        1.0 / r**2 + 1j * k / r
    ) * exp_jkr * (1 / (1j * omega)) * k

    # Actually let me use the standard Hertzian dipole formulas properly
    # H_phi = (j*k*p*dl)/(4*pi) * sin(theta) * (1/r + 1/(jkr^2)) * e^{-jkr}
    Hp = (1j * k * I * dl / (4 * math.pi)) * sin_theta * (
        1.0 / r + 1.0 / (1j * kr * r)
    ) * exp_jkr

    # E_r = (I*dl)/(2*pi*j*omega*eps) * cos(theta) * (1/r^3 + jk/r^2) * e^{-jkr}
    coeff = I * dl / (2 * math.pi * 1j * omega * EPS_0)
    Er = coeff * cos_theta * (1.0 / r**3 + 1j * k / r**2) * exp_jkr

    # E_theta = (I*dl)/(4*pi*j*omega*eps) * sin(theta) * (1/r^3 + jk/r^2 - k^2/r) * e^{-jkr}
    coeff2 = I * dl / (4 * math.pi * 1j * omega * EPS_0)
    Et = coeff2 * sin_theta * (1.0 / r**3 + 1j * k / r**2 - k**2 / r) * exp_jkr

    # Convert to Cartesian
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    Ex = Er * sin_theta * cos_phi + Et * cos_theta * cos_phi - 0 * sin_phi
    Ey = Er * sin_theta * sin_phi + Et * cos_theta * sin_phi + 0 * cos_phi
    Ez = Er * cos_theta - Et * sin_theta

    Hx = -Hp * sin_phi
    Hy = Hp * cos_phi
    Hz_field = np.zeros_like(Hx)

    return Ex, Ey, Ez, Hx, Hy, Hz_field


def _patch_cavity_fields(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
    freq_hz: float, params: dict,
) -> tuple:
    """
    Patch antenna cavity fields (TM10 mode under the patch).

    E_z = E0 * cos(π·x/L) under the patch
    Outside: radiating slots modeled as magnetic current sources.
    """
    L = params.get("patch_length_mm", 36) * 1e-3
    W = params.get("patch_width_mm", 44) * 1e-3
    h = params.get("substrate_height_mm", 1.6) * 1e-3

    omega = 2 * math.pi * freq_hz
    k = omega / C_0
    E0 = 1000.0  # V/m (normalized)

    # Under the patch: cavity field
    # TM10: Ez = E0 * cos(pi*x/L), for 0 < x < L, 0 < y < W, 0 < z < h
    # We map the patch centered at origin

    in_patch_x = (np.abs(X) <= L / 2)
    in_patch_y = (np.abs(Y) <= W / 2)
    under_patch = in_patch_x & in_patch_y & (Z >= 0) & (Z <= h * 2)

    # Ez under patch
    Ez = np.zeros_like(X, dtype=complex)
    Ez[under_patch] = E0 * np.cos(math.pi * X[under_patch] / L)

    # Outside: approximate as two radiating slots + fringing
    outside = ~under_patch
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r = np.maximum(r, 1e-6)
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(rho, Z)

    # Approximate far-field from slot pair
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    phi = np.arctan2(Y, X)

    # Slot radiation: two magnetic current sheets at x = ±L/2
    # Each slot has M = E0 * h in the y direction
    M = E0 * h * W
    exp_jkr = np.exp(-1j * k * r)

    # Simplified: treat as a combination of two slots
    phase_diff = k * L / 2 * np.sin(theta) * np.cos(phi)
    array_factor = 2 * np.cos(phase_diff)

    # E-field from slot pair (far-field)
    Et_far = (1j * k * M / (4 * math.pi * r)) * sin_theta * array_factor * exp_jkr

    Ez[outside] = Et_far[outside] * cos_theta[outside] * 0.01  # attenuated
    Ex = np.zeros_like(Ez)
    Ey = np.zeros_like(Ez)

    # H-field from E-field (approximate)
    Hx = -Ez / ETA_0 * np.sin(phi)
    Hy = Ez / ETA_0 * np.cos(phi)
    Hz_field = np.zeros_like(Hx)

    # Scale fields more reasonably
    Ex = Et_far * sin_theta * np.cos(phi) * 0.1

    return Ex, Ey, Ez, Hx, Hy, Hz_field


def compute_radiation_pattern_3d(
    antenna_type: str,
    freq_hz: float,
    params: dict,
    n_theta: int = 37,
    n_phi: int = 73,
) -> dict:
    """
    Compute 3D radiation pattern (gain in dBi).

    Returns pattern on a theta-phi grid suitable for 3D surface plotting.
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')

    if antenna_type in ("loop", "spiral"):
        # Magnetic dipole: G(θ) = 1.5 * sin²(θ)
        D = 1.5 * np.sin(THETA)**2
    elif antenna_type in ("meander", "mifa", "dipole"):
        # Electric dipole: G(θ) = 1.5 * sin²(θ)
        D = 1.5 * np.sin(THETA)**2
    elif antenna_type == "patch":
        # Patch: broadside, cos²(θ) pattern
        D = 6.0 * np.cos(THETA)**2
        D[THETA > np.pi / 2] = 0  # no back radiation (ground plane)
    elif antenna_type == "helix":
        h_params = params
        r = h_params.get("radius_mm", 10) * 1e-3
        C = 2 * np.pi * r
        lam = C_0 / freq_hz
        if 0.75 < C / lam < 1.33:
            # Axial mode: endfire pattern
            n_turns = int(h_params.get("num_turns", 5))
            D = 15 * n_turns * (C / lam)**2 * np.cos(THETA)**2
            D = np.maximum(D, 0.01)
        else:
            # Normal mode: omnidirectional like dipole
            D = 1.5 * np.sin(THETA)**2
    else:
        D = 1.5 * np.sin(THETA)**2

    D = np.maximum(D, 1e-6)
    gain_dbi = 10 * np.log10(D)

    # Convert to Cartesian for 3D surface plot
    R_plot = np.maximum(gain_dbi + 30, 0) / 30  # normalize for plotting
    X = R_plot * np.sin(THETA) * np.cos(PHI)
    Y = R_plot * np.sin(THETA) * np.sin(PHI)
    Z = R_plot * np.cos(THETA)

    return {
        "theta": THETA,
        "phi": PHI,
        "gain_linear": D,
        "gain_dbi": gain_dbi,
        "X": X, "Y": Y, "Z": Z,
        "R_plot": R_plot,
    }
