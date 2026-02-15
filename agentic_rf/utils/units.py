"""
Unit conversion utilities for RF engineering.
"""

import math


# ─── Power / amplitude conversions ──────────────────────────────────────────

def db_to_linear(db: float) -> float:
    """Convert dB (power) to linear ratio."""
    return 10.0 ** (db / 10.0)


def linear_to_db(ratio: float) -> float:
    """Convert linear power ratio to dB."""
    if ratio <= 0:
        return -float('inf')
    return 10.0 * math.log10(ratio)


def db_to_mag(db: float) -> float:
    """Convert dB to voltage magnitude (20*log10 convention)."""
    return 10.0 ** (db / 20.0)


def mag_to_db(mag: float) -> float:
    """Convert voltage magnitude to dB (20*log10 convention)."""
    if mag <= 0:
        return -float('inf')
    return 20.0 * math.log10(mag)


def dbm_to_watts(dbm: float) -> float:
    """Convert dBm to Watts."""
    return 10.0 ** ((dbm - 30.0) / 10.0)


def watts_to_dbm(watts: float) -> float:
    """Convert Watts to dBm."""
    if watts <= 0:
        return -float('inf')
    return 10.0 * math.log10(watts) + 30.0


# ─── S-parameter conversions ────────────────────────────────────────────────

def s11_to_vswr(s11_mag: float) -> float:
    """Convert |S11| (linear magnitude) to VSWR."""
    if s11_mag >= 1.0:
        return float('inf')
    if s11_mag < 0:
        s11_mag = abs(s11_mag)
    return (1.0 + s11_mag) / (1.0 - s11_mag)


def vswr_to_s11(vswr: float) -> float:
    """Convert VSWR to |S11| (linear magnitude)."""
    if vswr < 1.0:
        raise ValueError("VSWR must be >= 1.0")
    return (vswr - 1.0) / (vswr + 1.0)


def s11_to_return_loss(s11_mag: float) -> float:
    """Convert |S11| to return loss in dB (positive value)."""
    return -mag_to_db(s11_mag)


def s11_to_mismatch_loss(s11_mag: float) -> float:
    """Mismatch loss in dB: how much power is lost due to reflection."""
    return -10.0 * math.log10(1.0 - s11_mag**2)


# ─── Impedance conversions ──────────────────────────────────────────────────

def z_to_gamma(z: complex, z0: float = 50.0) -> complex:
    """Impedance to reflection coefficient."""
    return (z - z0) / (z + z0)


def gamma_to_z(gamma: complex, z0: float = 50.0) -> complex:
    """Reflection coefficient to impedance."""
    if abs(1.0 - gamma) < 1e-12:
        return complex(float('inf'), float('inf'))
    return z0 * (1.0 + gamma) / (1.0 - gamma)


def z_to_s11_db(z: complex, z0: float = 50.0) -> float:
    """Impedance to S11 in dB."""
    gamma = z_to_gamma(z, z0)
    return mag_to_db(abs(gamma))


# ─── Frequency / wavelength ─────────────────────────────────────────────────

def mhz(val: float) -> float:
    """Convert MHz to Hz."""
    return val * 1e6


def ghz(val: float) -> float:
    """Convert GHz to Hz."""
    return val * 1e9


def khz(val: float) -> float:
    """Convert kHz to Hz."""
    return val * 1e3


def mm(val: float) -> float:
    """Convert mm to meters."""
    return val * 1e-3


def mil(val: float) -> float:
    """Convert mils (thousandths of an inch) to meters."""
    return val * 25.4e-6


def oz_to_m(oz: float) -> float:
    """Convert copper weight (oz/ft²) to thickness in meters. 1 oz ≈ 35 μm."""
    return oz * 35e-6
