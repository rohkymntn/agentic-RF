"""
Physical constants for RF/antenna engineering.

All values in SI units unless otherwise noted.
"""

import math

# ─── Fundamental Constants ───────────────────────────────────────────────────
C_0 = 299_792_458.0            # Speed of light in vacuum [m/s]
MU_0 = 4.0 * math.pi * 1e-7   # Permeability of free space [H/m]
EPS_0 = 1.0 / (MU_0 * C_0**2) # Permittivity of free space [F/m]
ETA_0 = MU_0 * C_0             # Impedance of free space ≈ 376.73 Ω
BOLTZMANN = 1.380649e-23       # Boltzmann constant [J/K]
T_0 = 290.0                    # Standard noise temperature [K]

# ─── Common Characteristic Impedances ────────────────────────────────────────
Z0_DEFAULT = 50.0              # Standard system impedance [Ω]
Z0_75 = 75.0                   # 75 Ω systems (cable TV, etc.)

# ─── Frequency Bands of Interest ─────────────────────────────────────────────
# Medical / implant bands
MICS_BAND = (402e6, 405e6)              # Medical Implant Communication Service
MEDRADIO_BAND = (401e6, 406e6)          # MedRadio (broader allocation)
MEDRADIO_WBAN = (2360e6, 2400e6)        # MedRadio WBAN extension
ISM_433 = (433.05e6, 434.79e6)          # ISM 433 MHz (Europe)

# ISM bands
ISM_900 = (902e6, 928e6)               # ISM 900 MHz (Americas)
ISM_2400 = (2400e6, 2483.5e6)          # ISM 2.4 GHz (WiFi/BT)
ISM_5800 = (5725e6, 5875e6)            # ISM 5.8 GHz

# Wireless power transfer
WPT_QI_FREQ = (87e3, 205e3)            # Qi standard (low-freq inductive)
WPT_AIRFUEL_FREQ = 6.78e6              # AirFuel resonant (6.78 MHz)
WPT_A4WP_FREQ = 6.78e6                 # Alliance for Wireless Power


# ─── Derived Helpers ─────────────────────────────────────────────────────────

def freq_to_wavelength(freq_hz: float, er: float = 1.0) -> float:
    """Convert frequency [Hz] to wavelength [m] in medium with relative permittivity er."""
    return C_0 / (freq_hz * math.sqrt(er))


def wavelength_to_freq(wavelength_m: float, er: float = 1.0) -> float:
    """Convert wavelength [m] to frequency [Hz]."""
    return C_0 / (wavelength_m * math.sqrt(er))


def electrical_size(freq_hz: float, physical_size_m: float) -> float:
    """Return ka — the electrical size of an antenna (k = 2π/λ, a = radius of enclosing sphere)."""
    lam = freq_to_wavelength(freq_hz)
    k = 2.0 * math.pi / lam
    return k * physical_size_m


def chu_limit_q(ka: float) -> float:
    """
    Chu–Harrington minimum Q for a linearly-polarized antenna.

    Q_min = 1/(ka)^3 + 1/(ka)

    Lower Q → wider bandwidth.  For ka << 1, Q grows as 1/(ka)^3.
    """
    if ka <= 0:
        raise ValueError("ka must be positive")
    return 1.0 / ka**3 + 1.0 / ka


def chu_limit_bandwidth(ka: float, vswr_max: float = 2.0) -> float:
    """
    Maximum fractional bandwidth for an electrically small antenna (Chu limit).

    FBW ≈ (VSWR - 1) / (Q_min * sqrt(VSWR))
    """
    q = chu_limit_q(ka)
    return (vswr_max - 1.0) / (q * math.sqrt(vswr_max))


def skin_depth(freq_hz: float, sigma: float, mu_r: float = 1.0) -> float:
    """Skin depth [m] for a conductor at given frequency."""
    omega = 2.0 * math.pi * freq_hz
    return math.sqrt(2.0 / (omega * mu_r * MU_0 * sigma))


def free_space_path_loss(freq_hz: float, distance_m: float) -> float:
    """Free-space path loss in dB (Friis)."""
    lam = freq_to_wavelength(freq_hz)
    return 20.0 * math.log10(4.0 * math.pi * distance_m / lam)
