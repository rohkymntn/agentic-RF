"""
Material database for RF/antenna engineering.

Each material has:
  - name: human-readable name
  - eps_r: relative permittivity (real part)
  - tan_d: loss tangent (dielectric loss)
  - sigma: conductivity [S/m] (for conductors)
  - mu_r: relative permeability
  - density: [kg/m³] (for SAR calculations)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Material:
    """RF material properties (frequency-independent model)."""
    name: str
    eps_r: float = 1.0
    tan_d: float = 0.0
    sigma: float = 0.0
    mu_r: float = 1.0
    density: float = 0.0  # kg/m³, for SAR
    description: str = ""

    @property
    def is_conductor(self) -> bool:
        return self.sigma > 1e3

    @property
    def is_lossy_dielectric(self) -> bool:
        return self.tan_d > 0 and not self.is_conductor

    def effective_conductivity(self, freq_hz: float) -> float:
        """Total effective conductivity including dielectric loss: σ_eff = σ + ω·ε₀·εᵣ·tan_δ."""
        import math
        omega = 2.0 * math.pi * freq_hz
        from .constants import EPS_0
        return self.sigma + omega * EPS_0 * self.eps_r * self.tan_d


# ═══════════════════════════════════════════════════════════════════════════════
# Material Library
# ═══════════════════════════════════════════════════════════════════════════════

# --- Conductors ---
COPPER = Material("Copper", sigma=5.8e7, density=8960, description="Standard PCB trace")
ALUMINUM = Material("Aluminum", sigma=3.77e7, density=2700)
GOLD = Material("Gold", sigma=4.1e7, density=19300, description="Wire bonding, implant electrodes")
SILVER = Material("Silver", sigma=6.3e7, density=10490)
STEEL_STAINLESS = Material("Stainless Steel 316L", sigma=1.33e6, mu_r=1.0, density=8000,
                           description="Implant casings, biocompatible")
TITANIUM = Material("Titanium", sigma=2.38e6, density=4507, description="Implant grade")

# --- PCB Substrates ---
FR4 = Material("FR-4", eps_r=4.4, tan_d=0.02, density=1850,
               description="Standard PCB, lossy at high freq")
FR4_LOW_LOSS = Material("FR-4 Low Loss", eps_r=4.2, tan_d=0.01, density=1850)
ROGERS_4350B = Material("Rogers RO4350B", eps_r=3.66, tan_d=0.0037, density=1860,
                        description="Low-loss, good for RF")
ROGERS_3010 = Material("Rogers RO3010", eps_r=10.2, tan_d=0.0022, density=2800,
                       description="High-εr, antenna miniaturization")
ROGERS_5880 = Material("Rogers RT5880", eps_r=2.2, tan_d=0.0009, density=2200,
                       description="Very low loss, microwave")
ALUMINA = Material("Alumina (Al₂O₃)", eps_r=9.8, tan_d=0.0001, density=3950,
                   description="Ceramic, high Q")
LTCC = Material("LTCC", eps_r=7.8, tan_d=0.002, density=3200,
                description="Low-Temperature Co-fired Ceramic")

# --- Flexible / Wearable ---
POLYIMIDE = Material("Polyimide (Kapton)", eps_r=3.4, tan_d=0.004, density=1420,
                     description="Flex PCB substrate")
PDMS = Material("PDMS (Silicone)", eps_r=2.7, tan_d=0.01, density=970,
                description="Flexible encapsulant, biocompatible")
LCP = Material("Liquid Crystal Polymer", eps_r=3.0, tan_d=0.002, density=1400,
               description="Low moisture absorption, flex")

# --- Biological Tissue (approximate at 400 MHz) ---
MUSCLE = Material("Muscle Tissue (400 MHz)", eps_r=57.1, tan_d=0.0, sigma=0.80,
                  density=1041, description="High-loss biological tissue")
FAT = Material("Fat Tissue (400 MHz)", eps_r=11.6, tan_d=0.0, sigma=0.082,
               density=911, description="Lower-loss biological tissue")
SKIN_DRY = Material("Dry Skin (400 MHz)", eps_r=46.7, tan_d=0.0, sigma=0.69,
                    density=1109)
BONE_CORTICAL = Material("Cortical Bone (400 MHz)", eps_r=13.1, tan_d=0.0, sigma=0.09,
                         density=1908)
BLOOD = Material("Blood (400 MHz)", eps_r=64.2, tan_d=0.0, sigma=1.35,
                 density=1060)
BRAIN_GREY = Material("Grey Matter (400 MHz)", eps_r=56.8, tan_d=0.0, sigma=0.75,
                      density=1045)

# --- Encapsulants / Superstrates ---
SILICONE_MEDICAL = Material("Medical-Grade Silicone", eps_r=3.0, tan_d=0.005, density=1100,
                            description="Implant encapsulant")
PEEK = Material("PEEK", eps_r=3.2, tan_d=0.003, density=1300,
                description="Biocompatible polymer")
ZIRCONIA = Material("Zirconia", eps_r=27.0, tan_d=0.0005, density=6050,
                    description="Ceramic implant housing")

# --- Air / vacuum ---
AIR = Material("Air", eps_r=1.0006, tan_d=0.0, density=1.225)
VACUUM = Material("Vacuum", eps_r=1.0, tan_d=0.0, density=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Lookup
# ═══════════════════════════════════════════════════════════════════════════════

_LIBRARY: dict[str, Material] = {}

def _register_all():
    """Auto-register all Material instances in this module."""
    import sys
    module = sys.modules[__name__]
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, Material):
            _LIBRARY[obj.name.lower()] = obj
            _LIBRARY[name.lower()] = obj  # also register by variable name

_register_all()


def get_material(name: str) -> Material:
    """Look up a material by name (case-insensitive)."""
    key = name.lower()
    if key in _LIBRARY:
        return _LIBRARY[key]
    # fuzzy match
    candidates = [k for k in _LIBRARY if key in k]
    if candidates:
        return _LIBRARY[candidates[0]]
    raise KeyError(f"Material '{name}' not found. Available: {list_materials()}")


def list_materials() -> list[str]:
    """List all available material names."""
    seen = set()
    result = []
    for m in _LIBRARY.values():
        if m.name not in seen:
            seen.add(m.name)
            result.append(m.name)
    return sorted(result)


# ═══════════════════════════════════════════════════════════════════════════════
# Tissue layered model for implant antennas
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TissueLayer:
    """A single layer in a multi-layer tissue phantom."""
    material: Material
    thickness_mm: float  # thickness in mm (0 for outermost / infinite)

    @property
    def thickness_m(self) -> float:
        return self.thickness_mm * 1e-3


# Standard implant tissue phantoms
IMPLANT_PHANTOM_SIMPLE = [
    TissueLayer(MUSCLE, 0),  # infinite muscle (single-layer)
]

IMPLANT_PHANTOM_MULTILAYER = [
    TissueLayer(SKIN_DRY, 2.0),
    TissueLayer(FAT, 5.0),
    TissueLayer(MUSCLE, 0),  # semi-infinite
]

IMPLANT_PHANTOM_BRAIN = [
    TissueLayer(SKIN_DRY, 2.0),
    TissueLayer(BONE_CORTICAL, 7.0),
    TissueLayer(BRAIN_GREY, 0),
]
