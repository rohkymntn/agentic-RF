"""
Parametric antenna template library.

Each module implements one antenna family with:
  - Parameterized geometry generation
  - Analytical resonance/impedance estimates
  - Manufacturing constraints
"""

from .mifa import MIFATemplate
from .pifa import PIFATemplate
from .loop import LoopAntennaTemplate
from .patch import PatchAntennaTemplate
from .helix import HelixAntennaTemplate
from .meander import MeanderAntennaTemplate
from .spiral import SpiralAntennaTemplate

# Registry of all templates
TEMPLATE_REGISTRY: dict[str, type] = {
    "mifa": MIFATemplate,
    "pifa": PIFATemplate,
    "loop": LoopAntennaTemplate,
    "patch": PatchAntennaTemplate,
    "helix": HelixAntennaTemplate,
    "meander": MeanderAntennaTemplate,
    "spiral": SpiralAntennaTemplate,
}


def get_template(name: str):
    """Instantiate a template by name."""
    key = name.lower().strip()
    if key not in TEMPLATE_REGISTRY:
        raise KeyError(
            f"Unknown template '{name}'. Available: {list(TEMPLATE_REGISTRY.keys())}"
        )
    return TEMPLATE_REGISTRY[key]()


def list_templates() -> list[dict[str, str]]:
    """List all available templates with descriptions."""
    result = []
    for key, cls in TEMPLATE_REGISTRY.items():
        inst = cls()
        result.append({
            "key": key,
            "name": inst.name,
            "description": inst.description,
            "num_parameters": len(inst.parameters),
        })
    return result
