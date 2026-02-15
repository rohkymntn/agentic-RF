"""
Agent tool definitions.

Each tool is a callable function that the LLM agent can invoke.
Tools map natural-language design decisions to concrete engineering actions.
"""

from __future__ import annotations

import json
from typing import Any

from ..geometry.templates import get_template, list_templates
from ..geometry.base import AntennaTemplate
from ..solver.analytical import AnalyticalSolver
from ..optimizer.objectives import AntennaObjective, ObjectiveConfig
from ..optimizer.bayesian import BayesianOptimizer, BayesianOptimizerConfig
from ..optimizer.cmaes import CMAESOptimizer, CMAESConfig
from ..analysis.sparams import SParameterAnalyzer
from ..analysis.impedance import ImpedanceAnalyzer
from ..analysis.safety import SARAnalyzer
from ..measurement.touchstone import TouchstoneIO
from ..measurement.calibration import ModelCalibrator
from ..utils.constants import (
    freq_to_wavelength, electrical_size, chu_limit_q,
    chu_limit_bandwidth, free_space_path_loss,
    MICS_BAND, MEDRADIO_BAND, ISM_900, ISM_2400,
)
from ..utils.materials import get_material, list_materials
from ..utils.units import mhz, ghz, mm


# ═══════════════════════════════════════════════════════════════════════════════
# Tool implementations
# ═══════════════════════════════════════════════════════════════════════════════

def tool_list_templates() -> str:
    """List all available antenna templates with descriptions."""
    templates = list_templates()
    lines = ["Available antenna templates:\n"]
    for t in templates:
        lines.append(f"  - {t['key']}: {t['name']}")
        lines.append(f"    {t['description']}")
        lines.append(f"    Parameters: {t['num_parameters']}")
        lines.append("")
    return "\n".join(lines)


def tool_get_template_info(template_name: str) -> str:
    """Get detailed info about a specific antenna template."""
    try:
        tmpl = get_template(template_name)
    except KeyError as e:
        return str(e)

    lines = [
        f"Template: {tmpl.name}",
        f"Description: {tmpl.description}",
        f"\nParameters ({len(tmpl.parameters)}):",
    ]
    for p in tmpl.parameters:
        lines.append(
            f"  - {p.name}: [{p.min_val}, {p.max_val}], default={p.default} {p.unit}"
        )
        if p.description:
            lines.append(f"    {p.description}")

    defaults = tmpl.get_default_params()
    quick = tmpl.quick_evaluate(defaults, 403.5e6)
    lines.append(f"\nDefault design quick evaluation (at 403.5 MHz):")
    for k, v in quick.items():
        lines.append(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")

    return "\n".join(lines)


def tool_quick_evaluate(template_name: str, params: dict, freq_mhz: float = 403.5) -> str:
    """Quickly evaluate an antenna design using analytical models."""
    tmpl = get_template(template_name)
    result = tmpl.quick_evaluate(params, freq_mhz * 1e6)
    lines = [f"Quick evaluation of {tmpl.name} at {freq_mhz} MHz:"]
    for k, v in result.items():
        lines.append(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")
    return "\n".join(lines)


def tool_simulate_analytical(template_name: str, params: dict,
                              freq_start_mhz: float, freq_stop_mhz: float,
                              n_points: int = 101) -> dict:
    """Run analytical simulation over frequency sweep."""
    tmpl = get_template(template_name)
    solver = AnalyticalSolver(tmpl)
    geom = tmpl.generate_geometry(tmpl.validate_params(params))
    result = solver.simulate(geom, freq_start_mhz * 1e6, freq_stop_mhz * 1e6, n_points)
    return result.summary()


def tool_optimize_antenna(
    template_name: str,
    target_freq_mhz: float = 403.5,
    band_start_mhz: float = 402.0,
    band_stop_mhz: float = 405.0,
    max_radius_mm: float = 15.0,
    n_iterations: int = 50,
    optimizer_type: str = "bayesian",
    sar_constraint: bool = False,
) -> dict:
    """
    Run multi-objective optimization on an antenna design.

    Returns best design parameters and performance metrics.
    """
    tmpl = get_template(template_name)

    config = ObjectiveConfig(
        target_freq_hz=target_freq_mhz * 1e6,
        freq_band_start_hz=band_start_mhz * 1e6,
        freq_band_stop_hz=band_stop_mhz * 1e6,
        max_radius_mm=max_radius_mm,
        sar_constraint=sar_constraint,
    )

    objective = AntennaObjective(config, tmpl)

    if optimizer_type == "bayesian":
        opt_config = BayesianOptimizerConfig(
            n_initial=10, n_iterations=n_iterations, verbose=False
        )
        optimizer = BayesianOptimizer(opt_config)
        result = optimizer.optimize(objective)
    elif optimizer_type == "cmaes":
        opt_config = CMAESConfig(max_evaluations=n_iterations * 5, verbose=False)
        optimizer = CMAESOptimizer(opt_config)
        result = optimizer.optimize(objective)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    best = result["best_result"] if isinstance(result, dict) else result.best_result

    return {
        "best_params": best.parameters,
        "performance": {
            "s11_db": best.s11_db,
            "bandwidth_hz": best.bandwidth_hz,
            "efficiency": best.efficiency,
            "radius_mm": best.radius_mm,
            "total_score": best.total_score,
            "feasible": best.feasible,
            "violations": best.constraint_violations,
        },
        "n_evaluations": result["n_evaluations"] if isinstance(result, dict) else result.n_evaluations,
        "wall_time_s": result["wall_time_seconds"] if isinstance(result, dict) else result.wall_time_seconds,
    }


def tool_analyze_sparams(filepath: str) -> dict:
    """Load and analyze a Touchstone file (S1P/S2P from VNA)."""
    result = TouchstoneIO.read(filepath)
    analyzer = SParameterAnalyzer(result)
    report = analyzer.full_report()
    artifacts = TouchstoneIO.detect_fixture_artifacts(result)
    return {**report, "fixture_analysis": artifacts}


def tool_design_matching_network(z_real: float, z_imag: float,
                                  freq_mhz: float, z0: float = 50.0) -> str:
    """Design an impedance matching network for the antenna."""
    z_load = complex(z_real, z_imag)
    analyzer = ImpedanceAnalyzer(z0)

    matches = analyzer.synthesize_l_match(z_load, freq_mhz * 1e6)
    compensate = analyzer.compensate_reactance(z_load, freq_mhz * 1e6)

    lines = [f"Matching network for Z = {z_real:+.1f}{z_imag:+.1f}j Ω at {freq_mhz} MHz:"]

    if compensate["needed"]:
        lines.append(f"\n  Reactance compensation: {compensate['note']}")

    lines.append(f"\n  L-match solutions ({len(matches)} found):")
    for m in matches:
        lines.append(f"    {m.describe()}")

    return "\n".join(lines)


def tool_sar_check(input_power_mw: float, efficiency: float,
                    freq_mhz: float, tissue: str = "muscle") -> dict:
    """Check SAR compliance for implant antenna."""
    mat = get_material(tissue)
    analyzer = SARAnalyzer()
    report = analyzer.compliance_report(
        input_power_mw * 1e-3, efficiency, freq_mhz * 1e6, mat
    )
    return report


def tool_calculate_link_budget(
    freq_mhz: float, distance_m: float,
    tx_power_dbm: float = 0, tx_gain_dbi: float = 0,
    rx_gain_dbi: float = 0, extra_loss_db: float = 0,
) -> dict:
    """Calculate wireless link budget."""
    fspl = free_space_path_loss(freq_mhz * 1e6, distance_m)
    rx_power = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - fspl - extra_loss_db

    return {
        "fspl_db": fspl,
        "rx_power_dbm": rx_power,
        "tx_power_dbm": tx_power_dbm,
        "total_gain_dbi": tx_gain_dbi + rx_gain_dbi,
        "total_loss_db": fspl + extra_loss_db,
        "link_margin_db": rx_power - (-100),  # typical sensitivity
    }


def tool_chu_limit(freq_mhz: float, radius_mm: float) -> dict:
    """Calculate Chu-Harrington fundamental limit for given size and frequency."""
    ka = electrical_size(freq_mhz * 1e6, radius_mm * 1e-3)
    q = chu_limit_q(ka)
    bw = chu_limit_bandwidth(ka, 2.0)
    lam = freq_to_wavelength(freq_mhz * 1e6) * 1e3  # in mm

    return {
        "ka": ka,
        "q_min": q,
        "max_fractional_bw_pct": bw * 100,
        "max_bw_mhz": bw * freq_mhz,
        "wavelength_mm": lam,
        "electrical_size_description": (
            "electrically small (ka < 0.5)" if ka < 0.5
            else "moderate (0.5 < ka < 1)" if ka < 1
            else "electrically large (ka > 1)"
        ),
    }


def tool_list_materials() -> str:
    """List all available RF materials."""
    materials = list_materials()
    return "Available materials:\n" + "\n".join(f"  - {m}" for m in materials)


def tool_material_info(name: str) -> str:
    """Get detailed properties of a material."""
    mat = get_material(name)
    return (
        f"Material: {mat.name}\n"
        f"  εr = {mat.eps_r}, tan_δ = {mat.tan_d}\n"
        f"  σ = {mat.sigma} S/m, μr = {mat.mu_r}\n"
        f"  ρ = {mat.density} kg/m³\n"
        f"  {mat.description}"
    )


def tool_wpt_coil_design(
    tx_radius_mm: float, rx_radius_mm: float,
    tx_turns: int, rx_turns: int,
    distance_mm: float, freq_mhz: float,
) -> dict:
    """Design and evaluate a wireless power transfer coil pair."""
    from ..geometry.templates.spiral import SpiralAntennaTemplate

    spiral = SpiralAntennaTemplate()

    tx_params = spiral.validate_params({
        "inner_radius_mm": tx_radius_mm * 0.3,
        "outer_radius_mm": tx_radius_mm,
        "num_turns": tx_turns,
        "trace_width_mm": 1.0,
        "trace_gap_mm": 0.5,
    })
    rx_params = spiral.validate_params({
        "inner_radius_mm": rx_radius_mm * 0.3,
        "outer_radius_mm": rx_radius_mm,
        "num_turns": rx_turns,
        "trace_width_mm": 0.5,
        "trace_gap_mm": 0.3,
    })

    k = spiral.coupling_coefficient(tx_params, rx_params, distance_mm * 1e-3)
    eta = spiral.wpt_link_efficiency(tx_params, rx_params, distance_mm * 1e-3, freq_mhz * 1e6)
    L_tx = spiral.estimate_inductance_nh(tx_params)
    L_rx = spiral.estimate_inductance_nh(rx_params)
    Q_tx = spiral.estimate_q_factor(tx_params, freq_mhz * 1e6)
    Q_rx = spiral.estimate_q_factor(rx_params, freq_mhz * 1e6)

    return {
        "coupling_k": k,
        "link_efficiency_pct": eta * 100,
        "tx_inductance_nh": L_tx,
        "rx_inductance_nh": L_rx,
        "tx_q_factor": Q_tx,
        "rx_q_factor": Q_rx,
        "kQ_product": k * (Q_tx * Q_rx) ** 0.5,
        "notes": (
            f"k={k:.4f}, kQ={k * (Q_tx*Q_rx)**0.5:.1f}. "
            f"For good WPT efficiency, need kQ >> 1. "
            f"{'Good link' if k * (Q_tx*Q_rx)**0.5 > 1 else 'Weak link - increase coil size, turns, or Q'}."
        ),
    }


def tool_calibrate_model(sim_s1p_path: str, meas_s1p_path: str) -> dict:
    """Calibrate simulation model against measurement data."""
    sim = TouchstoneIO.read(sim_s1p_path)
    meas = TouchstoneIO.read(meas_s1p_path)

    calibrator = ModelCalibrator()
    cal = calibrator.calibrate(sim, meas)
    model = calibrator.build_reality_model(cal, {})

    return {
        "freq_shift_pct": cal.freq_error_pct,
        "er_correction": cal.er_correction,
        "loss_correction": cal.loss_correction,
        "rms_error_db": cal.rms_error_db,
        "correlation": cal.correlation,
        "reality_model": {
            "effective_er": model.effective_er,
            "effective_tan_d": model.effective_tan_d,
            "freq_offset_hz": model.freq_offset_hz,
        },
        "notes": cal.notes,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool registry for the LLM agent
# ═══════════════════════════════════════════════════════════════════════════════

AGENT_TOOLS = {
    "list_templates": {
        "function": tool_list_templates,
        "description": "List all available parametric antenna templates (MIFA, PIFA, loop, patch, helix, meander, spiral)",
        "parameters": {},
    },
    "get_template_info": {
        "function": tool_get_template_info,
        "description": "Get detailed information about a specific antenna template including parameters and default performance",
        "parameters": {
            "template_name": {"type": "string", "description": "Template key (e.g., 'mifa', 'pifa', 'loop')"},
        },
    },
    "quick_evaluate": {
        "function": tool_quick_evaluate,
        "description": "Quickly evaluate an antenna design at a specific frequency using analytical models",
        "parameters": {
            "template_name": {"type": "string"},
            "params": {"type": "object", "description": "Design parameter values"},
            "freq_mhz": {"type": "number", "description": "Frequency in MHz"},
        },
    },
    "simulate": {
        "function": tool_simulate_analytical,
        "description": "Run analytical frequency-sweep simulation of an antenna",
        "parameters": {
            "template_name": {"type": "string"},
            "params": {"type": "object"},
            "freq_start_mhz": {"type": "number"},
            "freq_stop_mhz": {"type": "number"},
            "n_points": {"type": "integer", "default": 101},
        },
    },
    "optimize": {
        "function": tool_optimize_antenna,
        "description": "Run multi-objective optimization to find the best antenna design for given constraints",
        "parameters": {
            "template_name": {"type": "string"},
            "target_freq_mhz": {"type": "number", "default": 403.5},
            "band_start_mhz": {"type": "number", "default": 402},
            "band_stop_mhz": {"type": "number", "default": 405},
            "max_radius_mm": {"type": "number", "default": 15},
            "n_iterations": {"type": "integer", "default": 50},
            "optimizer_type": {"type": "string", "default": "bayesian"},
            "sar_constraint": {"type": "boolean", "default": False},
        },
    },
    "analyze_measurement": {
        "function": tool_analyze_sparams,
        "description": "Load and analyze VNA measurement data from a Touchstone (S1P/S2P) file",
        "parameters": {
            "filepath": {"type": "string", "description": "Path to .s1p or .s2p file"},
        },
    },
    "design_matching_network": {
        "function": tool_design_matching_network,
        "description": "Design an impedance matching network for the antenna",
        "parameters": {
            "z_real": {"type": "number", "description": "Real part of antenna impedance (Ω)"},
            "z_imag": {"type": "number", "description": "Imaginary part of antenna impedance (Ω)"},
            "freq_mhz": {"type": "number"},
            "z0": {"type": "number", "default": 50},
        },
    },
    "sar_check": {
        "function": tool_sar_check,
        "description": "Check SAR safety compliance for implant/body-worn antenna",
        "parameters": {
            "input_power_mw": {"type": "number"},
            "efficiency": {"type": "number", "description": "Radiation efficiency (0-1)"},
            "freq_mhz": {"type": "number"},
            "tissue": {"type": "string", "default": "muscle"},
        },
    },
    "chu_limit": {
        "function": tool_chu_limit,
        "description": "Calculate Chu-Harrington fundamental size-bandwidth limit",
        "parameters": {
            "freq_mhz": {"type": "number"},
            "radius_mm": {"type": "number", "description": "Radius of smallest enclosing sphere"},
        },
    },
    "link_budget": {
        "function": tool_calculate_link_budget,
        "description": "Calculate wireless link power budget",
        "parameters": {
            "freq_mhz": {"type": "number"},
            "distance_m": {"type": "number"},
            "tx_power_dbm": {"type": "number", "default": 0},
            "tx_gain_dbi": {"type": "number", "default": 0},
            "rx_gain_dbi": {"type": "number", "default": 0},
            "extra_loss_db": {"type": "number", "default": 0},
        },
    },
    "list_materials": {
        "function": tool_list_materials,
        "description": "List all available RF materials in the database",
        "parameters": {},
    },
    "material_info": {
        "function": tool_material_info,
        "description": "Get detailed properties of a specific RF material",
        "parameters": {
            "name": {"type": "string", "description": "Material name"},
        },
    },
    "wpt_coil_design": {
        "function": tool_wpt_coil_design,
        "description": "Design and evaluate a wireless power transfer (WPT) coil pair",
        "parameters": {
            "tx_radius_mm": {"type": "number"},
            "rx_radius_mm": {"type": "number"},
            "tx_turns": {"type": "integer"},
            "rx_turns": {"type": "integer"},
            "distance_mm": {"type": "number"},
            "freq_mhz": {"type": "number"},
        },
    },
    "calibrate_model": {
        "function": tool_calibrate_model,
        "description": "Calibrate simulation model against VNA measurement to close sim-reality gap",
        "parameters": {
            "sim_s1p_path": {"type": "string"},
            "meas_s1p_path": {"type": "string"},
        },
    },
}


def get_tools_for_llm(provider: str = "openai") -> list[dict]:
    """
    Generate tool definitions in the format expected by LLM APIs.

    Args:
        provider: "openai" or "anthropic"

    Returns:
        List of tool definition dicts
    """
    tools = []
    for name, tool in AGENT_TOOLS.items():
        if provider == "openai":
            properties = {}
            required = []
            for pname, pinfo in tool["parameters"].items():
                prop = {"type": pinfo.get("type", "string")}
                if "description" in pinfo:
                    prop["description"] = pinfo["description"]
                properties[pname] = prop
                if "default" not in pinfo:
                    required.append(pname)

            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        elif provider == "anthropic":
            properties = {}
            required = []
            for pname, pinfo in tool["parameters"].items():
                prop = {"type": pinfo.get("type", "string")}
                if "description" in pinfo:
                    prop["description"] = pinfo["description"]
                properties[pname] = prop
                if "default" not in pinfo:
                    required.append(pname)

            tools.append({
                "name": name,
                "description": tool["description"],
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })

    return tools


def execute_tool(tool_name: str, arguments: dict) -> Any:
    """Execute a tool by name with given arguments."""
    if tool_name not in AGENT_TOOLS:
        return f"Error: Unknown tool '{tool_name}'. Available: {list(AGENT_TOOLS.keys())}"

    tool = AGENT_TOOLS[tool_name]
    func = tool["function"]

    # Apply defaults for missing optional parameters
    for pname, pinfo in tool["parameters"].items():
        if pname not in arguments and "default" in pinfo:
            arguments[pname] = pinfo["default"]

    try:
        result = func(**arguments)
        return result
    except Exception as e:
        return f"Error executing {tool_name}: {type(e).__name__}: {str(e)}"
