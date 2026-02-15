#!/usr/bin/env python3
"""
Agentic RF — Interactive Web Frontend

Launch: streamlit run frontend.py
"""

import json
import io
import time
from pathlib import Path

import numpy as np
import streamlit as st

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RF — Antenna Design Studio",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Imports (lazy, after page config) ──────────────────────────────────────
from agentic_rf.geometry.templates import get_template, list_templates, TEMPLATE_REGISTRY
from agentic_rf.geometry.base import AntennaTemplate, AntennaGeometry
from agentic_rf.solver.analytical import AnalyticalSolver
from agentic_rf.solver.base import SimulationResult
from agentic_rf.optimizer.objectives import AntennaObjective, ObjectiveConfig
from agentic_rf.optimizer.cmaes import CMAESOptimizer, CMAESConfig
from agentic_rf.optimizer.bayesian import BayesianOptimizer, BayesianOptimizerConfig
from agentic_rf.analysis.sparams import SParameterAnalyzer
from agentic_rf.analysis.impedance import ImpedanceAnalyzer, MatchingNetwork
from agentic_rf.analysis.safety import SARAnalyzer
from agentic_rf.measurement.touchstone import TouchstoneIO
from agentic_rf.measurement.calibration import ModelCalibrator
from agentic_rf.visualization.plotly_viz import (
    plot_geometry_3d, plot_s11_interactive, plot_vswr_interactive,
    plot_impedance_interactive, plot_smith_interactive,
    plot_field_heatmap, plot_field_arrows, plot_radiation_3d,
    plot_convergence_interactive, plot_pareto_interactive,
)
from agentic_rf.visualization.fields import compute_near_fields, compute_radiation_pattern_3d
from agentic_rf.utils.constants import (
    freq_to_wavelength, electrical_size, chu_limit_q, chu_limit_bandwidth,
    MICS_BAND, MEDRADIO_BAND, ISM_900, ISM_2400,
)
from agentic_rf.utils.materials import list_materials, get_material


# ═══════════════════════════════════════════════════════════════════════════════
# Session State Initialization
# ═══════════════════════════════════════════════════════════════════════════════

def init_session():
    """Initialize session state for design iteration tracking."""
    defaults = {
        "design_history": [],
        "current_design_idx": 0,
        "sim_result": None,
        "opt_result": None,
        "measurement_result": None,
        "iteration_notes": [],
        "feedback_log": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


def save_design_snapshot(template_name, params, notes=""):
    """Save current design to history."""
    snapshot = {
        "template": template_name,
        "params": dict(params),
        "notes": notes,
        "timestamp": time.strftime("%H:%M:%S"),
        "iteration": len(st.session_state.design_history) + 1,
    }
    # Add sim results if available
    if st.session_state.sim_result is not None:
        s = st.session_state.sim_result
        snapshot["metrics"] = s.summary()
    st.session_state.design_history.append(snapshot)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Design Specifications (always visible)
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/color/96/antenna.png", width=48)
    st.title("Agentic RF")
    st.caption("AI Antenna Design Studio")

    st.divider()
    st.subheader("📋 Design Specifications")

    # ── Application Type ─────────────────────────────────────────────────
    application = st.selectbox(
        "Application",
        ["Implant (MICS/MedRadio)", "IoT / Wireless Sensor", "Wearable",
         "Wireless Power Transfer", "Mobile / Handset", "Custom"],
        help="Determines default frequency, constraints, and safety requirements",
    )

    # ── Frequency Band ───────────────────────────────────────────────────
    band_presets = {
        "MICS (402–405 MHz)": (402.0, 405.0, 403.5),
        "MedRadio (401–406 MHz)": (401.0, 406.0, 403.5),
        "ISM 433 MHz": (433.05, 434.79, 433.92),
        "ISM 900 MHz": (902.0, 928.0, 915.0),
        "ISM 2.4 GHz": (2400.0, 2483.5, 2440.0),
        "ISM 5.8 GHz": (5725.0, 5875.0, 5800.0),
        "Qi WPT (6.78 MHz)": (6.0, 7.5, 6.78),
        "Custom": (100.0, 1000.0, 500.0),
    }

    # Auto-select band from application
    default_band_idx = 0
    if "IoT" in application:
        default_band_idx = 4  # 2.4 GHz
    elif "Wearable" in application:
        default_band_idx = 4
    elif "Power" in application:
        default_band_idx = 6  # Qi WPT
    elif "Mobile" in application:
        default_band_idx = 4

    band_choice = st.selectbox("Frequency Band", list(band_presets.keys()),
                                index=default_band_idx)
    f_start_default, f_stop_default, f_center_default = band_presets[band_choice]

    col1, col2 = st.columns(2)
    with col1:
        freq_start = st.number_input("Start (MHz)", value=f_start_default,
                                      min_value=0.01, format="%.2f")
    with col2:
        freq_stop = st.number_input("Stop (MHz)", value=f_stop_default,
                                     min_value=0.01, format="%.2f")
    target_freq = st.number_input("Target (MHz)", value=f_center_default,
                                   min_value=0.01, format="%.2f")

    # ── Size Constraints ─────────────────────────────────────────────────
    st.subheader("📐 Size Constraints")
    max_radius = st.slider("Max enclosing radius (mm)", 1.0, 100.0, 15.0, 0.5)

    # ── Environment ──────────────────────────────────────────────────────
    st.subheader("🌍 Environment")
    environment = st.selectbox(
        "Operating environment",
        ["Free Space", "On-Body Surface", "Implanted (Muscle)",
         "Implanted (Brain)", "Inside Plastic Enclosure", "Near Metal"],
    )

    # ── Performance Goals ────────────────────────────────────────────────
    st.subheader("🎯 Performance Goals")
    s11_threshold = st.slider("S11 threshold (dB)", -20.0, -3.0, -10.0, 0.5)
    min_bw = st.number_input("Min bandwidth (MHz)", value=3.0, min_value=0.1, format="%.1f")
    min_efficiency = st.slider("Min efficiency (%)", 1, 100, 10) / 100.0

    # ── Safety ───────────────────────────────────────────────────────────
    sar_enabled = "Implant" in application
    if sar_enabled:
        st.subheader("⚠️ Safety")
        max_power_mw = st.number_input("Max input power (mW)", value=25.0,
                                        min_value=0.1, format="%.1f")
        st.info("SAR/thermal analysis enabled for implant application")

    st.divider()

    # ── Chu Limit quick check ────────────────────────────────────────────
    st.subheader("📏 Fundamental Limit")
    ka = electrical_size(target_freq * 1e6, max_radius * 1e-3)
    if ka > 0:
        q_min = chu_limit_q(ka)
        max_fbw = chu_limit_bandwidth(ka, 2.0)
        max_bw_mhz = max_fbw * target_freq

        st.metric("ka (electrical size)", f"{ka:.4f}")
        st.metric("Q_min (Chu)", f"{q_min:.1f}")
        st.metric("Max BW (Chu limit)", f"{max_bw_mhz:.2f} MHz")

        if ka < 0.5:
            st.warning(f"⚡ Electrically small antenna (ka={ka:.3f} < 0.5). "
                       f"Bandwidth severely limited by physics.")
        if max_bw_mhz < min_bw:
            st.error(f"🚫 Requested BW ({min_bw:.1f} MHz) exceeds Chu limit "
                     f"({max_bw_mhz:.2f} MHz). Increase size or reduce BW requirement.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — Tabbed Interface
# ═══════════════════════════════════════════════════════════════════════════════

tab_design, tab_geometry, tab_simulate, tab_fields, tab_optimize, \
    tab_matching, tab_safety, tab_measure, tab_history = st.tabs([
    "🔧 Design",
    "📐 Geometry",
    "📊 Simulate",
    "🧲 Fields",
    "🎯 Optimize",
    "⚡ Matching",
    "🛡️ Safety",
    "📏 Measure",
    "📝 History",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Design — Template Selection & Parameter Editor
# ═══════════════════════════════════════════════════════════════════════════════

with tab_design:
    st.header("Antenna Template & Parameters")

    # Template selection
    templates = list_templates()
    template_keys = [t["key"] for t in templates]
    template_names = [t["name"] for t in templates]
    template_descs = [t["description"] for t in templates]

    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        selected_idx = st.radio(
            "Select antenna type",
            range(len(templates)),
            format_func=lambda i: f"{template_names[i]}",
            help="Choose the antenna topology best suited for your application",
        )

    template_key = template_keys[selected_idx]
    template = get_template(template_key)

    with col_info:
        st.markdown(f"**{template.name}**")
        st.write(template.description)

        # Suitability indicator
        suitability = {
            "mifa": {"Implant": "⭐⭐⭐", "IoT": "⭐⭐⭐⭐", "Wearable": "⭐⭐⭐⭐", "WPT": "⭐"},
            "pifa": {"Implant": "⭐⭐", "IoT": "⭐⭐⭐", "Wearable": "⭐⭐⭐", "WPT": "⭐", "Mobile": "⭐⭐⭐⭐⭐"},
            "loop": {"Implant": "⭐⭐⭐⭐", "IoT": "⭐⭐", "WPT": "⭐⭐⭐⭐⭐"},
            "patch": {"Implant": "⭐", "IoT": "⭐⭐⭐", "Mobile": "⭐⭐"},
            "helix": {"Implant": "⭐⭐⭐⭐", "IoT": "⭐⭐", "WPT": "⭐⭐"},
            "meander": {"Implant": "⭐⭐⭐⭐⭐", "IoT": "⭐⭐⭐⭐", "Wearable": "⭐⭐⭐"},
            "spiral": {"Implant": "⭐⭐⭐", "IoT": "⭐⭐", "WPT": "⭐⭐⭐⭐⭐"},
        }
        suits = suitability.get(template_key, {})
        if suits:
            st.markdown("**Suitability:** " + " | ".join(f"{k}: {v}" for k, v in suits.items()))

    st.divider()

    # ── Parameter Editor ─────────────────────────────────────────────────
    st.subheader("Parameter Editor")
    st.caption("Adjust parameters below. Changes update the geometry and quick evaluation in real time.")

    params = {}
    n_cols = 3
    param_list = template.parameters
    cols = st.columns(n_cols)

    for i, p in enumerate(param_list):
        with cols[i % n_cols]:
            if p.is_integer:
                val = st.slider(
                    f"{p.name}",
                    int(p.min_val), int(p.max_val), int(p.default),
                    help=p.description,
                    key=f"param_{template_key}_{p.name}",
                )
            else:
                val = st.slider(
                    f"{p.name} ({p.unit})" if p.unit else p.name,
                    float(p.min_val), float(p.max_val), float(p.default),
                    step=(p.max_val - p.min_val) / 100,
                    help=p.description,
                    key=f"param_{template_key}_{p.name}",
                )
            params[p.name] = val

    st.divider()

    # ── Quick Evaluation ─────────────────────────────────────────────────
    st.subheader("Quick Analytical Evaluation")

    quick = template.quick_evaluate(params, target_freq * 1e6)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Resonance", f"{quick['f_res_mhz']:.1f} MHz",
                   delta=f"{quick['f_res_mhz'] - target_freq:.1f} MHz from target")
    with col_m2:
        s11_val = quick['s11_db']
        st.metric("S11 @ target", f"{s11_val:.1f} dB",
                   delta="Good" if s11_val < s11_threshold else "Poor")
    with col_m3:
        st.metric("VSWR", f"{quick['vswr']:.2f}" if quick['vswr'] < 100 else ">100")
    with col_m4:
        st.metric("Size (radius)", f"{quick['enclosing_radius_mm']:.1f} mm")

    col_m5, col_m6, col_m7, col_m8 = st.columns(4)
    with col_m5:
        st.metric("ka", f"{quick['ka']:.4f}")
    with col_m6:
        st.metric("Q_min", f"{quick['q_min']:.1f}")
    with col_m7:
        st.metric("Max BW (Chu)", f"{quick['bandwidth_pct']:.3f}%")
    with col_m8:
        st.metric("Z_in", f"{quick['z_real']:.1f} + j{quick['z_imag']:.1f} Ω")

    # ── User Feedback ────────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Design Feedback")
    feedback = st.text_area(
        "Notes / feedback for this iteration",
        placeholder="e.g., 'Resonance too high — need more length. Try increasing num_sections.'",
        key="design_feedback",
    )

    if st.button("💾 Save Design Snapshot", type="primary"):
        save_design_snapshot(template_key, params, feedback)
        st.success(f"Design #{len(st.session_state.design_history)} saved!")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Geometry — Interactive 3D View
# ═══════════════════════════════════════════════════════════════════════════════

with tab_geometry:
    st.header("3D Antenna Geometry")

    validated_params = template.validate_params(params)
    geometry = template.generate_geometry(validated_params)
    geometry.compute_bounding_box()

    col_3d, col_info3d = st.columns([3, 1])

    with col_3d:
        fig_3d = plot_geometry_3d(geometry, title=f"{template.name} — 3D View")
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_info3d:
        st.subheader("Geometry Info")
        bb = geometry.bounding_box
        if bb:
            dx = (bb[1].x - bb[0].x) * 1e3
            dy = (bb[1].y - bb[0].y) * 1e3
            dz = (bb[1].z - bb[0].z) * 1e3
            st.metric("Bounding box X", f"{dx:.1f} mm")
            st.metric("Bounding box Y", f"{dy:.1f} mm")
            st.metric("Bounding box Z", f"{dz:.1f} mm")
        st.metric("Enclosing radius", f"{geometry.enclosing_radius * 1e3:.1f} mm")
        st.metric("Traces", str(len(geometry.traces)))
        st.metric("Boxes", str(len(geometry.boxes)))
        st.metric("Ports", str(len(geometry.ports)))

        lam = freq_to_wavelength(target_freq * 1e6) * 1e3
        st.metric("λ at target", f"{lam:.1f} mm")
        st.metric("Size / λ", f"{geometry.enclosing_radius * 1e3 / lam:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Simulate — Frequency Sweep
# ═══════════════════════════════════════════════════════════════════════════════

with tab_simulate:
    st.header("Simulation")

    col_sim_ctrl, col_sim_opt = st.columns([1, 1])
    with col_sim_ctrl:
        sim_start = st.number_input("Sweep start (MHz)", value=freq_start * 0.8,
                                     min_value=0.01, format="%.2f", key="sim_start")
        sim_stop = st.number_input("Sweep stop (MHz)", value=freq_stop * 1.2,
                                    min_value=0.01, format="%.2f", key="sim_stop")
        sim_points = st.slider("Frequency points", 51, 501, 201, key="sim_points")
    with col_sim_opt:
        solver_choice = st.selectbox("Solver", ["Analytical (fast)", "OpenEMS (FDTD)", "NEC2 (MoM)"])
        if solver_choice != "Analytical (fast)":
            st.warning("Full-wave solvers require external installation. Using analytical.")

    run_sim = st.button("▶️ Run Simulation", type="primary", key="run_sim")

    if run_sim:
        with st.spinner("Running simulation..."):
            solver = AnalyticalSolver(template)
            geom = template.generate_geometry(template.validate_params(params))
            result = solver.simulate(geom, sim_start * 1e6, sim_stop * 1e6, sim_points)
            st.session_state.sim_result = result

    result = st.session_state.sim_result
    if result is not None:
        # Summary metrics
        summary = result.summary()
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Resonance", f"{summary['resonance_mhz']:.1f} MHz" if summary['resonance_mhz'] else "N/A")
        with mc2:
            st.metric("Min S11", f"{summary['min_s11_db']:.1f} dB" if summary['min_s11_db'] else "N/A")
        with mc3:
            st.metric("BW (-10 dB)", f"{summary['bandwidth_mhz']:.2f} MHz")
        with mc4:
            st.metric("Solve time", f"{summary['solve_time_s']*1000:.1f} ms")

        # Plots in 2x2 grid
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_s11 = plot_s11_interactive(
                result,
                target_band=(freq_start * 1e6, freq_stop * 1e6),
            )
            st.plotly_chart(fig_s11, use_container_width=True)

            fig_smith = plot_smith_interactive(result)
            st.plotly_chart(fig_smith, use_container_width=True)

        with col_p2:
            fig_vswr = plot_vswr_interactive(result)
            st.plotly_chart(fig_vswr, use_container_width=True)

            fig_z = plot_impedance_interactive(result)
            st.plotly_chart(fig_z, use_container_width=True)

        # Band compliance check
        st.subheader("Band Compliance")
        analyzer = SParameterAnalyzer(result)
        compliance = analyzer.check_band_coverage(freq_start * 1e6, freq_stop * 1e6, s11_threshold)
        if compliance.get("covered"):
            st.success(f"✅ Band covered! {compliance['coverage_pct']:.0f}% of band below {s11_threshold} dB. "
                       f"Worst S11: {compliance['worst_s11_db']:.1f} dB")
        else:
            st.error(f"❌ Band NOT covered. {compliance.get('coverage_pct', 0):.0f}% coverage. "
                     f"Worst S11: {compliance.get('worst_s11_db', 0):.1f} dB")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Fields — Near-field E/H/B & Radiation Pattern
# ═══════════════════════════════════════════════════════════════════════════════

with tab_fields:
    st.header("Field Visualization")

    col_fc, col_fp = st.columns([1, 2])

    with col_fc:
        st.subheader("Settings")
        field_freq = st.number_input("Frequency (MHz)", value=target_freq,
                                      min_value=0.01, key="field_freq")
        field_plane = st.selectbox("Observation plane", ["xz", "xy", "yz"])
        field_extent = st.slider("Extent (mm)", 10.0, 500.0, 100.0, key="field_extent")
        field_resolution = st.slider("Resolution", 30, 100, 50, key="field_res")
        field_offset = st.slider("Plane offset (mm)", -50.0, 50.0, 0.0, key="field_offset")
        field_type = st.selectbox("Field quantity", ["B_mag", "H_mag", "E_mag",
                                                      "Bx", "By", "Bz", "Hx", "Hy", "Hz"])

        # Map template to field model
        type_map = {
            "mifa": "dipole", "pifa": "dipole", "meander": "dipole",
            "loop": "loop", "spiral": "loop",
            "patch": "patch", "helix": "helix",
        }
        ant_type = type_map.get(template_key, "dipole")

        compute_fields = st.button("🧲 Compute Fields", type="primary", key="compute_fields")

    with col_fp:
        if compute_fields:
            with st.spinner("Computing near fields..."):
                fdata = compute_near_fields(
                    ant_type, field_freq * 1e6, params,
                    plane=field_plane, extent_mm=field_extent,
                    n_points=field_resolution, z_offset_mm=field_offset,
                )
                st.session_state["field_data"] = fdata

        if "field_data" in st.session_state:
            fdata = st.session_state["field_data"]

            # Heatmap
            unit_labels = {
                "E_mag": "|E| (V/m)", "H_mag": "|H| (A/m)", "B_mag": "|B| (T)",
            }
            fig_heat = plot_field_heatmap(
                fdata, field_type,
                title=f"{unit_labels.get(field_type, field_type)} — {field_plane} plane @ {field_freq} MHz",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # Vector arrows
            vec_comp = "H" if field_type.startswith(("B", "H")) else "E"
            fig_vec = plot_field_arrows(
                fdata, vec_comp,
                title=f"{vec_comp}-field vectors — {field_plane} plane",
            )
            st.plotly_chart(fig_vec, use_container_width=True)

    # 3D radiation pattern
    st.divider()
    st.subheader("3D Radiation Pattern")

    if st.button("📡 Compute Pattern", key="compute_pattern"):
        with st.spinner("Computing radiation pattern..."):
            pat = compute_radiation_pattern_3d(ant_type, target_freq * 1e6, params)
            st.session_state["pattern_data"] = pat

    if "pattern_data" in st.session_state:
        fig_pat = plot_radiation_3d(st.session_state["pattern_data"],
                                    f"3D Radiation Pattern @ {target_freq} MHz")
        st.plotly_chart(fig_pat, use_container_width=True)

        pat = st.session_state["pattern_data"]
        peak_gain = float(np.max(pat["gain_dbi"]))
        st.metric("Peak Gain", f"{peak_gain:.1f} dBi")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: Optimize — Multi-objective Optimization
# ═══════════════════════════════════════════════════════════════════════════════

with tab_optimize:
    st.header("Multi-Objective Optimization")

    col_oc, col_ow = st.columns([1, 1])

    with col_oc:
        st.subheader("Optimizer Settings")
        opt_method = st.selectbox("Algorithm",
                                   ["CMA-ES", "Bayesian (GP surrogate)"],
                                   help="CMA-ES: fast, good for nonconvex. Bayesian: best for expensive sims.")
        opt_iterations = st.slider("Iterations", 10, 300, 50, key="opt_iters")
        opt_seed = st.number_input("Random seed", value=42, min_value=0, key="opt_seed")

    with col_ow:
        st.subheader("Objective Weights")
        w_s11 = st.slider("S11 (match)", 0.0, 2.0, 1.0, 0.1, key="w_s11")
        w_bw = st.slider("Bandwidth", 0.0, 2.0, 0.5, 0.1, key="w_bw")
        w_eff = st.slider("Efficiency", 0.0, 2.0, 0.8, 0.1, key="w_eff")
        w_size = st.slider("Size (minimize)", 0.0, 2.0, 0.3, 0.1, key="w_size")
        w_robust = st.slider("Robustness", 0.0, 2.0, 0.2, 0.1, key="w_robust")

    run_opt = st.button("🚀 Run Optimization", type="primary", key="run_opt")

    if run_opt:
        config = ObjectiveConfig(
            target_freq_hz=target_freq * 1e6,
            freq_band_start_hz=freq_start * 1e6,
            freq_band_stop_hz=freq_stop * 1e6,
            s11_threshold_db=s11_threshold,
            min_bandwidth_hz=min_bw * 1e6,
            min_efficiency=min_efficiency,
            max_radius_mm=max_radius,
            s11_weight=w_s11,
            bandwidth_weight=w_bw,
            efficiency_weight=w_eff,
            size_weight=w_size,
            robustness_weight=w_robust,
            sar_constraint=sar_enabled,
        )

        objective = AntennaObjective(config, template)

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        metric_cols = st.columns(3)
        best_s11_display = metric_cols[0].empty()
        best_size_display = metric_cols[1].empty()
        evals_display = metric_cols[2].empty()

        def opt_callback(iteration, params_i, score):
            pct = min(iteration / max(opt_iterations, 1), 1.0)
            progress_bar.progress(pct)
            status_text.text(f"Iteration {iteration} — Score: {score:.4f}")
            best = objective.best_result
            if best:
                best_s11_display.metric("Best S11", f"{best.s11_db:.1f} dB")
                best_size_display.metric("Best Size", f"{best.radius_mm:.1f} mm")
            evals_display.metric("Evaluations", str(objective.eval_count))

        with st.spinner("Optimizing..."):
            if opt_method == "CMA-ES":
                optimizer = CMAESOptimizer(CMAESConfig(
                    max_evaluations=opt_iterations * 5,
                    random_seed=opt_seed, verbose=False,
                ))
            else:
                optimizer = BayesianOptimizer(BayesianOptimizerConfig(
                    n_initial=10, n_iterations=opt_iterations,
                    random_seed=opt_seed, verbose=False,
                ))

            opt_result = optimizer.optimize(objective, callback=opt_callback)
            st.session_state.opt_result = opt_result
            progress_bar.progress(1.0)

    if st.session_state.opt_result is not None:
        opt_result = st.session_state.opt_result
        best = opt_result.get("best_result") if isinstance(opt_result, dict) else opt_result.best_result

        st.success("Optimization complete!")

        st.subheader("Best Design Found")
        col_bp, col_bm = st.columns(2)

        with col_bp:
            st.write("**Optimized Parameters:**")
            for k, v in best.parameters.items():
                st.write(f"  - `{k}`: **{v:.4f}**")

            if st.button("📥 Apply to Parameter Editor"):
                for k, v in best.parameters.items():
                    key = f"param_{template_key}_{k}"
                    if key in st.session_state:
                        st.session_state[key] = v
                st.rerun()

        with col_bm:
            st.write("**Performance:**")
            st.metric("S11", f"{best.s11_db:.1f} dB")
            st.metric("Bandwidth", f"{best.bandwidth_hz / 1e6:.2f} MHz")
            st.metric("Efficiency", f"{best.efficiency * 100:.1f}%")
            st.metric("Size", f"{best.radius_mm:.1f} mm")
            st.metric("Score", f"{best.total_score:.4f}")
            if not best.feasible:
                st.warning(f"Constraint violations: {', '.join(best.constraint_violations)}")

        # Convergence plot
        conv = opt_result.get("convergence_history") if isinstance(opt_result, dict) else getattr(opt_result, 'convergence_history', [])
        if conv:
            fig_conv = plot_convergence_interactive(conv)
            st.plotly_chart(fig_conv, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: Matching Network
# ═══════════════════════════════════════════════════════════════════════════════

with tab_matching:
    st.header("Impedance Matching Network")

    st.write("Design a matching network to transform the antenna impedance to 50 Ω.")

    z_in = template.estimate_impedance(params, target_freq * 1e6)

    col_z, col_match = st.columns([1, 2])

    with col_z:
        st.subheader("Antenna Impedance")
        st.metric("R (real)", f"{z_in.real:.1f} Ω")
        st.metric("X (imag)", f"{z_in.imag:+.1f} Ω")
        st.metric("|Z|", f"{abs(z_in):.1f} Ω")

        from agentic_rf.utils.units import z_to_gamma, s11_to_vswr, mag_to_db
        gamma = z_to_gamma(z_in)
        st.metric("S11 (unmatched)", f"{mag_to_db(abs(gamma)):.1f} dB")
        st.metric("VSWR (unmatched)", f"{s11_to_vswr(abs(gamma)):.2f}" if abs(gamma) < 0.99 else ">100")

    with col_match:
        st.subheader("Matching Solutions")

        analyzer = ImpedanceAnalyzer(50.0)

        # Reactance compensation
        comp = analyzer.compensate_reactance(z_in, target_freq * 1e6)
        if comp["needed"]:
            st.info(f"**Reactance compensation:** {comp['note']}")

        # L-match
        matches = analyzer.synthesize_l_match(z_in, target_freq * 1e6)
        if matches:
            for i, m in enumerate(matches):
                with st.expander(f"Solution {i+1}: {m.topology}", expanded=(i == 0)):
                    st.write(m.describe())
                    for c in m.components:
                        if c["type"] == "L":
                            st.write(f"  **{c['position'].title()} Inductor:** {c['value']*1e9:.2f} nH")
                        else:
                            st.write(f"  **{c['position'].title()} Capacitor:** {c['value']*1e12:.2f} pF")
        else:
            st.warning("No L-match solution found. Try a Pi or T network, or adjust the antenna design.")

        # Q factor
        if st.session_state.sim_result is not None:
            result = st.session_state.sim_result
            if result.z_input.size > 0:
                q_data = analyzer.extract_q(result.frequencies_hz, result.z_input[:, 0])
                st.divider()
                st.subheader("Loaded Q Factor")
                st.metric("Q (loaded)", f"{q_data['Q_loaded']:.1f}")
                st.metric("Resonance (from X=0)", f"{q_data['f0_mhz']:.1f} MHz")
                st.metric("R at resonance", f"{q_data['R_at_resonance']:.1f} Ω")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: Safety — SAR & Thermal
# ═══════════════════════════════════════════════════════════════════════════════

with tab_safety:
    st.header("Safety Analysis")

    if not sar_enabled:
        st.info("SAR analysis is available for implant applications. "
                "Select an implant application type in the sidebar to enable.")
    else:
        col_sin, col_sout = st.columns([1, 2])

        with col_sin:
            st.subheader("Input Parameters")
            sar_power = st.number_input("Input power (mW)", value=max_power_mw,
                                         min_value=0.01, key="sar_power")
            sar_eff = st.slider("Estimated efficiency (%)", 1, 50, 10, key="sar_eff") / 100
            sar_tissue = st.selectbox("Tissue type",
                                       ["Muscle", "Fat", "Skin (Dry)", "Brain (Grey)", "Blood"],
                                       key="sar_tissue")
            sar_distance = st.slider("Distance to tissue (mm)", 0.0, 10.0, 0.0, 0.5, key="sar_dist")

            run_sar = st.button("🛡️ Run SAR Analysis", type="primary", key="run_sar")

        with col_sout:
            if run_sar:
                tissue_map = {
                    "Muscle": "muscle", "Fat": "fat", "Skin (Dry)": "skin_dry",
                    "Brain (Grey)": "brain_grey", "Blood": "blood",
                }
                tissue_mat = get_material(tissue_map.get(sar_tissue, "muscle"))

                sar_analyzer = SARAnalyzer()
                report = sar_analyzer.compliance_report(
                    sar_power * 1e-3, sar_eff, target_freq * 1e6,
                    tissue_mat, sar_distance * 1e-3,
                )

                st.subheader("SAR Results")

                if report["overall_compliant"]:
                    st.success("✅ Overall COMPLIANT with safety limits")
                else:
                    st.error("❌ EXCEEDS safety limits — reduce power or improve efficiency")

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.metric("Peak SAR (1g)", f"{report['sar']['peak_1g_w_per_kg']:.3f} W/kg")
                    st.metric("FCC limit (1g)", f"{report['sar']['limit_fcc_1g']} W/kg")
                    st.metric("FCC compliant", "✅ Yes" if report['sar']['compliant_fcc'] else "❌ No")
                with col_s2:
                    st.metric("Peak SAR (10g)", f"{report['sar']['peak_10g_w_per_kg']:.3f} W/kg")
                    st.metric("ICNIRP limit (10g)", f"{report['sar']['limit_icnirp_10g']} W/kg")
                    st.metric("ICNIRP compliant", "✅ Yes" if report['sar']['compliant_icnirp'] else "❌ No")

                st.metric("Max safe power", f"{report['sar']['max_safe_power_mw']:.1f} mW")

                st.subheader("Thermal Analysis")
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.metric("Peak ΔT", f"{report['thermal']['peak_temp_rise_c']:.3f} °C")
                    st.metric("Limit (implant)", f"{report['thermal']['limit_c']} °C")
                with col_t2:
                    st.metric("Thermal compliant", "✅ Yes" if report['thermal']['compliant'] else "❌ No")
                    if report['thermal']['safe_duration_min']:
                        st.metric("Safe duration", f"{report['thermal']['safe_duration_min']:.0f} min")
                    else:
                        st.metric("Safe duration", "Unlimited")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8: Measurement — Import VNA Data & Calibration
# ═══════════════════════════════════════════════════════════════════════════════

with tab_measure:
    st.header("Measurement & Calibration")

    st.write("Upload VNA measurement data (Touchstone S1P/S2P) to compare with simulation "
             "and calibrate your model.")

    uploaded_file = st.file_uploader(
        "Upload Touchstone file (.s1p or .s2p)",
        type=["s1p", "s2p", "S1P", "S2P"],
        key="upload_touchstone",
    )

    if uploaded_file is not None:
        # Save to temp file for parsing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=f".{uploaded_file.name.split('.')[-1]}",
                                          delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Parsing measurement data..."):
            meas_result = TouchstoneIO.read(tmp_path)
            st.session_state.measurement_result = meas_result

        st.success(f"Loaded: {uploaded_file.name} — "
                   f"{len(meas_result.frequencies_hz)} points, "
                   f"{meas_result.frequencies_hz[0]/1e6:.1f}–{meas_result.frequencies_hz[-1]/1e6:.1f} MHz")

        # Plot measurement
        fig_meas = plot_s11_interactive(meas_result, title=f"Measured S11 — {uploaded_file.name}")
        st.plotly_chart(fig_meas, use_container_width=True)

        # Fixture artifacts
        artifacts = TouchstoneIO.detect_fixture_artifacts(meas_result)
        if artifacts.get("ripple_detected"):
            st.warning(f"⚠️ Fixture artifacts detected: {artifacts['suggestion']}")
        else:
            st.info(f"Data looks clean. Estimated cable delay: {artifacts.get('cable_delay_ns', 0):.2f} ns")

        # Sim vs Measurement comparison
        if st.session_state.sim_result is not None:
            st.divider()
            st.subheader("Simulation vs Measurement")

            sim = st.session_state.sim_result

            # Overlay plot
            import plotly.graph_objects as go
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(
                x=sim.frequencies_hz / 1e6, y=sim.s11,
                name="Simulated", line=dict(color="blue", width=2),
            ))
            fig_compare.add_trace(go.Scatter(
                x=meas_result.frequencies_hz / 1e6, y=meas_result.s11,
                name="Measured", line=dict(color="red", width=2, dash="dash"),
            ))
            fig_compare.update_layout(
                title="S11: Simulation vs Measurement",
                xaxis_title="Frequency (MHz)", yaxis_title="S11 (dB)",
                height=400,
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            # Calibration
            st.subheader("Model Calibration")
            if st.button("🔧 Calibrate Model", type="primary", key="run_cal"):
                with st.spinner("Calibrating..."):
                    calibrator = ModelCalibrator(template)
                    cal = calibrator.calibrate(sim, meas_result, params)
                    model = calibrator.build_reality_model(cal, params)

                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.metric("Frequency error", f"{cal.freq_error_pct:+.1f}%")
                    st.metric("εr correction", f"×{cal.er_correction:.3f}")
                    st.metric("Loss correction", f"×{cal.loss_correction:.3f}")
                with col_c2:
                    st.metric("RMS error", f"{cal.rms_error_db:.2f} dB")
                    st.metric("Correlation", f"{cal.correlation:.4f}")
                    st.metric("Effective εr", f"{model.effective_er:.2f}")

                st.info(cal.notes)

                # Suggest redesign
                new_params = calibrator.suggest_redesign(cal, params, target_freq * 1e6)
                st.subheader("Suggested Parameter Corrections")
                for k in params:
                    if abs(new_params.get(k, params[k]) - params[k]) > 0.01:
                        st.write(f"  `{k}`: {params[k]:.3f} → **{new_params[k]:.3f}**")

                if st.button("📥 Apply Corrections"):
                    for k, v in new_params.items():
                        key = f"param_{template_key}_{k}"
                        if key in st.session_state:
                            st.session_state[key] = v
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9: Design History & Export
# ═══════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.header("Design History & Export")

    history = st.session_state.design_history

    if not history:
        st.info("No design snapshots yet. Use the 💾 Save button in the Design tab to save iterations.")
    else:
        st.subheader(f"Saved Designs ({len(history)} iterations)")

        for i, snap in enumerate(reversed(history)):
            with st.expander(
                f"**#{snap['iteration']}** — {snap['template'].upper()} @ {snap['timestamp']}",
                expanded=(i == 0),
            ):
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    st.write("**Parameters:**")
                    for k, v in snap["params"].items():
                        st.write(f"  `{k}`: {v:.4f}")
                with col_h2:
                    if "metrics" in snap:
                        st.write("**Performance:**")
                        for k, v in snap["metrics"].items():
                            if v is not None:
                                st.write(f"  `{k}`: {v:.4f}" if isinstance(v, float) else f"  `{k}`: {v}")
                    if snap.get("notes"):
                        st.write(f"**Notes:** {snap['notes']}")

                if st.button(f"📥 Restore Design #{snap['iteration']}", key=f"restore_{i}"):
                    for k, v in snap["params"].items():
                        key = f"param_{snap['template']}_{k}"
                        st.session_state[key] = v
                    st.rerun()

    # ── Export ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Export")

    col_e1, col_e2, col_e3 = st.columns(3)

    with col_e1:
        if st.button("📄 Export Design JSON"):
            export = {
                "template": template_key,
                "parameters": params,
                "target_freq_mhz": target_freq,
                "band_mhz": [freq_start, freq_stop],
                "quick_eval": template.quick_evaluate(params, target_freq * 1e6),
            }
            json_str = json.dumps(export, indent=2, default=str)
            st.download_button("Download JSON", json_str, "antenna_design.json", "application/json")

    with col_e2:
        if st.button("📜 Export OpenEMS Script"):
            from agentic_rf.solver.openems_solver import OpenEMSSolver
            solver = OpenEMSSolver()
            geom = template.generate_geometry(template.validate_params(params))
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as tmp:
                script = solver.generate_script(geom, freq_start * 1e6, freq_stop * 1e6, tmp.name)
                st.download_button("Download Script", script, "openems_simulation.py", "text/x-python")

    with col_e3:
        if st.session_state.sim_result is not None:
            if st.button("📊 Export S1P"):
                result = st.session_state.sim_result
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".s1p", delete=False) as tmp:
                    TouchstoneIO.write_s1p(
                        tmp.name, result.frequencies_hz,
                        result.s11_complex, 50.0, "RI",
                    )
                    with open(tmp.name, 'r') as f:
                        s1p_content = f.read()
                    st.download_button("Download S1P", s1p_content,
                                       "simulated.s1p", "text/plain")


# ═══════════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()
st.caption("Agentic RF v0.1.0 — Solver-in-the-loop, measurement-in-the-loop antenna design. "
           "Analytical models are approximate (±20%). Use full-wave simulation for final validation.")
