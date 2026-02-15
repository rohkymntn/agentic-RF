# Agentic RF

**AI-driven antenna design that actually produces buildable antennas.**

Agentic RF is a solver-in-the-loop and measurement-in-the-loop system for designing, optimizing, simulating, and validating RF antennas. It combines LLM reasoning with real electromagnetic physics — not just formulas, but actual parametric geometry generation, EM solver interfaces, multi-objective optimization, and measurement calibration.

## Why This Exists

LLM + formulas alone won't design good antennas. Real antenna performance emerges from Maxwell's equations + geometry + materials + environment. This system gives an AI agent the tools to:

1. **Generate real parametric geometry** from a library of proven antenna families
2. **Simulate** using analytical fast-screening or full-wave FDTD (OpenEMS)
3. **Optimize** with Bayesian, CMA-ES, or NSGA-II multi-objective algorithms
4. **Ingest measurements** from VNA data (Touchstone S1P/S2P)
5. **Calibrate** simulation models against reality
6. **Close the loop** — re-optimize with reality-corrected models

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LLM Agent Layer                       │
│  (Design reasoning, topology selection, interpretation)  │
├─────────────┬──────────────┬────────────┬───────────────┤
│  Geometry   │   Solvers    │ Optimizer  │ Measurement   │
│  Generator  │              │            │    Loop       │
│             │ ● Analytical │ ● Bayesian │               │
│ ● MIFA     │ ● OpenEMS    │ ● CMA-ES   │ ● Touchstone  │
│ ● PIFA     │   (FDTD)     │ ● NSGA-II  │ ● Calibration │
│ ● Loop     │ ● NEC2       │            │ ● Reality     │
│ ● Patch    │   (MoM)      │            │   Update      │
│ ● Helix    │              │            │               │
│ ● Meander  ├──────────────┼────────────┤               │
│ ● Spiral   │   Analysis   │   Safety   │               │
│             │ ● S-params   │ ● SAR      │               │
│             │ ● Impedance  │ ● Thermal  │               │
│             │ ● Matching   │ ● FCC/ICNIRP│              │
├─────────────┴──────────────┴────────────┴───────────────┤
│             Visualization & Export                        │
│  Smith chart · S-params · Patterns · 3D · Pareto front  │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the web UI (recommended)
streamlit run frontend.py

# Run the CLI demo (no API key needed)
python app.py demo

# Interactive agent chat (needs OpenAI or Anthropic key)
export OPENAI_API_KEY=sk-...
python app.py chat

# Quick optimization
python app.py optimize -t meander -f 403.5 -s 10 -n 100

# Evaluate a specific design
python app.py evaluate -t pifa -f 2450

# Analyze VNA measurement
python app.py analyze measurement.s1p
```

## Antenna Templates

| Template | Key | Best For | Miniaturization |
|----------|-----|----------|-----------------|
| Meandered Inverted-F (MIFA) | `mifa` | IoT, wearable, compact PCB | Excellent |
| Planar Inverted-F (PIFA) | `pifa` | Mobile phones, laptops | Very good |
| Loop | `loop` | WPT, NFC, RFID, implant links | Good |
| Microstrip Patch | `patch` | Directional, GNSS, radar | Moderate |
| Helix | `helix` | Implant (normal), satellite (axial) | Excellent (normal mode) |
| Meander Line | `meander` | Implant, extreme miniaturization | Best |
| Planar Spiral | `spiral` | WPT coils, broadband | Good |

## What "Good Antennas" Means

This system optimizes multiple objectives simultaneously:

- **Match**: S11 / VSWR across the target band
- **Efficiency**: radiation efficiency (or link efficiency for near-field)
- **Pattern/Gain**: far-field directivity or near-field coupling
- **Bandwidth vs Size**: respects the Chu-Harrington fundamental limit
- **Robustness**: sensitivity to detuning (hand, enclosure, tissue, packaging)
- **Constraints**: size, keepouts, ground plane, materials, manufacturability
- **Safety**: SAR / thermal analysis for implant applications

## Key Features

### Parametric Geometry Engine
Each antenna template is parameterized with physically meaningful dimensions. The engine generates solver-ready 3D geometry with traces, substrate boxes, ground planes, and ports.

### Multi-Solver Support
- **Analytical** (built-in): ~1ms per evaluation, ±20% accuracy. For screening.
- **OpenEMS** (FDTD): Full-wave, minutes per sim. For accurate design.
- **NEC2** (MoM): Wire antennas. Fast and accurate for loops/helices.

### Multi-Objective Optimization
- **Bayesian** (scikit-optimize): Best when sims are expensive. Builds a GP surrogate.
- **CMA-ES** (pycma): Excellent for nonconvex landscapes with parameter interactions.
- **NSGA-II** (pymoo): Produces Pareto front showing tradeoffs (size vs BW vs efficiency).

### Measurement-in-the-Loop
- Parse Touchstone S1P/S2P files from any VNA
- Detect fixture/cable artifacts (ripple, delay)
- Calibrate simulation model against measurement
- Re-optimize with reality-corrected parameters
- Close the sim → fab → measure → re-sim → re-optimize loop

### Safety Analysis
- SAR estimation (1g and 10g averaging)
- Thermal rise from Pennes bioheat model
- FCC and ICNIRP compliance checking
- Maximum safe power calculation

### Material Database
30+ materials including: PCB substrates (FR-4, Rogers), conductors (Cu, Au, Ti), biological tissues (muscle, fat, bone, blood, brain), encapsulants (PDMS, PEEK, silicone), and ceramics.

## Design Workflow

```
1. Requirements → "I need a 403 MHz implant antenna, max 10mm, in muscle tissue"
       │
2. Feasibility → Chu limit check: ka=0.08, Q_min=19531, max BW=0.003%
       │         "Physically possible but very narrowband. Use high-Q resonator."
       │
3. Template  → Meander line (best miniaturization for planar implant)
       │
4. Optimize  → CMA-ES / Bayesian over parameter space
       │         Objectives: S11 < -10 dB, max BW, min size, SAR compliant
       │
5. Simulate  → Analytical fast-screen → OpenEMS full-wave verification
       │
6. Match     → L-network: series L = 47 nH, shunt C = 3.3 pF
       │
7. Fabricate  → Export geometry, generate OpenEMS script
       │
8. Measure   → VNA → .s1p file → load into system
       │
9. Calibrate → Compare sim vs meas → εr was 4.4, actual ≈ 4.8
       │
10. Re-optimize → With corrected model → final design
```

## Project Structure

```
agentic_rf/
├── agent/              # LLM agent: orchestrator, tools, prompts
├── geometry/           # Parametric antenna templates
│   └── templates/      # MIFA, PIFA, loop, patch, helix, meander, spiral
├── solver/             # EM solver interfaces
│   ├── analytical.py   # Fast analytical models
│   ├── openems_solver.py  # OpenEMS FDTD
│   └── nec2_solver.py  # NEC2 MoM
├── optimizer/          # Multi-objective optimization
│   ├── bayesian.py     # Bayesian (GP surrogate)
│   ├── cmaes.py        # CMA-ES
│   └── genetic.py      # NSGA-II
├── measurement/        # VNA data, calibration, reality update
├── analysis/           # S-params, impedance, matching, SAR
├── visualization/      # Plots: Smith, patterns, geometry, Pareto
└── utils/              # Constants, materials, units
```

## Optional Dependencies

For full-wave simulation:
```bash
# OpenEMS (FDTD solver)
# Ubuntu: sudo apt install openems
# macOS: brew install openems
pip install CSXCAD openEMS

# NEC2 (wire antenna solver)
pip install PyNEC
```

## API Usage (Python)

```python
from agentic_rf.geometry.templates import get_template
from agentic_rf.solver.analytical import AnalyticalSolver
from agentic_rf.optimizer.objectives import AntennaObjective, ObjectiveConfig
from agentic_rf.optimizer.cmaes import CMAESOptimizer

# Get a template and evaluate
meander = get_template("meander")
result = meander.quick_evaluate(meander.get_default_params(), 403.5e6)
print(f"Resonance: {result['f_res_mhz']:.1f} MHz, S11: {result['s11_db']:.1f} dB")

# Optimize
config = ObjectiveConfig(target_freq_hz=403.5e6, max_radius_mm=10.0)
objective = AntennaObjective(config, meander)
optimizer = CMAESOptimizer()
result = optimizer.optimize(objective)
print(f"Best S11: {result['best_result'].s11_db:.1f} dB")

# Simulate
solver = AnalyticalSolver(meander)
geom = meander.generate_geometry(result['best_params'])
sim = solver.simulate(geom, 380e6, 430e6, 201)
print(f"BW: {sim.bandwidth_hz()/1e6:.2f} MHz")

# WPT link design
from agentic_rf.geometry.templates.spiral import SpiralAntennaTemplate
spiral = SpiralAntennaTemplate()
k = spiral.coupling_coefficient(tx_params, rx_params, distance_m=0.03)
eta = spiral.wpt_link_efficiency(tx_params, rx_params, 0.03, 6.78e6)
```

## Web Frontend

The interactive web UI (`streamlit run frontend.py`) provides a complete design studio:

### Tabs & Workflow

| Tab | What It Does | User Inputs |
|-----|-------------|-------------|
| **Design** | Template selection + parameter sliders with live analytical evaluation | Select antenna type, adjust all parameters via sliders, set feedback notes |
| **Geometry** | Interactive 3D Plotly view of the antenna (rotate, zoom, pan) | View only — updates from parameter changes |
| **Simulate** | Frequency sweep showing S11, VSWR, Smith chart, impedance | Set sweep range, number of points, click Run |
| **Fields** | Near-field E/H/B heatmaps and vector plots + 3D radiation pattern | Choose plane (XZ/XY/YZ), extent, resolution, field quantity, frequency |
| **Optimize** | Multi-objective optimization with live progress bar and convergence plot | Select algorithm, iterations, tune objective weights (S11/BW/efficiency/size/robustness) |
| **Matching** | Automatic L-match network synthesis + reactance compensation | View-only — computed from current impedance |
| **Safety** | SAR (1g/10g) and thermal analysis with FCC/ICNIRP compliance | Set input power, efficiency estimate, tissue type, distance |
| **Measure** | Upload VNA data (.s1p/.s2p), compare sim vs measurement, calibrate model | Upload Touchstone file, click Calibrate, click Apply Corrections |
| **History** | Browse all saved design iterations, restore any past design, export | Click Save in Design tab; export JSON, OpenEMS script, or S1P |

### Sidebar (Always Visible)

The sidebar contains all design specifications that affect every tab:
- **Application type** — implant, IoT, wearable, WPT, mobile
- **Frequency band** — presets (MICS, ISM 433/900/2.4G/5.8G, Qi) or custom
- **Target frequency** — center frequency for optimization/evaluation
- **Size constraint** — max enclosing radius in mm
- **Environment** — free space, on-body, implanted tissue type
- **Performance goals** — S11 threshold, min bandwidth, min efficiency
- **Chu limit** — automatic fundamental limit calculation shown live

### Iterative Design Loop

1. Set specs in sidebar → select template → adjust parameters
2. Simulate → see S11, VSWR, Smith chart
3. Not happy? → tweak parameters or run optimizer
4. Visualize fields → check B-field pattern
5. Design matching network → get component values
6. Save snapshot with notes → iterate
7. Fabricate → measure with VNA → upload .s1p
8. Calibrate → model corrects itself → re-optimize → done

## License

MIT
