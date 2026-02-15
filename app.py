#!/usr/bin/env python3
"""
Agentic RF — AI-driven antenna design system.

Usage:
    python app.py chat              # Interactive agent chat
    python app.py optimize          # Quick optimization
    python app.py evaluate          # Evaluate a design
    python app.py analyze FILE      # Analyze measurement data
    python app.py demo              # Run demo showing all capabilities
"""

import sys
import json
import argparse

import numpy as np


def cmd_chat(args):
    """Interactive chat with the antenna design agent."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    from agentic_rf.agent.orchestrator import AntennaDesignAgent

    console = Console()

    console.print(Panel.fit(
        "[bold blue]Agentic RF[/bold blue] — AI Antenna Design Agent\n"
        "Type your design request. Type 'quit' to exit.\n"
        f"Mode: {'Online (LLM)' if args.api_key else 'Offline (direct tools)'}",
        border_style="blue",
    ))

    agent = AntennaDesignAgent(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
    )

    if args.freq:
        agent.session.target_freq_mhz = args.freq
    if args.size:
        agent.session.max_size_mm = args.size

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/bold green] ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ('quit', 'exit', 'q'):
            break

        if not user_input.strip():
            continue

        with console.status("[bold blue]Thinking...[/bold blue]"):
            if args.api_key:
                response = agent.chat(user_input)
            else:
                response = agent.chat_offline(user_input)

        console.print(f"\n[bold blue]Agent:[/bold blue]")
        try:
            console.print(Markdown(response))
        except Exception:
            console.print(response)

    console.print("\n[dim]Session ended.[/dim]")


def cmd_optimize(args):
    """Run quick antenna optimization."""
    from rich.console import Console
    from rich.table import Table

    from agentic_rf.agent.tools import tool_optimize_antenna

    console = Console()
    console.print(f"\n[bold]Optimizing {args.template} antenna[/bold]")
    console.print(f"  Target: {args.freq} MHz, Max size: {args.size} mm")
    console.print(f"  Optimizer: {args.optimizer}, Iterations: {args.iterations}")
    console.print()

    with console.status("[bold blue]Running optimization...[/bold blue]"):
        result = tool_optimize_antenna(
            template_name=args.template,
            target_freq_mhz=args.freq,
            band_start_mhz=args.freq - 1.5,
            band_stop_mhz=args.freq + 1.5,
            max_radius_mm=args.size,
            n_iterations=args.iterations,
            optimizer_type=args.optimizer,
            sar_constraint=args.sar,
        )

    # Display results
    table = Table(title="Optimized Design")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    for k, v in result["best_params"].items():
        table.add_row(k, f"{v:.4f}")

    console.print(table)

    perf_table = Table(title="Performance")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="green")

    for k, v in result["performance"].items():
        if isinstance(v, float):
            perf_table.add_row(k, f"{v:.4f}")
        else:
            perf_table.add_row(k, str(v))

    console.print(perf_table)
    console.print(f"\n  Evaluations: {result['n_evaluations']}, Time: {result['wall_time_s']:.1f}s")


def cmd_evaluate(args):
    """Evaluate a specific antenna design."""
    from rich.console import Console
    from rich.table import Table

    from agentic_rf.geometry.templates import get_template

    console = Console()
    tmpl = get_template(args.template)

    params = tmpl.get_default_params()
    if args.params:
        for p in args.params:
            key, val = p.split("=")
            params[key.strip()] = float(val.strip())

    result = tmpl.quick_evaluate(params, args.freq * 1e6)

    table = Table(title=f"{tmpl.name} @ {args.freq} MHz")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for k, v in result.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))

    console.print(table)


def cmd_analyze(args):
    """Analyze measurement data."""
    from rich.console import Console

    from agentic_rf.agent.tools import tool_analyze_sparams

    console = Console()
    console.print(f"\n[bold]Analyzing: {args.file}[/bold]\n")

    result = tool_analyze_sparams(args.file)
    console.print_json(json.dumps(result, indent=2, default=str))


def cmd_demo(args):
    """Run a comprehensive demo of all capabilities."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    console.print(Panel.fit(
        "[bold blue]Agentic RF Demo[/bold blue]\n"
        "Demonstrating antenna design workflow for MICS-band implant antenna",
        border_style="blue",
    ))

    # Step 1: Check fundamental limits
    console.print("\n[bold]Step 1: Fundamental Limits (Chu-Harrington)[/bold]")
    from agentic_rf.agent.tools import tool_chu_limit
    chu = tool_chu_limit(freq_mhz=403.5, radius_mm=10.0)
    for k, v in chu.items():
        console.print(f"  {k}: {v}")

    # Step 2: List templates
    console.print("\n[bold]Step 2: Available Antenna Templates[/bold]")
    from agentic_rf.geometry.templates import list_templates
    for t in list_templates():
        console.print(f"  [{t['key']}] {t['name']} ({t['num_parameters']} params)")

    # Step 3: Evaluate default meander
    console.print("\n[bold]Step 3: Evaluate Default Meander Antenna @ 403.5 MHz[/bold]")
    from agentic_rf.geometry.templates import get_template
    meander = get_template("meander")
    defaults = meander.get_default_params()
    quick = meander.quick_evaluate(defaults, 403.5e6)

    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in quick.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)

    # Step 4: Quick optimization
    console.print("\n[bold]Step 4: Quick Optimization (CMA-ES, 50 iterations)[/bold]")
    from agentic_rf.agent.tools import tool_optimize_antenna
    with console.status("Optimizing..."):
        opt_result = tool_optimize_antenna(
            template_name="meander",
            target_freq_mhz=403.5,
            band_start_mhz=402.0,
            band_stop_mhz=405.0,
            max_radius_mm=10.0,
            n_iterations=30,
            optimizer_type="cmaes",
        )

    console.print(f"  Best S11: {opt_result['performance']['s11_db']:.1f} dB")
    console.print(f"  Size: {opt_result['performance']['radius_mm']:.1f} mm")
    console.print(f"  Evaluations: {opt_result['n_evaluations']}")
    console.print(f"  Time: {opt_result['wall_time_s']:.1f}s")

    # Step 5: SAR check
    console.print("\n[bold]Step 5: SAR Safety Check[/bold]")
    from agentic_rf.agent.tools import tool_sar_check
    sar = tool_sar_check(input_power_mw=25, efficiency=0.1, freq_mhz=403.5)
    console.print(f"  SAR (1g): {sar['sar']['peak_1g_w_per_kg']:.3f} W/kg")
    console.print(f"  FCC compliant: {sar['sar']['compliant_fcc']}")
    console.print(f"  Max safe power: {sar['sar']['max_safe_power_mw']:.1f} mW")

    # Step 6: Matching network
    console.print("\n[bold]Step 6: Matching Network Design[/bold]")
    z = meander.estimate_impedance(opt_result['best_params'], 403.5e6)
    from agentic_rf.agent.tools import tool_design_matching_network
    match = tool_design_matching_network(z.real, z.imag, 403.5)
    console.print(f"  Antenna Z = {z.real:.1f} + j{z.imag:.1f} Ω")
    console.print(match)

    # Step 7: WPT coil design
    console.print("\n[bold]Step 7: Wireless Power Transfer Coil Design[/bold]")
    from agentic_rf.agent.tools import tool_wpt_coil_design
    wpt = tool_wpt_coil_design(
        tx_radius_mm=25, rx_radius_mm=8,
        tx_turns=5, rx_turns=10,
        distance_mm=30, freq_mhz=6.78,
    )
    console.print(f"  Coupling k: {wpt['coupling_k']:.4f}")
    console.print(f"  Link efficiency: {wpt['link_efficiency_pct']:.1f}%")
    console.print(f"  kQ product: {wpt['kQ_product']:.1f}")
    console.print(f"  {wpt['notes']}")

    # Step 8: Simulation
    console.print("\n[bold]Step 8: Analytical Frequency Sweep[/bold]")
    from agentic_rf.solver.analytical import AnalyticalSolver
    solver = AnalyticalSolver(meander)
    geom = meander.generate_geometry(meander.validate_params(opt_result['best_params']))
    sim = solver.simulate(geom, 380e6, 430e6, 101)
    summary = sim.summary()
    console.print(f"  Resonance: {summary['resonance_mhz']:.1f} MHz")
    console.print(f"  Min S11: {summary['min_s11_db']:.1f} dB")
    console.print(f"  Bandwidth: {summary['bandwidth_mhz']:.2f} MHz")
    console.print(f"  Solve time: {summary['solve_time_s']*1000:.1f} ms")

    console.print("\n[bold green]Demo complete![/bold green] "
                  "Use 'python app.py chat' for interactive design sessions.")


def main():
    parser = argparse.ArgumentParser(
        description="Agentic RF — AI-driven antenna design system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py demo                          # Run capabilities demo
  python app.py chat                          # Interactive design chat
  python app.py optimize -t meander -f 403.5  # Optimize meander for MICS
  python app.py evaluate -t pifa -f 2450      # Evaluate PIFA at 2.45 GHz
  python app.py analyze data.s1p              # Analyze VNA measurement
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive agent chat")
    chat_parser.add_argument("--provider", choices=["openai", "anthropic"],
                             default="openai")
    chat_parser.add_argument("--model", type=str, default=None)
    chat_parser.add_argument("--api-key", type=str, default=None, dest="api_key")
    chat_parser.add_argument("--freq", type=float, default=403.5,
                             help="Target frequency (MHz)")
    chat_parser.add_argument("--size", type=float, default=15.0,
                             help="Max antenna size (mm)")

    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Run optimization")
    opt_parser.add_argument("-t", "--template", type=str, default="meander",
                            help="Antenna template")
    opt_parser.add_argument("-f", "--freq", type=float, default=403.5,
                            help="Target frequency (MHz)")
    opt_parser.add_argument("-s", "--size", type=float, default=15.0,
                            help="Max size (mm)")
    opt_parser.add_argument("-n", "--iterations", type=int, default=50)
    opt_parser.add_argument("--optimizer", choices=["bayesian", "cmaes"],
                            default="cmaes")
    opt_parser.add_argument("--sar", action="store_true", help="Enable SAR constraint")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a design")
    eval_parser.add_argument("-t", "--template", type=str, default="meander")
    eval_parser.add_argument("-f", "--freq", type=float, default=403.5)
    eval_parser.add_argument("-p", "--params", nargs="*",
                             help="Parameters as key=value pairs")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze measurement")
    analyze_parser.add_argument("file", type=str, help="Touchstone file path")

    # Demo command
    subparsers.add_parser("demo", help="Run demo")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "chat": cmd_chat,
        "optimize": cmd_optimize,
        "evaluate": cmd_evaluate,
        "analyze": cmd_analyze,
        "demo": cmd_demo,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
