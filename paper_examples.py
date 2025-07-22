#!/usr/bin/env python3
"""
Paper Figure Generator

Generate figures for the research paper. Can be run from command line with options
to select specific figures or generate all figures.

Usage:
    python paper_figures.py --figures 1 2 5    # Generate figures 1, 2, and 5
    python paper_figures.py --figure 3         # Generate only figure 3
    python paper_figures.py --all              # Generate all figures (default)
    python paper_figures.py                    # Generate all figures (default)
"""

# === Standard Library ===
import argparse
import os
import sys
import warnings

# Suppress specific pandas warning
warnings.filterwarnings(
    "ignore",
    message=".*Pyarrow will become a required dependency of pandas.*",
    category=DeprecationWarning,
)

# === Third-Party Libraries ===
import numpy as np
import pandas as pd
from tqdm import tqdm

import jax
import jax.numpy as jnp
from numpy.linalg import eigvalsh

# === Local Modules ===
from configs import Lorenz63Config, Lorenz96Config, LorenzSystem
from models import lorenz63_step, lorenz96_step
from metrics import rmse, mbe

from plotting import (
    plot_rmse_stackplot,
    plot_cross_val,
    create_sigma_histogram,
    create_rmse_heatmap,
    create_dof_slope_plot,
    create_combined_rmse_bias_plot,
    create_mixed_layout_custom,
    create_stacked_bar_plot,
)

from cost_functions import (
    bayes_cost_function,
    dci_cost_function,
    dci_wme_cost_function,
)


def setup_environment():
    """Setup JAX and other environment configurations."""
    # For reproducibility
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    jax.config.update("jax_enable_x64", True)

    # Make sure JAX is set to version 0.5.2
    if jax.__version__ != "0.5.2":
        raise ValueError(
            f"JAX version {jax.__version__} is not supported. Please use JAX version 0.5.2."
        )


def lorenz63_system_setup():
    """Setup Lorenz 63 system functions."""
    # Lorenz 63 parameters
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def lorenz63(t, state):
        """Lorenz 63 system"""
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    def lorenz63_jacobian(state):
        """Jacobian of Lorenz 63"""
        x, y, z = state
        return np.array([[-sigma, sigma, 0], [rho - z, -1, -x], [y, x, -beta]])

    def propagate_jacobian(z0, t_final, dt=0.01):
        """Tangent linear propagation of Jacobian"""
        n, J, z, t = len(z0), np.eye(len(z0)), np.array(z0), 0.0

        while t < t_final:
            A = lorenz63_jacobian(z)
            J += dt * A @ J
            z += dt * np.array(lorenz63(0, z))
            t += dt
        return J

    return lorenz63, lorenz63_jacobian, propagate_jacobian


def generate_figure_1():
    """Generate Lorenz 63 example in Figure 1 - Sigma bound distribution."""
    print("Generating Figure 1: Lorenz 63 sigma bound distribution...")

    lorenz63, lorenz63_jacobian, propagate_jacobian = lorenz63_system_setup()

    # Parameters and setup
    gamma, sigma_obs_sq, N_obs, t_final = 0.1, 2.0, 5, 10.0
    H = np.array([[1, 0, 0], [0, 1, 0]])  # Observation operator

    # Generate initial conditions and compute bounds
    np.random.seed(0)
    initial_conditions = np.random.uniform(-5, 5, size=(50, 3))

    def compute_bound(z0):
        """Compute bound for given initial condition"""
        Q = H @ propagate_jacobian(z0, t_final)
        lambda_min = eigvalsh(Q @ Q.T).min()
        return (gamma * sigma_obs_sq**2) / (N_obs * lambda_min)

    bounds = [compute_bound(z0) for z0 in initial_conditions]
    data = np.array(bounds)

    plot_params = {
        "lines.linewidth": 4,
        "lines.markeredgecolor": "black",
        "legend.fontsize": 24,
        "legend.frameon": False,
        "xtick.labelsize": 28,
        "ytick.labelsize": 28,
        "axes.labelsize": 30,
        "axes.labelpad": 12,
        "figure.figsize": (16, 10),
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }

    fig, ax = create_sigma_histogram(
        data,
        plot_params,
        save_plot=True,
        file_name="sigma_b_bound_distribution.png",
    )
    print("Figure 1 generated successfully!")


def generate_figure_2():
    """Generate Lorenz 63 example in Figure 2 - RMSE heatmap."""
    print("Generating Figure 2: Lorenz 63 RMSE heatmap...")

    # Define cost functions
    cost_functions = {
        "4DVar": bayes_cost_function,
        "DCI": dci_cost_function,
        "DCI_WME": dci_wme_cost_function,
    }
    np.random.seed(42)
    noise_levels = [0.25, 1.0, 2.25, 4.0, 6.25, 9.0]
    inflation_factors = 4.0 * jnp.array(noise_levels)
    pair_levels = list(zip(noise_levels, inflation_factors.tolist()))
    noise_inflate_dfs = []

    for level, inflation_factor in tqdm(
        pair_levels, desc="Running Inflation Level Sensitivity"
    ):
        config = Lorenz63Config.default_config(
            total_steps=1000,
            window_size=20,
            obs_frequency=4,
            obs_std=jnp.sqrt(level),
            inflation_factor=inflation_factor,
        )
        system = LorenzSystem(config, lorenz63_step, cost_functions)
        results = system.run_experiment()
        df = pd.DataFrame.from_dict(results["results"], orient="index")
        noise_inflate_dfs.append(df)

    pair_rmse_inflate = pd.concat(map(lambda df: df.rmse, noise_inflate_dfs), axis=1)
    pair_rmse_inflate.columns = pair_levels

    df = pd.DataFrame(
        {
            "Sigma": np.sqrt(noise_levels),
            "Alpha": inflation_factors,
            "4DVar": pair_rmse_inflate.loc["4DVar"].values,
            "DCI": pair_rmse_inflate.loc["DCI"].values,
            "DCI WME": pair_rmse_inflate.loc["DCI_WME"].values,
        }
    )
    df_melt = df.melt(id_vars=["Sigma", "Alpha"], var_name="Method", value_name="RMSE")

    plot_params = {
        "lines.linewidth": 2,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "axes.labelpad": 10,
        "figure.figsize": (12, 10),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "mathtext.default": "bf",
        "font.family": "sans serif",
        "font.weight": "bold",
        "font.size": 25,
        "axes.titleweight": "bold",
    }

    fig, ax, pivot = create_rmse_heatmap(
        df_melt,
        plot_params,
        save_plot=True,
        file_name="obs_inflation_levels_l63.png",
    )
    print("Figure 2 generated successfully!")


def generate_figure_3():
    """Generate Lorenz 63 example in Figure 3 - RMSE stackplot."""
    print("Generating Figure 3: Lorenz 63 RMSE stackplot...")

    cost_functions = {
        "Bayes": bayes_cost_function,
        "DCI": dci_cost_function,
        "DCI_WME": dci_wme_cost_function,
    }

    # Create system and run experiment
    config_avg = Lorenz63Config.default_config(
        total_steps=1000,
        window_size=20,
        obs_frequency=4,
        obs_std=2.0,
        inflation_factor=12,
    )
    system_avg = LorenzSystem(config_avg, lorenz63_step, cost_functions)
    results_avg = system_avg.run_experiment()

    # Calculate RMSE scores for all models
    rmse_scores = pd.DataFrame(
        {
            model: rmse(results_avg["analysis"][model], results_avg["trajectory"])
            for model in cost_functions.keys()
        }
    )

    plot_params = {
        "lines.linewidth": 8,
        "lines.markersize": 25,
        "lines.markeredgecolor": "black",
        "legend.fontsize": 65,
        "legend.frameon": False,
        "legend.title_fontsize": "50",
        "xtick.labelsize": 80,
        "ytick.labelsize": 80,
        "axes.labelsize": 82,
        "axes.labelpad": 20,
        "axes.titlesize": 50,
        "figure.figsize": (30, 20),
    }

    fig, ax = plot_rmse_stackplot(
        rmse_scores,
        plot_params=plot_params,
        save_plot=True,
        file_name="tarmse_plot_l63.png",
    )
    print("Figure 3 generated successfully!")


def generate_figure_4():
    """Generate Lorenz 63 example in Figure 4 - Cross-validation plot."""
    print("Generating Figure 4: Lorenz 63 cross-validation plot...")

    cost_functions = {"DCI_WME": dci_wme_cost_function}
    noise_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    obs_freqs = [4, 5, 10]

    rmse_noise_dfs = []

    for obs_freq in obs_freqs:
        # Run experiments for all noise levels
        dfs = []
        for noise in noise_levels:
            config = Lorenz63Config.default_config(
                total_steps=500,
                window_size=20,
                obs_frequency=obs_freq,
                obs_std=noise,
                inflation_factor=4 * (noise**2),
            )
            system = LorenzSystem(config, lorenz63_step, cost_functions)
            results = system.run_experiment()
            dfs.append(pd.DataFrame.from_dict(results["results"], orient="index"))

        # Concatenate and store results
        combined = pd.concat([df.rmse for df in dfs], axis=1)
        combined.columns = noise_levels
        rmse_noise_dfs.append(combined)

    # Create final observation density dataframe
    obs_df = pd.concat(
        [
            df.assign(Observation_Density=freq)
            for df, freq in zip(rmse_noise_dfs, obs_freqs)
        ]
    )

    dci_wme_obs = obs_df.loc["DCI_WME"]
    dci_wme_obs = dci_wme_obs.iloc[::-1]
    dci_wme_obs = dci_wme_obs.set_index("Observation_Density")

    plot_params = {
        "lines.linewidth": 10,
        "lines.markersize": 20,
        "lines.markeredgecolor": "black",
        "legend.fontsize": 30,
        "legend.frameon": False,
        "legend.title_fontsize": 20,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "axes.labelsize": 32,
        "axes.labelpad": 10,
        "axes.titlesize": 20,
        "figure.figsize": (10, 8),
        "font.weight": "bold",
    }

    plot_cross_val(
        dci_wme_obs.T,
        plot_params,
        save_plot=True,
        file_name="dci_wme_noisevdensity.png",
    )
    print("Figure 4 generated successfully!")


def generate_figure_5():
    """Generate Lorenz 96 example in Figure 5 - DOF slope plot."""
    print("Generating Figure 5: Lorenz 96 DOF slope plot...")

    # Define cost functions
    cost_functions = {
        "Bayes": bayes_cost_function,
        "DCI": dci_cost_function,
        "DCI_WME": dci_wme_cost_function,
    }

    dofs = [12, 24, 36, 48, 64]
    inf_factors = [5, 5, 5, 5, 5]
    dof_dfs = []

    for dof, level in zip(dofs, inf_factors):
        # Create system with default configuration
        config_96 = Lorenz96Config.default_config(
            state_dim=dof,
            obs_dim=dof // 2,
            total_steps=500,
            window_size=20,
            obs_frequency=4,
            obs_std=1.2,
            inflation_factor=level,
            seed=13,
        )
        system_96 = LorenzSystem(config_96, lorenz96_step, cost_functions)
        system_96.obs_system._setup_observation_operator()
        results_96 = system_96.run_experiment()
        dof_dfs.append(pd.DataFrame.from_dict(results_96["results"], orient="index"))

    rmse_dofs = pd.concat(map(lambda df: df.rmse, dof_dfs), axis=1).T.reset_index(
        drop=True
    )
    rmse_dofs.index = dofs

    plot_params = {
        "lines.linewidth": 16,
        "lines.markersize": 50,
        "lines.markeredgecolor": "black",
        "legend.fontsize": 60,
        "legend.frameon": False,
        "xtick.labelsize": 80,
        "ytick.labelsize": 80,
        "axes.labelsize": 82,
        "axes.labelpad": 20,
        "axes.titlesize": 50,
        "figure.figsize": (30, 20),
    }

    # Define colors and styles for each model
    colors = {"4D-Var": "#2E86AB", "DC-WME 4D-Var": "#A23B72"}
    markers = {"4D-Var": "o", "DC-WME 4D-Var": "^"}
    model_labels = ["4D-Var", "DC-WME 4D-Var"]

    create_dof_slope_plot(
        rmse_dofs,
        dofs,
        model_labels,
        colors,
        markers,
        plot_params,
        save_plot=True,
        file_name="L96_rmse_dof.png",
    )
    print("Figure 5 generated successfully!")


def generate_figure_6():
    """Generate Lorenz 96 example in Figure 6 - Combined RMSE/bias plot."""
    print("Generating Figure 6: Lorenz 96 combined RMSE/bias plot...")

    # Define cost functions
    cost_functions = {
        "Bayes": bayes_cost_function,
        "DCI": dci_cost_function,
        "DCI_WME": dci_wme_cost_function,
    }

    config_96 = Lorenz96Config.default_config(
        state_dim=48,
        obs_dim=48 // 2,
        total_steps=1000,
        window_size=20,
        obs_frequency=4,
        obs_std=1.2,
        inflation_factor=5,
        seed=13,
    )
    system_96 = LorenzSystem(config_96, lorenz96_step, cost_functions)
    system_96.obs_system._setup_observation_operator()

    results_avg_96 = system_96.run_experiment()
    models = list(cost_functions.keys())

    # Normalization helper
    def normalize_series(series):
        max_abs = np.max(np.abs(series))
        if max_abs == 0:
            return series, 1.0
        return series / max_abs, max_abs

    # Compute scores
    rmse_scores_background = {
        m: rmse(results_avg_96["backgrounds"][m], results_avg_96["trajectory"])
        for m in models
    }
    bias_scores_background = {
        m: mbe(results_avg_96["backgrounds"][m], results_avg_96["trajectory"])
        for m in models
    }
    rmse_scores_analysis = {
        m: rmse(results_avg_96["analysis"][m], results_avg_96["trajectory"])
        for m in models
    }
    bias_scores_analysis = {
        m: mbe(results_avg_96["analysis"][m], results_avg_96["trajectory"])
        for m in models
    }

    # Convert to DataFrames
    rmse_scores_background = pd.DataFrame(rmse_scores_background)
    bias_scores_background = pd.DataFrame(bias_scores_background)
    rmse_scores_analysis = pd.DataFrame(rmse_scores_analysis)
    bias_scores_analysis = pd.DataFrame(bias_scores_analysis)

    model_configs = {
        "Bayes": {"title": "4D-Var", "col": 0},
        "DCI": {"title": "DC 4D-Var", "col": 1},
        "DCI_WME": {"title": "DC-WME 4D-Var", "col": 2},
    }

    plot_params = {
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "lines.markeredgecolor": "black",
        "legend.fontsize": 20,
        "legend.frameon": False,
        "legend.title_fontsize": "50",
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 25,
        "figure.figsize": (15, 12),
    }

    fig, ax = create_combined_rmse_bias_plot(
        rmse_scores_background,
        rmse_scores_analysis,
        bias_scores_background,
        bias_scores_analysis,
        model_configs,
        plot_params,
        normalize_series,
        save_plot=True,
        file_name="combined_rmse_bias_plot_l96.png",
    )
    print("Figure 6 generated successfully!")


def generate_figure_7():
    """Generate Lorenz 96 example in Figure 7 - Window size plot."""
    print("Generating Figure 7: Lorenz 96 window size plot...")

    cost_functions = {
        "Bayes": bayes_cost_function,
        "DCI": dci_cost_function,
        "DCI_WME": dci_wme_cost_function,
    }
    models = list(cost_functions.keys())
    rmse_scores_analysis = {}
    window_sizes = [2, 5, 10]
    rmse_scores_dfs = []

    for size in window_sizes:
        config_96 = Lorenz96Config.default_config(
            state_dim=48,
            obs_dim=48 // 2,
            total_steps=1000,
            window_size=size,
            obs_frequency=1,
            obs_std=1.2,
            inflation_factor=5,
            seed=13,
        )
        system_96 = LorenzSystem(config_96, lorenz96_step, cost_functions)
        system_96.obs_system._setup_observation_operator()
        results_window_96 = system_96.run_experiment()

        for model in models:
            rmse_scores_analysis[model] = rmse(
                results_window_96["analysis"][model], results_window_96["trajectory"]
            )

        rmse_scores_df = pd.DataFrame(rmse_scores_analysis)
        rmse_scores_dfs.append(rmse_scores_df)

    rmse_4dvar = {
        size: df["Bayes"].values for size, df in zip(window_sizes, rmse_scores_dfs)
    }
    rmse_dci = {
        size: df["DCI"].values for size, df in zip(window_sizes, rmse_scores_dfs)
    }
    rmse_dci_wme = {
        size: df["DCI_WME"].values for size, df in zip(window_sizes, rmse_scores_dfs)
    }

    # Time axis
    T = 1000
    time = np.arange(T)

    colors = {
        "4D-Var": "#1192e8",  # IBM Blue
        "DC 4D-Var": "#fa4d56",  # IBM Red
        "DC-WME 4D-Var": "#24a148",  # IBM Green
    }

    plot_params = {
        "lines.linewidth": 3,
        "lines.markeredgecolor": "black",
        "legend.fontsize": 15,
        "legend.frameon": False,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.labelsize": 18,
        "axes.titleweight": "bold",
        "axes.titlesize": 18,
        "figure.figsize": (18, 8),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "mathtext.default": "bf",
        "font.family": "STIXGeneral",
        "font.weight": "bold",
    }

    create_mixed_layout_custom(
        time,
        rmse_4dvar,
        rmse_dci,
        rmse_dci_wme,
        window_sizes,
        colors,
        plot_params,
        save_plot=True,
        file_name="window_size_plot_l96.png",
    )
    print("Figure 7 generated successfully!")


def generate_figure_8():
    """Generate Lorenz 96 example in Figure 8 - Computational cost plot."""
    print("Generating Figure 8: Lorenz 96 computational cost plot...")

    cost_functions = {
        "Bayes": bayes_cost_function,
        "DCI": dci_cost_function,
        "DCI_WME": dci_wme_cost_function,
    }

    dofs = [24, 36, 48, 64, 72, 84, 96]
    inf_factors = 7 * np.array([1, 1, 1, 1, 1, 1, 1, 1])
    comp_costs = {}

    for dof, level in tqdm(
        zip(dofs, inf_factors),
        total=len(dofs),
        desc="Running experiments",
        unit="experiment",
    ):

        # Update progress bar description with current DOF
        tqdm.write(f"Processing DOF: {dof}")
        config_96 = Lorenz96Config.default_config(
            state_dim=dof,
            obs_dim=dof // 2,
            total_steps=20000,
            window_size=10,
            obs_frequency=2,
            obs_std=0.5,
            inflation_factor=level,
            seed=55486,
        )
        system_96 = LorenzSystem(config_96, lorenz96_step, cost_functions)
        system_96.obs_system._setup_observation_operator()
        results_96 = system_96.run_experiment()
        comp_costs[dof] = results_96["timing_results"]

    # Extract comp_costs for plotting
    degrees_of_freedom = list(comp_costs.keys())
    methods = ["4D-Var", "DC 4D-Var", "DC-WME 4D-Var"]

    # Create arrays for each method
    bayes_times = [comp_costs[dof]["Bayes"] for dof in degrees_of_freedom]
    dci_times = [comp_costs[dof]["DCI"] for dof in degrees_of_freedom]
    dci_wme_times = [comp_costs[dof]["DCI_WME"] for dof in degrees_of_freedom]

    # Set up the stacked bar chart
    x = np.arange(len(degrees_of_freedom))  # the label locations

    # Plot parameters
    plot_params = {
        "lines.markeredgecolor": "black",
        "legend.fontsize": 27,
        "legend.frameon": False,
        "xtick.labelsize": 27,
        "ytick.labelsize": 27,
        "axes.labelsize": 27,
        "axes.labelpad": 10,
        "figure.figsize": (18, 8),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "mathtext.default": "bf",
        "font.family": "STIXGeneral",
        "font.weight": "bold",
    }

    colors = {
        "Bayes": "#4E79A7",  # Blue
        "DCI": "#F28E2B",  # Orange
        "DCI_WME": "#E15759",  # Red
    }

    times = [bayes_times, dci_times, dci_wme_times]
    fig, ax, bars = create_stacked_bar_plot(
        x,
        times,
        methods,
        colors,
        degrees_of_freedom,
        plot_params,
        save_plot=True,
        file_name="comp_cost_plot_l96.png",
    )
    print("Figure 8 generated successfully!")


def main():
    """Main function to handle command line arguments and figure generation."""
    parser = argparse.ArgumentParser(
        description="Generate figures for the research paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --figures 1 2 5    Generate figures 1, 2, and 5
  %(prog)s --figure 3         Generate only figure 3
  %(prog)s --all              Generate all figures (default)
  %(prog)s                    Generate all figures (default)
        """,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--figures",
        type=int,
        nargs="+",
        choices=range(1, 9),
        metavar="N",
        help="Generate specific figures (1-8). Can specify multiple figures.",
    )
    group.add_argument(
        "--figure",
        type=int,
        choices=range(1, 9),
        metavar="N",
        help="Generate a single figure (1-8).",
    )
    group.add_argument(
        "--all", action="store_true", help="Generate all figures (default behavior)."
    )

    args = parser.parse_args()

    # Determine which figures to generate
    if args.figures:
        figures_to_generate = sorted(set(args.figures))
    elif args.figure:
        figures_to_generate = [args.figure]
    else:
        # Default: generate all figures
        figures_to_generate = list(range(1, 9))

    # Setup environment
    setup_environment()

    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)

    # Figure generation functions
    figure_functions = {
        1: generate_figure_1,
        2: generate_figure_2,
        3: generate_figure_3,
        4: generate_figure_4,
        5: generate_figure_5,
        6: generate_figure_6,
        7: generate_figure_7,
        8: generate_figure_8,
    }

    print(f"Generating figures: {figures_to_generate}")
    print("=" * 50)

    # Generate requested figures
    for fig_num in figures_to_generate:
        try:
            figure_functions[fig_num]()
            print()
        except Exception as e:
            print(f"Error generating Figure {fig_num}: {e}")
            sys.exit(1)

    print("=" * 50)
    print("All requested figures generated successfully!")


if __name__ == "__main__":
    main()
