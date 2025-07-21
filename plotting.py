import os

import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


plot_params = {
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "mathtext.default": "bf",
    "font.family": "STIXGeneral",
    "font.weight": "bold",
}
plt.rcParams.update(plot_params)

sns.set_palette("bright")


def plot_rmse_stackplot(
    rmse_scores,
    plot_params=None,
    show_plot=False,
    save_plot=False,
    file_name="tarmse_plot_l63.png",
    colors=None,
    labels=None,
    figsize=None,
):
    """
    Create a stacked area plot of RMSE scores with overlaid line plots.

    Parameters:
    -----------
    rmse_scores : pandas.DataFrame
        DataFrame with columns ['DCI_WME', 'Bayes', 'DCI'] and time index
    plot_params : dict, optional
        Matplotlib rc parameters to use as context
    save : bool, default False
        Whether to save the figure
    file_name : str, default 'tarmse_plot_l63.png'
        Filename for saved figure
    colors : list, optional
        Custom colors for the plot. Default is ['#B28B6B', '#FFA733', '#6ACC65']
    labels : list, optional
        Custom labels for the series. Default is ['DC-WME 4D-Var', '4D-Var', 'DC 4D-Var']
    figsize : tuple, optional
        Figure size (width, height)

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """

    # Set default colors and labels
    if colors is None:
        colors = ["#B28B6B", "#FFA733", "#6ACC65"]  # Brown, Orange, Green
    if labels is None:
        labels = ["DC-WME 4D-Var", "4D-Var", "DC 4D-Var"]

    # Use plot context if provided
    context = plt.rc_context(plot_params) if plot_params else plt.rc_context()

    with context:
        # Time axis
        time = rmse_scores.index

        # Stack data in order: bottom to top (each row is a time series)
        series = [
            rmse_scores["DCI_WME"].values,
            rmse_scores["Bayes"].values,
            rmse_scores["DCI"].values,
        ]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Stackplot fills the areas
        stacked = ax.stackplot(time, series, labels=labels, colors=colors, alpha=0.55)

        # Compute top edges and plot them
        cumulative = np.zeros_like(series[0])
        for i, (y, color) in enumerate(zip(series, colors)):
            cumulative += y
            ax.plot(
                time,
                cumulative,
                color=color,
                linewidth=5.5,
                label="_nolegend_",  # avoids duplicate legend
            )

        # Axes and styling
        ax.set_ylim(bottom=0)
        ax.set_xlim([time.min(), time.max()])
        ax.grid(visible=True, alpha=0.3, linestyle="--", linewidth=1)
        ax.set_xlabel("Time Steps", fontweight="bold")
        ax.set_ylabel("RMSE", fontweight="bold")
        ax.legend(loc="best", framealpha=0.95, fancybox=True, shadow=True)
        sns.despine()
        plt.tight_layout()

        if save_plot:
            plt.savefig(
                os.path.join("figures", file_name), dpi=300, bbox_inches="tight"
            )
        if show_plot:
            plt.show()

    return fig, ax


def plot_cross_val(
    heatmap_df, plot_params, show_plot=False, save_plot=False, file_name=None
):

    mat_array = jnp.asarray(heatmap_df)

    # Create the plot

    with plt.rc_context(plot_params):
        _, ax = plt.subplots()
        pcm = ax.pcolormesh(
            mat_array,
            edgecolors="white",
            cmap="plasma",
        )

        # Format the plot
        plt.colorbar(pcm, ax=ax)
        plt.xlabel("Observation Density", fontweight="bold")
        plt.ylabel(r"$\sigma_{\mathrm{obs}}$", fontsize=44, fontweight="bold")
        ax.set_xticks(
            ticks=np.arange(len(heatmap_df.columns)) + 0.5, labels=heatmap_df.columns
        )
        ax.set_yticks(
            ticks=np.arange(len(heatmap_df.index)) + 0.5, labels=heatmap_df.index
        )
        if save_plot:
            plt.savefig(
                os.path.join("figures", file_name), dpi=300, bbox_inches="tight"
            )

        if show_plot:
            plt.show()


def create_sigma_histogram(
    data,
    plot_params,
    file_name="sigma_b_bound_distribution.png",
    xlabel="Estimated Lower Bound on $\\sigma_b^2$",
    ylabel="Frequency",
    color="darkmagenta",
    bins=35,
    save_plot=False,
    show_plot=False,
    dpi=300,
):
    """
    Create a histogram with KDE overlay for sigma bound distribution.

    Parameters:
    -----------
    data : array-like
        The data to plot
    plot_params : dict
        Matplotlib rc parameters for styling
    filename : str, optional
        Path to save the figure (default: "sigma_b_bound_distribution.png")
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    color : str, optional
        Color for the histogram and KDE (default: "darkmagenta")
    bins : int, optional
        Number of bins for histogram (default: 35)
    save_plot : bool, optional
        Whether to save the plot (default: True)
    show_plot : bool, optional
        Whether to display the plot (default: True)
    dpi : int, optional
        DPI for saved figure (default: 300)

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """

    with plt.rc_context(plot_params):
        # Create figure and axis
        fig, ax = plt.subplots()

        # Set background color for better contrast
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Create histogram with improved styling
        sns.histplot(
            data,
            kde=True,
            log_scale=(True, False),
            color=color,
            edgecolor="white",  # White edges for cleaner look
            linewidth=0.8,
            alpha=0.8,  # Slight transparency
            bins=bins,  # More bins for smoother distribution
            line_kws={"linewidth": 3, "color": color, "alpha": 0.9},
        )

        # Add mean line with improved styling
        mean_val = data.mean()
        ax.axvline(
            mean_val,
            color="#e74c3c",
            linestyle="--",
            linewidth=4,
            alpha=1.0,
            label=f"Mean = {mean_val:.1f}",
        )

        # Improve axis labels and title
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")

        # Enhance legend
        legend = ax.legend(
            loc="upper right",
            frameon=False,
            fancybox=True,
            shadow=True,
            framealpha=0.9,
            facecolor="white",
            edgecolor="white",
        )

        # Add subtle grid for better readability
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        # Improve tick formatting
        ax.tick_params(axis="both", which="major", length=8, width=1.2)

        plt.tight_layout()

        # Save plot if requested
        if save_plot:
            plt.savefig(
                os.path.join("figures", file_name),
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )

        # Show plot if requested
        if show_plot:
            plt.show()

    return fig, ax


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import traceback


def create_rmse_heatmap(
    df_melt,
    plot_params,
    index_col="Method",
    columns_col="Sigma",
    values_col="RMSE",
    file_name="obs_inflation_levels_l63.png",
    xlabel=r"$\sigma_{obs}$",
    ylabel="",
    cmap="YlOrRd",
    annot=True,
    fmt=".2f",
    linewidth=2,
    linecolor="black",
    mark_minimum=True,
    star_color="deepskyblue",
    star_outline_color="white",
    star_size=32,
    star_outline_size=36,
    xlabel_fontsize=36,
    cbar_label="RMSE",
    save_plot=False,
    show_plot=False,
    dpi=300,
):
    """
    Create a heatmap with optional minimum value markers.
    Enhanced with comprehensive error handling for debugging.
    """

    # print("Starting heatmap creation...")

    try:
        # Input validation
        if df_melt is None or df_melt.empty:
            raise ValueError("Input dataframe is None or empty")

        required_cols = [index_col, columns_col, values_col]
        missing_cols = [col for col in required_cols if col not in df_melt.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataframe: {missing_cols}")

        # print(f"DataFrame shape: {df_melt.shape}")
        # print(f"DataFrame columns: {list(df_melt.columns)}")
        # print(
        #     f"Using columns - Index: {index_col}, Columns: {columns_col}, Values: {values_col}"
        # )

    except Exception as e:
        print(f"ERROR in input validation: {e}")
        traceback.print_exc()
        return None, None, None

    try:
        # Create pivot table
        pivot = df_melt.pivot(index=index_col, columns=columns_col, values=values_col)

        # Check for NaN values
        if pivot.isnull().any().any():
            print("WARNING: Pivot table contains NaN values")
            print(f"NaN count: {pivot.isnull().sum().sum()}")

    except Exception as e:
        print(f"ERROR in pivot table creation: {e}")
        traceback.print_exc()
        return None, None, None

    try:
        # Set up plotting context
        with plt.rc_context(plot_params):
            fig, ax = plt.subplots(figsize=(10, 8))

            try:
                # Create the heatmap
                heatmap = sns.heatmap(
                    pivot,
                    annot=annot,
                    fmt=fmt,
                    linewidth=linewidth,
                    linecolor=linecolor,
                    cmap=cmap,
                    cbar_kws={"label": cbar_label},
                    ax=ax,
                )
                # print("Heatmap created successfully")

            except Exception as e:
                print(f"ERROR in heatmap creation: {e}")
                traceback.print_exc()
                plt.close(fig)
                return None, None, None

            try:
                # Mark minimum values with stars if requested
                if mark_minimum:
                    # print("Adding minimum value markers...")
                    for col_idx, col in enumerate(pivot.columns):
                        try:

                            # Get the minimum value and its index
                            min_row_idx = pivot[col].idxmin()

                            # Get position in the pivot table
                            row_idx = pivot.index.get_loc(min_row_idx)

                            # Add white outline (larger star)
                            ax.text(
                                col_idx + 0.5,
                                row_idx + 0.3,
                                "★",
                                horizontalalignment="center",
                                verticalalignment="center",
                                color=star_outline_color,
                                fontsize=star_outline_size,
                                fontweight="bold",
                            )

                            # Add colored star on top (smaller)
                            ax.text(
                                col_idx + 0.5,
                                row_idx + 0.3,
                                "★",
                                horizontalalignment="center",
                                verticalalignment="center",
                                color=star_color,
                                fontsize=star_size,
                                fontweight="bold",
                            )
                            # print(f"Star added for column {col}")

                        except Exception as e:
                            print(
                                f"ERROR adding star for column {col} (index {col_idx}): {e}"
                            )
                            traceback.print_exc()
                            # Continue with other columns
                            continue

            except Exception as e:
                print(f"ERROR in minimum value marking section: {e}")
                traceback.print_exc()
                # Continue without stars

            try:
                # Set labels

                plt.xlabel(xlabel, fontsize=xlabel_fontsize)
                plt.ylabel(ylabel)

            except Exception as e:
                print(f"ERROR setting labels: {e}")
                traceback.print_exc()

            try:
                # Make colorbar label bold

                cbar = ax.collections[0].colorbar
                cbar.set_label(cbar_label, fontweight="bold")

            except Exception as e:
                print(f"ERROR setting colorbar label: {e}")
                traceback.print_exc()

            try:
                plt.tight_layout()

            except Exception as e:
                print(f"ERROR in tight_layout: {e}")
                traceback.print_exc()

            try:
                # Save plot if requested
                if save_plot:

                    # Check if figures directory exists
                    figures_dir = "figures"
                    if not os.path.exists(figures_dir):
                        print(f"Creating directory: {figures_dir}")
                        os.makedirs(figures_dir)

                    full_path = os.path.join(figures_dir, file_name)
                    plt.savefig(full_path, dpi=dpi, bbox_inches="tight")

            except Exception as e:
                print(f"ERROR saving plot: {e}")
                traceback.print_exc()

            try:
                # Show plot if requested
                if show_plot:
                    plt.show()

            except Exception as e:
                print(f"ERROR showing plot: {e}")
                traceback.print_exc()

        return fig, ax, pivot

    except Exception as e:
        print(f"ERROR in plotting context: {e}")
        traceback.print_exc()
        return None, None, None


# def create_rmse_heatmap(
#     df_melt,
#     plot_params,
#     index_col="Method",
#     columns_col="Sigma",
#     values_col="RMSE",
#     file_name="obs_inflation_levels_l63.png",
#     xlabel=r"$\sigma_{obs}$",
#     ylabel="",
#     cmap="YlOrRd",
#     annot=True,
#     fmt=".2f",
#     linewidth=2,
#     linecolor="black",
#     mark_minimum=True,
#     star_color="deepskyblue",
#     star_outline_color="white",
#     star_size=32,
#     star_outline_size=36,
#     xlabel_fontsize=36,
#     cbar_label="RMSE",
#     save_plot=False,
#     show_plot=False,
#     dpi=300,
# ):
#     """
#     Create a heatmap with optional minimum value markers.

#     Parameters:
#     -----------
#     df_melt : pandas.DataFrame
#         Melted dataframe with data to plot
#     plot_params : dict
#         Matplotlib rc parameters for styling
#     index_col : str, optional
#         Column to use as heatmap index (default: 'Method')
#     columns_col : str, optional
#         Column to use as heatmap columns (default: 'Sigma')
#     values_col : str, optional
#         Column to use as heatmap values (default: 'RMSE')
#     filename : str, optional
#         Path to save the figure
#     xlabel : str, optional
#         X-axis label (default: r"$\sigma_{obs}$")
#     ylabel : str, optional
#         Y-axis label (default: '')
#     cmap : str, optional
#         Colormap for heatmap (default: "YlOrRd")
#     annot : bool, optional
#         Whether to annotate cells with values (default: True)
#     fmt : str, optional
#         Format string for annotations (default: ".2f")
#     linewidth : float, optional
#         Width of lines separating cells (default: 2)
#     linecolor : str, optional
#         Color of lines separating cells (default: "black")
#     mark_minimum : bool, optional
#         Whether to mark minimum values with stars (default: True)
#     star_color : str, optional
#         Color of the star markers (default: 'deepskyblue')
#     star_outline_color : str, optional
#         Color of star outline (default: 'white')
#     star_size : int, optional
#         Size of star markers (default: 32)
#     star_outline_size : int, optional
#         Size of star outline (default: 36)
#     xlabel_fontsize : int, optional
#         Font size for x-axis label (default: 36)
#     cbar_label : str, optional
#         Label for colorbar (default: 'RMSE')
#     save_plot : bool, optional
#         Whether to save the plot (default: True)
#     show_plot : bool, optional
#         Whether to display the plot (default: True)
#     dpi : int, optional
#         DPI for saved figure (default: 300)

#     Returns:
#     --------
#     fig, ax : matplotlib figure and axis objects
#     pivot : pandas.DataFrame
#         The pivoted data used for the heatmap
#     """

#     with plt.rc_context(plot_params):
#         # Create pivot table
#         pivot = df_melt.pivot(index=index_col, columns=columns_col, values=values_col)

#         # Create the heatmap
#         ax = sns.heatmap(
#             pivot,
#             annot=annot,
#             fmt=fmt,
#             linewidth=linewidth,
#             linecolor=linecolor,
#             cmap=cmap,
#             cbar_kws={"label": cbar_label},
#         )

#         # Mark minimum values with stars if requested
#         if mark_minimum:
#             for col_idx, col in enumerate(pivot.columns):
#                 min_row_idx = pivot[col].idxmin()  # Get the index of minimum value
#                 row_idx = pivot.index.get_loc(
#                     min_row_idx
#                 )  # Get position in the pivot table

#                 # Add white outline (larger star)
#                 ax.text(
#                     col_idx + 0.5,
#                     row_idx + 0.3,
#                     "★",
#                     horizontalalignment="center",
#                     verticalalignment="center",
#                     color=star_outline_color,
#                     fontsize=star_outline_size,
#                     fontweight="bold",
#                 )

#                 # Add colored star on top (smaller)
#                 ax.text(
#                     col_idx + 0.5,
#                     row_idx + 0.3,
#                     "★",
#                     horizontalalignment="center",
#                     verticalalignment="center",
#                     color=star_color,
#                     fontsize=star_size,
#                     fontweight="bold",
#                 )

#         # Set labels
#         plt.xlabel(xlabel, fontsize=xlabel_fontsize)
#         plt.ylabel(ylabel)

#         # Make colorbar label bold
#         cbar = ax.collections[0].colorbar
#         cbar.set_label(cbar_label, fontweight="bold")

#         plt.tight_layout()

#         # Get figure object
#         fig = plt.gcf()

#         # Save plot if requested
#         if save_plot:
#             plt.savefig(
#                 os.path.join("figures", file_name), dpi=dpi, bbox_inches="tight"
#             )

#         # Show plot if requested
#         if show_plot:
#             plt.show()

#     return fig, ax, pivot


def create_dof_slope_plot(
    rmse_dofs,
    dofs,
    model_labels,
    colors,
    markers,
    plot_params,
    show_plot=False,
    save_plot=False,
    file_name="L96_rmse_dof.png",
):
    """
    Create a multi-point slope plot for DOF vs RMSE data

    Parameters:
    rmse_dofs: DataFrame with RMSE values, columns [0,2] correspond to the two models
    dofs: list/array of degrees of freedom values
    """

    with plt.rc_context(plot_params):
        fig, ax = plt.subplots(figsize=(30, 20))

        # Extract data for the two models (columns 0 and 2)
        model_data = {
            "4D-Var": rmse_dofs.iloc[:, 0].values,
            "DC-WME 4D-Var": rmse_dofs.iloc[:, 2].values,
        }

        # Plot slope lines for each model
        for model_name in model_labels:
            values = model_data[model_name]

            # Plot the main line with markers
            ax.plot(
                range(len(dofs)),
                values,
                marker=markers[model_name],
                color=colors[model_name],
                markeredgecolor="white",
                markeredgewidth=6,
                alpha=0.9,
                label=model_name,
                solid_capstyle="round",
            )

            if model_name == "4D-Var":
                # Add value labels above each point
                for i, (dof, val) in enumerate(zip(dofs, values)):
                    ax.text(
                        i,
                        val + ((values.max() - values.min()) * 0.03) + 0.01,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=45,
                        color=colors[model_name],
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="white",
                            alpha=0.8,
                            edgecolor=colors[model_name],
                            linewidth=2,
                        ),
                    )
            else:
                for i, (dof, val) in enumerate(zip(dofs, values)):
                    ax.text(
                        i,
                        val + ((values.max() - values.min()) * 0.03) - 0.085,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=45,
                        color=colors[model_name],
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="white",
                            alpha=0.8,
                            edgecolor=colors[model_name],
                            linewidth=2,
                        ),
                    )

        # Calculate and display performance metrics
        model1_vals = model_data["4D-Var"]
        model2_vals = model_data["DC-WME 4D-Var"]

        # Customize the plot
        ax.set_xlim(-0.3, len(dofs) - 0.7)

        # Set y-axis limits with padding
        all_values = np.concatenate([model1_vals, model2_vals])
        y_min, y_max = all_values.min(), all_values.max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.1)

        # Customize x-axis
        ax.set_xticks(range(len(dofs)))
        ax.set_xticklabels([f"{dof}" for dof in dofs], fontweight="bold")

        # Add labels and title
        ax.set_xlabel("Degrees of Freedom", fontweight="bold")
        ax.set_ylabel("Time Averaged RMSE", fontweight="bold")

        # Remove top and right spines
        sns.despine()

        # Add connecting lines between points (subtle)
        for model_name in model_labels:
            values = model_data[model_name]
            ax.plot(
                range(len(dofs)),
                values,
                color=colors[model_name],
                linewidth=4,
                alpha=0.3,
                zorder=0,
            )

        # Highlight the best performance point for each model
        for idx, model_name in enumerate(model_labels):
            values = model_data[model_name]
            best_idx = np.argmin(values)
            if model_name == "4D-Var":
                ax.scatter(
                    best_idx,
                    values[best_idx],
                    s=800,
                    color=colors[model_name],
                    marker="*",
                    edgecolor="gold",
                    linewidth=6,
                    zorder=10,
                    alpha=0.9,
                    label="Optimal Performance",
                )
            else:
                ax.scatter(
                    best_idx,
                    values[best_idx],
                    s=800,
                    color=colors[model_name],
                    marker="*",
                    edgecolor="gold",
                    linewidth=6,
                    zorder=10,
                    alpha=0.9,
                )

        # Add legend
        ax.legend(
            loc="upper left",
            framealpha=0.9,
            fancybox=True,
            shadow=True,
        )

        plt.tight_layout()

        if save_plot:
            plt.savefig(
                os.path.join("figures", file_name), dpi=300, bbox_inches="tight"
            )

        if show_plot:
            plt.show()


def create_combined_rmse_bias_plot(
    rmse_scores_background,
    rmse_scores_analysis,
    bias_scores_background,
    bias_scores_analysis,
    model_configs,
    plot_params,
    normalize_series,
    file_name="combined_rmse_bias_plot_l96.png",
    figsize=None,
    background_color="#3B4CC0",  # Indigo
    analysis_color="#FABD2F",
    pos_color="#1FA187",
    neg_color="#D43D51",
    stackplot_alpha=0.5,
    line_width=2,
    bar_width=0.8,
    grid_alpha=0.5,
    grid_linewidth=0.7,
    legend_fontsize=20,
    scale_text_fontsize=14,
    scale_text_color="gray",
    bias_ylim=(-1.1, 1.1),
    save_plot=False,
    show_plot=False,
    dpi=300,
):
    """
    Create a combined RMSE and bias analysis plot with multiple models.

    Parameters:
    -----------
    rmse_scores_background : dict
        Dictionary with model names as keys and background RMSE scores as values
    rmse_scores_analysis : dict
        Dictionary with model names as keys and analysis RMSE scores as values
    bias_scores_background : dict
        Dictionary with model names as keys and background bias scores as values
    bias_scores_analysis : dict
        Dictionary with model names as keys and analysis bias scores as values
    model_configs : dict
        Dictionary with model configurations. Each model should have 'col' and 'title' keys
        Example: {'model1': {'col': 0, 'title': 'Model 1'}, ...}
    plot_params : dict
        Matplotlib rc parameters for styling
    normalize_series : function
        Function to normalize bias series. Should return (normalized_values, scale)
    filename : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height). If None, uses matplotlib default
    background_color : str, optional
        Color for background data (default: '#1f77b4')
    analysis_color : str, optional
        Color for analysis data (default: '#ff7f0e')
    pos_color : str, optional
        Color for positive bias bars (default: '#d62728')
    neg_color : str, optional
        Color for negative bias bars (default: '#2ca02c')
    stackplot_alpha : float, optional
        Alpha transparency for stackplot (default: 0.5)
    line_width : float, optional
        Width of lines in stackplot (default: 2)
    bar_width : float, optional
        Width of bias bars (default: 0.8)
    grid_alpha : float, optional
        Alpha transparency for grid (default: 0.5)
    grid_linewidth : float, optional
        Width of grid lines (default: 0.7)
    legend_fontsize : int, optional
        Font size for legend (default: 20)
    scale_text_fontsize : int, optional
        Font size for scale text (default: 14)
    scale_text_color : str, optional
        Color for scale text (default: 'gray')
    bias_ylim : tuple, optional
        Y-axis limits for bias plots (default: (-1.1, 1.1))
    save_plot : bool, optional
        Whether to save the plot (default: True)
    show_plot : bool, optional
        Whether to display the plot (default: True)
    dpi : int, optional
        DPI for saved figure (default: 300)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    ax : numpy.ndarray
        Array of subplot axes (3 rows x N columns)

    Notes:
    ------
    - Requires a normalize_series function that takes a series and returns (normalized_values, scale)
    - The number of columns is determined by the number of models in model_configs
    - Each model in model_configs should have 'col' and 'title' keys
    """

    with plt.rc_context(plot_params):
        # Determine number of columns from model_configs
        n_cols = len(model_configs)

        # Create figure with custom size if provided
        if figsize:
            fig, ax = plt.subplots(3, n_cols, sharex=True, figsize=figsize)
        else:
            fig, ax = plt.subplots(3, n_cols, sharex=True)

        # Ensure ax is 2D even for single column
        if n_cols == 1:
            ax = ax.reshape(-1, 1)

        legend_handles = []
        legend_labels = []

        for model, model_config in model_configs.items():
            col = model_config["col"]

            # ─── Row 0: RMSE Stacked Area Plot ───────────────
            current_ax = ax[0, col]
            background_rmse = rmse_scores_background[model].values
            analysis_rmse = rmse_scores_analysis[model].values
            x_pos = np.arange(len(background_rmse))

            series = [analysis_rmse, background_rmse]
            colors = [analysis_color, background_color]

            # Stackplot fills
            current_ax.stackplot(x_pos, series, colors=colors, alpha=stackplot_alpha)

            # Top edges of each layer for clarity
            cumulative = np.zeros_like(series[0])
            for i, (y, color) in enumerate(zip(series, colors)):
                cumulative += y
                current_ax.plot(
                    x_pos,
                    cumulative,
                    color=color,
                    linewidth=line_width,
                    label="_nolegend_",
                )

            current_ax.set_title(model_config["title"], fontweight="bold")
            if col == 0:
                current_ax.set_ylabel("RMSE", fontweight="bold")
            current_ax.grid(
                axis="y", linestyle="--", linewidth=grid_linewidth, alpha=grid_alpha
            )
            current_ax.spines["top"].set_visible(False)
            current_ax.spines["right"].set_visible(False)

            # Add legend handles once
            if col == 0:
                legend_handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        color=background_color,
                        linewidth=0,
                        marker="s",
                        markersize=12,
                        label="Background",
                    ),
                    plt.Line2D(
                        [0],
                        [0],
                        color=analysis_color,
                        linewidth=0,
                        marker="s",
                        markersize=12,
                        label="Analysis",
                    ),
                    plt.Line2D(
                        [0],
                        [0],
                        color=pos_color,
                        linewidth=0,
                        marker="s",
                        markersize=12,
                        label="Positive Bias",
                    ),
                    plt.Line2D(
                        [0],
                        [0],
                        color=neg_color,
                        linewidth=0,
                        marker="s",
                        markersize=12,
                        label="Negative Bias",
                    ),
                ]
                legend_labels = [
                    "Background",
                    "Analysis",
                    "Positive Bias",
                    "Negative Bias",
                ]

            # ─── Row 1: Normalized Bias for Background ──────────────
            current_ax = ax[1, col]
            signed_bias_bg = bias_scores_background[model]
            normalized_bias, bg_scale = normalize_series(signed_bias_bg)

            for i, val in enumerate(normalized_bias):
                bar_color = pos_color if val >= 0 else neg_color
                current_ax.bar(i, val, color=bar_color, width=bar_width)

            current_ax.axhline(0, color="black", linewidth=1.0)
            current_ax.set_ylim(*bias_ylim)
            current_ax.text(
                0.95,
                0.85,
                f"Max: {bg_scale:.2f}",
                transform=current_ax.transAxes,
                ha="right",
                fontsize=scale_text_fontsize,
                color=scale_text_color,
            )

            if col == 0:
                current_ax.set_ylabel("Bias (Background)", fontweight="bold")

            current_ax.grid(
                axis="y", linestyle="--", linewidth=grid_linewidth, alpha=grid_alpha
            )
            current_ax.spines["top"].set_visible(False)
            current_ax.spines["right"].set_visible(False)

            # ─── Row 2: Normalized Bias for Analysis ────────────────
            current_ax = ax[2, col]
            signed_bias_an = bias_scores_analysis[model]
            normalized_bias, an_scale = normalize_series(signed_bias_an)

            for i, val in enumerate(normalized_bias):
                bar_color = pos_color if val >= 0 else neg_color
                current_ax.bar(i, val, color=bar_color, width=bar_width)

            current_ax.axhline(0, color="black", linewidth=1.0)
            current_ax.set_ylim(*bias_ylim)
            current_ax.text(
                0.95,
                0.85,
                f"Max: {an_scale:.2f}",
                transform=current_ax.transAxes,
                ha="right",
                fontsize=scale_text_fontsize,
                color=scale_text_color,
            )

            current_ax.set_xlabel("Time Steps", fontweight="bold")
            if col == 0:
                current_ax.set_ylabel("Bias (Analysis)", fontweight="bold")

            current_ax.grid(
                axis="y", linestyle="--", linewidth=grid_linewidth, alpha=grid_alpha
            )
            current_ax.spines["top"].set_visible(False)
            current_ax.spines["right"].set_visible(False)

        # ─── Shared Legend ───────────────────────────────────────
        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=4,
            fontsize=legend_fontsize,
            frameon=False,
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)

        # Save plot if requested
        if save_plot:
            plt.savefig(
                os.path.join("figures", file_name), bbox_inches="tight", dpi=dpi
            )

        # Show plot if requested
        if show_plot:
            plt.show()

    return fig, ax


# Mixed layout with plots + stacked bar chart (simplified)
def create_mixed_layout_custom(
    time,
    rmse_4dvar,
    rmse_dci,
    rmse_dci_wme,
    window_sizes,
    colors,
    plot_params,
    show_plot=False,
    save_plot=False,
    file_name="window_size_plot_l96.png",
):
    """Mixed layout with your data using window sizes 2, 5, and 10 - with stacked bar chart"""

    with plt.rc_context(plot_params):
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        # Store line handles and labels for legend
        handles, labels = None, None

        # Three time series plots
        positions = [(0, 0), (0, 1), (1, 0)]

        for idx, w in enumerate(window_sizes):
            ax = fig.add_subplot(gs[positions[idx]])

            # Your original plotting code adapted
            (line1,) = ax.plot(
                time,
                rmse_4dvar[w],
                label="4D-Var",
                color=colors["4D-Var"],
                linestyle="-",
                linewidth=2.5,
            )
            (line2,) = ax.plot(
                time,
                rmse_dci[w],
                label="DC 4D-Var",
                color=colors["DC 4D-Var"],
                linestyle="--",
                linewidth=2.5,
            )
            (line3,) = ax.plot(
                time,
                rmse_dci_wme[w],
                label="DC-WME 4D-Var",
                color=colors["DC-WME 4D-Var"],
                linestyle=(0, (3, 1, 1, 1)),
                linewidth=2.5,
            )

            # Store handles from first plot for legend
            if handles is None and labels is None:
                handles = [line1, line2, line3]
                labels = [line.get_label() for line in handles]

            ax.set_title(f"Window Size = {w}", fontweight="bold")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
            ax.tick_params(axis="both", which="major", pad=8)

            # Your original labeling logic
            if idx >= 0:  # Bottom row (indices 1, 2)
                ax.set_xlabel("Time Steps", fontweight="bold")
            if idx % 2 == 0:  # Left column (indices 0, 2)
                ax.set_ylabel("RMSE", fontweight="bold")

        # Stacked bar chart comparison (mean RMSE) in position (1,1)
        ax_bar = fig.add_subplot(gs[1, 1])

        # Calculate mean RMSE values for each method and window size
        var_means = [np.mean(rmse_4dvar[w]) for w in window_sizes]
        dci_means = [np.mean(rmse_dci[w]) for w in window_sizes]
        dci_wme_means = [np.mean(rmse_dci_wme[w]) for w in window_sizes]

        # Create stacked bars
        x = np.arange(len(window_sizes))
        width = 0.6  # Wider bars for stacked chart

        # Stack the bars - each method stacked on top of the previous
        bars1 = ax_bar.bar(
            x,
            var_means,
            width,
            label="4D-Var",
            color=colors["4D-Var"],
            alpha=0.8,
            edgecolor="white",
            linewidth=2,
        )
        bars2 = ax_bar.bar(
            x,
            dci_means,
            width,
            bottom=var_means,
            label="DC 4D-Var",
            color=colors["DC 4D-Var"],
            alpha=0.8,
            edgecolor="white",
            linewidth=2,
        )
        bars3 = ax_bar.bar(
            x,
            dci_wme_means,
            width,
            bottom=np.array(var_means) + np.array(dci_means),
            label="DC-WME 4D-Var",
            color=colors["DC-WME 4D-Var"],
            alpha=0.8,
            edgecolor="white",
            linewidth=2,
        )

        # Add value labels on each segment of the stacked bars
        for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
            # Label for first segment (4D-Var)
            height1 = bar1.get_height()
            ax_bar.text(
                bar1.get_x() + bar1.get_width() / 2.0,
                height1 / 2,
                f"{var_means[i]:.2f}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white",
            )

            # Label for second segment (DC 4D-Var)
            height2 = bar2.get_height()
            y_pos2 = var_means[i] + height2 / 2
            ax_bar.text(
                bar2.get_x() + bar2.get_width() / 2.0,
                y_pos2,
                f"{dci_means[i]:.2f}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white",
            )

            # Label for third segment (DC-WME 4D-Var)
            height3 = bar3.get_height()
            y_pos3 = var_means[i] + dci_means[i] + height3 / 2
            ax_bar.text(
                bar3.get_x() + bar3.get_width() / 2.0,
                y_pos3,
                f"{dci_wme_means[i]:.2f}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white",
            )

        ax_bar.set_xlabel("Window Size", fontweight="bold")
        ax_bar.set_title("Time Averaged RMSE", fontweight="bold")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(window_sizes)
        ax_bar.grid(True, alpha=0.3, axis="y")
        ax_bar.tick_params(axis="both", which="major", pad=8)
        ax_bar.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax_bar.legend(loc="upper right")

        plt.tight_layout(rect=[0, 0, 1, 0.9])

        if save_plot:
            plt.savefig(
                os.path.join("figures", file_name), dpi=300, bbox_inches="tight"
            )

        if show_plot:
            plt.show()


def create_stacked_bar_plot(
    x_values,
    data_series,
    labels,
    colors,
    x_labels,
    plot_params,
    file_name="comp_cost_plot_l96.png",
    xlabel="Degrees of Freedom",
    ylabel="Runtime (seconds)",
    bar_width=0.5,
    edge_color="white",
    edge_linewidth=1,
    annotation_fontsize=22,
    annotation_color="white",
    annotation_fontweight="bold",
    show_annotations=True,
    annotation_format="{:.1f}",
    legend_position="upper center",
    legend_bbox=(0.5, 1.12),
    legend_ncol=3,
    show_y_ticks=False,
    grid_linestyle="--",
    grid_linewidth=0.8,
    grid_alpha=0.7,
    save_plot=False,
    show_plot=False,
):
    """
    Create a stacked bar chart with value annotations on each segment.

    This function uses plot_params for most styling (figsize, DPI, fonts, etc.)
    and only requires essential data and layout parameters.

    Parameters:
    -----------
    x_values : array-like
        X-axis positions for the bars
    data_series : list of array-like
        List of data arrays to stack. Each array should have the same length as x_values
        Example: [bayes_times, dci_times, dci_wme_times]
    labels : list of str
        Labels for each data series (for legend)
    colors : list of str or dict
        Colors for each data series. Can be a list or dict with series names
    x_labels : array-like
        Labels for x-axis ticks
    plot_params : dict
        Matplotlib rc parameters for styling. Should include:
        - 'figure.figsize': figure size
        - 'savefig.dpi': DPI for saved figures
        - 'legend.fontsize': legend font size
        - 'legend.frameon': whether to show legend frame
        - 'axes.labelsize': axis label font size
        - Other font and styling parameters
    filename : str, optional
        Path to save the figure
    xlabel : str, optional
        X-axis label (default: 'Degrees of Freedom')
    ylabel : str, optional
        Y-axis label (default: 'Runtime (seconds)')
    bar_width : float, optional
        Width of the bars (default: 0.5)
    edge_color : str, optional
        Color of bar edges (default: 'white')
    edge_linewidth : float, optional
        Width of bar edges (default: 1)
    annotation_fontsize : int, optional
        Font size for value annotations (default: 22)
    annotation_color : str, optional
        Color for value annotations (default: 'white')
    annotation_fontweight : str, optional
        Font weight for annotations (default: 'bold')
    show_annotations : bool, optional
        Whether to show value annotations on bars (default: True)
    annotation_format : str, optional
        Format string for annotations (default: '{:.1f}')
    legend_position : str, optional
        Legend position (default: 'upper center')
    legend_bbox : tuple, optional
        Legend bbox_to_anchor (default: (0.5, 1.12))
    legend_ncol : int, optional
        Number of legend columns (default: 3)
    show_y_ticks : bool, optional
        Whether to show y-axis ticks (default: False)
    grid_linestyle : str, optional
        Grid line style (default: '--')
    grid_linewidth : float, optional
        Grid line width (default: 0.8)
    grid_alpha : float, optional
        Grid transparency (default: 0.7)
    save_plot : bool, optional
        Whether to save the plot (default: True)
    show_plot : bool, optional
        Whether to display the plot (default: True)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    ax : matplotlib.axes.Axes
        The subplot axes object
    bars : list
        List of bar container objects for each series

    Examples:
    ---------
    # Basic usage
    fig, ax, bars = create_stacked_bar_plot(
        x, [bayes_times, dci_times, dci_wme_times],
        ['4D-Var', 'DC 4D-Var', 'DC-WME 4D-Var'],
        colors, degrees_of_freedom, plot_params
    )

    # Custom bar styling
    fig, ax, bars = create_stacked_bar_plot(
        x, [bayes_times, dci_times, dci_wme_times],
        ['4D-Var', 'DC 4D-Var', 'DC-WME 4D-Var'],
        ['steelblue', 'orange', 'darkgreen'],
        degrees_of_freedom, plot_params,
        bar_width=0.6,
        filename="custom_runtime_plot.png"
    )

    # Without annotations
    fig, ax, bars = create_stacked_bar_plot(
        x, data_series, labels, colors, x_labels, plot_params,
        show_annotations=False,
        show_y_ticks=True
    )
    """

    with plt.rc_context(plot_params):
        # Create figure (size controlled by plot_params)
        fig, ax = plt.subplots()

        # Convert colors to list if it's a dictionary
        if isinstance(colors, dict):
            color_list = list(colors.values())
        else:
            color_list = colors

        # Ensure we have enough colors
        if len(color_list) < len(data_series):
            raise ValueError(
                f"Need at least {len(data_series)} colors, got {len(color_list)}"
            )

        bars = []
        bottoms = np.zeros(len(x_values))

        # Create stacked bars
        for i, (data, label, color) in enumerate(zip(data_series, labels, color_list)):
            bar = ax.bar(
                x_values,
                data,
                bar_width,
                bottom=bottoms,
                label=label,
                color=color,
                edgecolor=edge_color,
                linewidth=edge_linewidth,
            )
            bars.append(bar)

            # Update bottoms for next stack
            bottoms += np.array(data)

        # Add annotations if requested
        if show_annotations:
            for i in range(len(x_values)):
                y_base = 0
                for j, data in enumerate(data_series):
                    value = data[i]
                    # Place annotation at the center of each segment
                    ax.text(
                        x_values[i],
                        y_base + value / 2,
                        annotation_format.format(value),
                        ha="center",
                        va="center",
                        fontsize=annotation_fontsize,
                        fontweight=annotation_fontweight,
                        color=annotation_color,
                    )
                    y_base += value

        # Set axis labels (font size controlled by plot_params)
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")

        # Set x-axis ticks and labels
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels)

        # Configure y-axis ticks
        if not show_y_ticks:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        # Add legend (font size and frame controlled by plot_params)
        ax.legend(loc=legend_position, bbox_to_anchor=legend_bbox, ncol=legend_ncol)

        # Add horizontal grid lines
        ax.yaxis.grid(
            True, linestyle=grid_linestyle, linewidth=grid_linewidth, alpha=grid_alpha
        )
        ax.set_axisbelow(True)

        plt.tight_layout()

        # Save plot if requested (DPI and bbox controlled by plot_params)
        if save_plot:
            plt.savefig(os.path.join("figures", file_name))

        # Show plot if requested
        if show_plot:
            plt.show()

    return fig, ax, bars
