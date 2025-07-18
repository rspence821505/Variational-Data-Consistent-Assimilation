import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import numpy as np
import os


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


def plot_models(models, config, results, obs_indices, plot_params):
    # Generate xticks for multiples of 25
    plot_ticks = jnp.arange(0, config.total_steps, step=config.window_size)
    mask = -1
    colors = ["darkviolet", "orange", "deepskyblue"]
    line_styles = ["dashed", "dashed", "dashed"]

    # Create figure and subplots
    with plt.rc_context(plot_params):
        fig, axes = plt.subplots(3, 1, sharex=True)
        fig.tight_layout(pad=3.0)  # Add padding between subplots

        # Create plots for each dimension
        for i in range(3):
            ax = axes[i]

            # Plot true signal
            ax.plot(
                results["trajectory"][obs_indices, :][:mask, i],
                color="green",
                label=f"True Signal",
            )

            # Plot model analyses
            for idx, model in enumerate(models):
                ax.plot(
                    results["analysis"][model][obs_indices, :][:mask, i],
                    color=colors[idx],
                    linestyle=line_styles[idx],
                    label=f"{model} Analysis",
                )

            # Plot observations if available
            if i < config.obs_dim:
                ax.plot(
                    results["y_obs"].T[:mask, i],
                    "o",
                    color="red",
                    label=f"Observations",
                )

            window = len(results["trajectory"][obs_indices, :][::mask, i]) - 1

            # Add vertical lines for each value in plot_ticks
            for tick in jnp.linspace(0, window, num=len(plot_ticks)):
                ax.axvline(x=tick, color="gray", linestyle="--", linewidth=0.5)

            # Set x-ticks using plot_ticks
            ax.set_xticks(jnp.linspace(0, window, num=len(plot_ticks)))
            ax.set_xticklabels(plot_ticks.astype(int), rotation=45)

            # Set labels
            ax.set_ylabel(f"$Z_{i+1}$")
            if i == 2:  # Only add xlabel to bottom subplot
                ax.set_xlabel("Time")

        # Add single legend at the top of the figure
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            bbox_to_anchor=(0.5, 1.0),  # Position above all subplots
            loc="center",
            ncol=len(models) + 2,  # All items in one row
            bbox_transform=fig.transFigure,
        )

        plt.show()


def plot_metric(df, xlabel, ylabel, plot_params, file_name=None, save=False):
    with plt.rc_context(plot_params):
        df.T.plot(style=["r-o", "b-o", "g-o", "y-o", "m-o"])
        plt.xlabel(xlabel, fontweight="bold")
        plt.ylabel(ylabel, fontweight="bold")
        plt.legend(loc="upper left")
        if save:
            plt.savefig(f"{file_name}.png")
        plt.show()


def plot_heatmap(df, plot_params, plot_name=None):
    # Find the first row where DCI_WME is lower than Bayes
    star_x = None
    for idx in df.index:
        if df.loc[idx]["DCI_WME"] < df.loc[idx]["Bayes"]:
            star_x = df.index.get_loc(idx)
            break

    # Find the minimum value and its location (excluding NaN values)
    min_value = df.min().min()
    min_metric = df.min().idxmin()
    min_iteration = df[min_metric].idxmin()

    with plt.rc_context(plot_params):
        ax = sns.heatmap(
            df.T,
            cmap="magma",
            annot=True,
            fmt=".2f",
            annot_kws={"size": 18},
            cbar_kws={"label": "RMSE", "fraction": 0.2, "aspect": 15, "pad": 0.015},
            linewidth=0.5,
            linecolor="white",
            xticklabels=df.index.tolist(),
            yticklabels=["Bayes", "DCI", "DCI_WME"],
        )

        # Plot yellow star for minimum value
        min_x = df.index.tolist().index(min_iteration)
        min_y = ["Bayes", "DCI", "DCI_WME"].index(min_metric)
        plt.plot(
            min_x + 0.5,
            min_y + 0.2,
            marker="*",
            color="yellow",
            markersize=25,
            markeredgecolor="white",
            markeredgewidth=2,
            zorder=3,
        )

        # Plot blue star at first occurrence where DCI_WME < Bayes
        if star_x is not None:
            star_y = ["Bayes", "DCI", "DCI_WME"].index("DCI_WME")
            plt.plot(
                star_x + 0.5,
                star_y + 0.2,
                marker="*",
                color="aqua",
                markersize=25,
                markeredgecolor="white",
                markeredgewidth=2,
                zorder=3,
            )

        plt.xlabel(r"Inflation Factor $\alpha$", fontweight="bold")
        plt.tight_layout()

        if plot_name is not None:
            plt.savefig(os.path.join("/content", plot_name))

        plt.show()


def plot_rmse_stackplot(
    rmse_scores,
    plot_params=None,
    save=False,
    filename="tarmse_plot_l63.png",
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
    filename : str, default 'tarmse_plot_l63.png'
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

        if save:
            plt.savefig(filename, dpi=300, bbox_inches="tight")

        plt.show()

    return fig, ax


def plot_cross_val(heatmap_df, plot_params, file_name=None):

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
        plt.ylabel("$\sigma_{\mathrm{obs}}$", fontsize=44, fontweight="bold")
        ax.set_xticks(
            ticks=np.arange(len(heatmap_df.columns)) + 0.5, labels=heatmap_df.columns
        )
        ax.set_yticks(
            ticks=np.arange(len(heatmap_df.index)) + 0.5, labels=heatmap_df.index
        )
        if file_name is not None:
            plt.savefig(f"{file_name}.png")
        plt.show()
