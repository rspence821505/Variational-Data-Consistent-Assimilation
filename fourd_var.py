import jax
import jax.numpy as jnp
import jaxopt
from functools import partial
from typing import Callable, Tuple, Any
from models import integrate

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
jax.config.update("jax_enable_x64", True)


# @partial(jax.jit, static_argnums=(9, 10, 11, 12))
def optimize_4dvar(
    z0: jnp.ndarray,
    z_b: jnp.ndarray,
    y_obs: jnp.ndarray,
    obs_indices: jnp.ndarray,
    H: jnp.ndarray,
    B_inv: jnp.ndarray,
    R_inv: jnp.ndarray,
    L_inv: jnp.ndarray,
    Q_zb: jnp.ndarray,
    total_steps: int,
    dt: float,
    params: tuple,
    cost_function: Callable,
    model_step_fn: Callable,
) -> Tuple[jnp.ndarray, jaxopt.OptStep]:
    """
    Perform 4D-Var optimization to minimize the cost function using the JaxOpt's LBFGS solver.

    Parameters
    ----------
    z0 : jnp.ndarray
        The initial state estimate.
    z_b : jnp.ndarray
        The background state estimate. This is z_init in documentation
    y_obs : jnp.ndarray
        The observed data.
    obs_indices : jnp.ndarray
        Indices of the observations taken from the state vector.
    H : jnp.ndarray
        The observation operator matrix.
    B_inv : jnp.ndarray
        The inverse of the background error covariance matrix.
    R_inv : jnp.ndarray
        The inverse of the observation error covariance matrix.
    L_inv : jnp.ndarray
        The inverse of the predicted covariance matrix.
    total_steps : int
        The total number of time steps for the assimilation window.
    dt : float
        The time step size.
    params : tuple
        The model parameters.
    cost_function : Callable
        The cost function to be minimized.
    model_step_fn : Callable
        The function that defines the model dynamics.

    Returns
    -------
    Tuple[jnp.ndarray, jaxopt.OptStep]
        The optimized state estimate and the optimization state information.
    """

    def cost_fn(z0):
        total_cost, _ = cost_function(
            z0,
            z_b,
            y_obs,
            obs_indices,
            H,
            B_inv,
            R_inv,
            L_inv,
            Q_zb,
            total_steps,
            dt,
            params,
            model_step_fn,
        )
        return total_cost

    solver = jaxopt.LBFGS(fun=cost_fn, maxiter=500)
    result = solver.run(z0)

    # Get final cost components
    _, cost_components = cost_function(
        result.params,
        z_b,
        y_obs,
        obs_indices,
        H,
        B_inv,
        R_inv,
        L_inv,
        Q_zb,
        total_steps,
        dt,
        params,
        model_step_fn,
    )

    return result.params, result.state, cost_components


def multi_window_4dvar(
    true_initial_state: jnp.ndarray,
    y_obs: jnp.ndarray,
    obs_indices: jnp.ndarray,
    H: jnp.ndarray,
    B: jnp.ndarray,
    R: jnp.ndarray,
    window_size: int,
    dt: float,
    obs_frequency: int,
    num_windows: int,
    params: tuple,
    cost_fn: Callable[..., float],
    model_step_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Perform 4D-Var over multiple assimilation windows with a customizable number of observations per window.

    Parameters
    ----------
    true_initial_state : jnp.ndarray
        The true initial state for the model.
    y_obs : jnp.ndarray
        Observed data (observations).
    obs_indices : jnp.ndarray
        Indices of observed variables taken from the state vector.
    H : jnp.ndarray
        Observation operator matrix.
    B : jnp.ndarray
        Background error covariance matrix. This is Sigma_init in documentation.
    R : jnp.ndarray
        Observation error covariance matrix. This is Sigma_obs in documentation.
    window_size : int
        Length of the assimilation window in time steps.
    dt : float
        Time step size for the forward model.
    obs_frequency : int
        Frequency of observations in time steps.
    num_windows : int
        Number of assimilation windows.
    params : tuple
        Parameters for the model.
    cost_fn : Callable[..., float]
        Function to compute the 4D-Var cost.
    model_step_fn : Callable[[jnp.ndarray, float], jnp.ndarray]
        Function representing the model's time step.

    Returns
    -------
    solver_state_infos : jnp.ndarray
        Jaxopt Solver State information for each assimilation window.
    optimized_states : jnp.ndarray
        Optimized Initial states propagated back through the model during analysis in each assimilation window.
    analysis_times : jnp.ndarray
        Analysis states over the assimilation windows.
    background_times : jnp.ndarray
        Background states over the assimilation windows.
    initial_states : jnp.ndarray
        Initial states used in each assimilation window.

    Notes
    -----
    This function performs a sequential assimilation process where the state is
    iteratively updated over multiple windows. The observation operator and model
    dynamics are applied within each window.

    """

    # Get Inverse Covariance matrices
    R_inv = jnp.linalg.inv(R)
    B_inv = jnp.linalg.inv(B)

    # Get Predicted Covariance
    P = H @ B @ H.T  # This is Sigma_pred in documentation
    L_inv = jnp.linalg.inv(P)

    # Get number of observations to use for each assimilation window
    obs_per_window = window_size // obs_frequency

    # @scan_tqdm(num_windows, print_rate=1, tqdm_type="notebook", position=0, leave=True)
    def assimilation_window(
        carry: jnp.ndarray, window_idx: int
    ) -> Tuple[jnp.ndarray, Tuple[Any, ...]]:
        previous_state = carry

        # Calculate and get observed indices for the current window
        indices = jnp.arange(obs_per_window) + (window_idx * obs_per_window)
        yobs_current_window = jnp.take(y_obs, indices, axis=0)

        # Run 4D-Var for the current window
        z0 = previous_state  # Initial guess
        z_b = previous_state  # Background state

        # Get background state for plotting and reference purposes only
        _, background = integrate(z_b, window_size, dt, params, model_step_fn)

        # Get Predicted Mean
        z_b_obs = background[obs_indices]
        Q_zb = H @ z_b_obs.T

        # Find Optimized Iniial State via 4DVar
        optimized_state, solver_state_info, cost_components = optimize_4dvar(
            z0,
            z_b,
            yobs_current_window,
            obs_indices,
            H,
            B_inv,
            R_inv,
            L_inv,
            Q_zb,
            window_size,
            dt,
            params,
            cost_fn,
            model_step_fn,
        )

        # Propagate the optimized initial state through the model
        _, analysis = integrate(optimized_state, window_size, dt, params, model_step_fn)

        return analysis[-1], (
            solver_state_info,
            optimized_state,
            analysis,
            background,
            analysis[-1],
            cost_components,
        )

    # Initial State
    # init = PBar(id=1,carry=true_initial_state)

    # Run scan over all the assimilation windows
    _, (
        solver_state_infos,
        optimized_states,
        analysis_times,
        background_times,
        initial_states,
        cost_components,
    ) = jax.lax.scan(assimilation_window, true_initial_state, jnp.arange(num_windows))

    return (
        solver_state_infos,
        optimized_states,
        analysis_times,
        background_times,
        initial_states,
        cost_components,
    )
