import jax
import jax.numpy as jnp
from functools import partial
from models import integrate

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
jax.config.update("jax_enable_x64", True)


@partial(jax.jit)
def background_loss(z, z_b, B_inv):
    """Calculate the background loss term."""
    # jax.debug.print("z: {x} ", x = z)
    diff_b = z - z_b
    return 0.5 * jnp.dot(diff_b, jnp.dot(B_inv, diff_b))


@partial(jax.jit)
def observation_loss(Qz, y_obs, R_inv):
    """Calculate the observation loss term."""
    obs_diff = Qz - y_obs.T
    return 0.5 * jnp.sum(obs_diff * (R_inv @ obs_diff))


@partial(jax.jit)
def prediction_loss(Qz, Q_zb, L_inv):
    """Calculate the prediction loss term."""
    pred_diff = Qz - Q_zb
    return 0.5 * jnp.sum(pred_diff * (L_inv @ pred_diff))


@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def get_trajectory_observations(
    z, obs_indices, H, Q_zb, total_steps, dt, params, model_step_fn
):
    """Propagate state through model and get observations."""
    _, trajectory = integrate(z, total_steps, dt, params, model_step_fn)
    trajectory_obs = trajectory[obs_indices]
    return jnp.dot(H, trajectory_obs.T)


def wme_map(Qz, y_obs, var, num_obs):
    """Calculate Weighted Mean Error terms."""
    wme = (1 / jnp.sqrt(num_obs)) * jnp.sum((Qz - y_obs.T) / jnp.sqrt(var), axis=1)
    return wme


def initialize_wme_terms(y_obs, R_inv, L_inv):
    """Initialize WME-specific terms."""
    num_obs = y_obs.shape[0]
    obs_var = jnp.diag(jnp.linalg.inv(R_inv))[0]
    L_inv_wme = (obs_var / num_obs) * L_inv
    return num_obs, obs_var, L_inv_wme


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Cost Functions \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
@partial(jax.jit, static_argnums=(9, 10, 11, 12))
def bayes_cost_function(
    z,
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
):
    """Vectorized cost function for 4D-Var with a generic model."""
    J_b = background_loss(z, z_b, B_inv)
    Qz = get_trajectory_observations(
        z, obs_indices, H, Q_zb, total_steps, dt, params, model_step_fn
    )
    J_o = observation_loss(Qz, y_obs, R_inv)
    return J_b + J_o, (J_b, J_o)


@partial(jax.jit, static_argnums=(9, 10, 11, 12))
def dci_cost_function(
    z,
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
):
    """DCI cost function variant 1."""
    J_b = background_loss(z, z_b, B_inv)
    Qz = get_trajectory_observations(
        z, obs_indices, H, Q_zb, total_steps, dt, params, model_step_fn
    )
    J_o = observation_loss(Qz, y_obs, R_inv)
    J_p = prediction_loss(Qz, Q_zb, L_inv)
    return J_b + J_o - J_p, (J_b, J_o, J_p)


@partial(jax.jit, static_argnums=(9, 10, 11, 12))
def dci_wme_cost_function(
    z,
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
):
    """DCI WME cost function variant 2."""
    num_obs, obs_var, L_inv_wme = initialize_wme_terms(y_obs, R_inv, L_inv)
    J_b = background_loss(z, z_b, B_inv)
    Qz = get_trajectory_observations(
        z, obs_indices, H, Q_zb, total_steps, dt, params, model_step_fn
    )

    obs_wme = wme_map(Qz, y_obs, obs_var, num_obs)

    J_o = 0.5 * jnp.linalg.norm(obs_wme, ord=2) ** 2

    pred_var = jnp.diag(jnp.linalg.inv(L_inv))[0]
    Qz_wme = wme_map(Qz, y_obs, obs_var, num_obs)
    Qzb_wme = wme_map(Q_zb, y_obs, obs_var, num_obs)
    pred_diff = Qz_wme - Qzb_wme

    J_p = 0.5 * jnp.sum(pred_diff * (L_inv_wme @ pred_diff))

    return J_b + J_o - J_p, (J_b, J_o, J_p)
