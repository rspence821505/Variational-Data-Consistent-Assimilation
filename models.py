import jax.numpy as jnp
import jax
from functools import partial

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
jax.config.update("jax_enable_x64", True)


def henon_step(x, dt, params):
    """
    One step of the Henon map.
    parameter a controls the amount of stretching
    and the parameter b controls the thickness
    of folding.
    """
    a, b = params
    dx = 1 - (a * x[0] ** 2) + x[1]
    dy = b * x[0]
    return jnp.array([dx, dy])


def lorenz63_step(x, dt, params):
    """One step of the Lorenz 63 model."""
    SIGMA, RHO, BETA = params
    dx = SIGMA * (x[1] - x[0])
    dy = x[0] * (RHO - x[2]) - x[1]
    dz = x[0] * x[1] - BETA * x[2]
    return x + dt * jnp.array([dx, dy, dz])


def lorenz96_step(x, dt, params):
    """One step of the Lorenz 96 model in a vectorized manner."""
    F = params
    N = x.size
    dx = (jnp.roll(x, -1) - jnp.roll(x, 2)) * jnp.roll(x, 1) - x + F
    return x + dt * dx


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def integrate(x0, steps, dt, params, model_step_fn):
    """Integrates a dynamical model over multiple steps."""

    def step_fn(x, _):
        return model_step_fn(x, dt, params), x

    return jax.lax.scan(step_fn, x0, None, length=steps)
