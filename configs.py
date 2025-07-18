import jax
import jax.numpy as jnp
from dataclasses import dataclass
import dataclasses
from typing import Callable, Dict, Any, Tuple, Union
import time
from models import integrate
from fourd_var import multi_window_4dvar
from metrics import (
    rmse,
    relative_error,
    misfit,
)

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
jax.config.update("jax_enable_x64", True)


@dataclass
class ExperimentConfig:
    """Configuration for a 4D-Var experiment."""

    initial_state: jnp.ndarray
    y_obs: jnp.ndarray
    obs_indices_per_window: jnp.ndarray
    H: jnp.ndarray
    B: jnp.ndarray
    R: jnp.ndarray
    inflation_factor: float
    window_size: int
    dt: float
    obs_frequency: int
    num_windows: int
    params: tuple
    ode_system: Callable


@dataclass
class FourDVarResults:
    """Results from a 4D-Var experiment."""

    solver_state_infos: Any
    optimized_states: jnp.ndarray
    analysis: jnp.ndarray
    background: jnp.ndarray
    initial_states: jnp.ndarray
    cost_components: jnp.ndarray

    def get_stacked_results(self):
        """Stack results for postprocessing."""
        return (
            jnp.vstack(self.optimized_states),
            jnp.vstack(self.analysis),
            jnp.vstack(self.background),
        )


def calculate_metrics(
    analysis: jnp.ndarray,
    trajectory: jnp.ndarray,
    obs_indices: jnp.ndarray,
    y_obs: jnp.ndarray,
    H: jnp.ndarray,
) -> Dict[str, float]:
    """Calculate error metrics for the analysis."""

    Hz = H @ trajectory.T
    # print(f"Hz shape: {Hz.shape}")
    # print(f"y_obs shape: {y_obs.shape}")

    return {
        "rmse": rmse(analysis, trajectory).mean(),
        "relative_error": relative_error(analysis, trajectory).mean(),
        "misfit": misfit(analysis[obs_indices], y_obs, H).mean(),
    }


@dataclass
class HenonConfig:
    """Configuration parameters for Lorenz system simulation"""

    dt: float
    window_size: int
    total_steps: int
    obs_std: float
    obs_frequency: int
    state_dim: int
    obs_dim: int
    inflation_factor: float
    seed: int
    params: tuple
    true_initial_state: jnp.ndarray
    initial_guess: jnp.ndarray

    @classmethod
    def default_config(cls, **kwargs):
        """Creates a default configuration for Lorenz63 system with optional overrides.

        Args:
            **kwargs: Any configuration parameters you want to override
        """
        # Create default config
        default_config = cls(
            dt=0.01,
            window_size=20,
            total_steps=200,
            obs_std=1.2,
            obs_frequency=5,
            state_dim=3,
            obs_dim=2,
            inflation_factor=1,
            seed=0,
            params=(1.4, 0.3),
            true_initial_state=jnp.array([1.2, 1.8]),
            initial_guess=jnp.array([3.30432987, 6.83276001]),
        )

        if kwargs:
            default_config = dataclasses.replace(default_config, **kwargs)

        return default_config


@dataclass
class Lorenz63Config:
    """Configuration parameters for Lorenz system simulation"""

    dt: float
    window_size: int
    total_steps: int
    obs_std: float
    obs_frequency: int
    state_dim: int
    obs_dim: int
    inflation_factor: float
    seed: int
    params: tuple
    true_initial_state: jnp.ndarray
    initial_guess: jnp.ndarray

    @classmethod
    def default_config(cls, **kwargs):
        """Creates a default configuration for Lorenz63 system with optional overrides.

        Args:
            **kwargs: Any configuration parameters you want to override
        """
        # Create default config
        default_config = cls(
            dt=0.01,
            window_size=20,
            total_steps=200,
            obs_std=1.2,
            obs_frequency=5,
            state_dim=3,
            obs_dim=2,
            inflation_factor=1,
            seed=0,
            params=(10.0, 28.0, 8.0 / 3.0),
            true_initial_state=jnp.array([1.2, 1.8, 1.5]),
            initial_guess=jnp.array([3.30432987, 6.83276001, 2.27297259]),
        )

        if kwargs:
            default_config = dataclasses.replace(default_config, **kwargs)

        return default_config


@dataclass
class Lorenz96Config:
    """Configuration parameters for Lorenz system simulation"""

    dt: float
    window_size: int
    total_steps: int
    obs_std: float
    obs_frequency: int
    state_dim: int
    obs_dim: int
    inflation_factor: float
    seed: int
    params: tuple
    true_initial_state: jnp.ndarray
    initial_guess: jnp.ndarray

    @classmethod
    def default_config(cls, **kwargs):
        """Creates a default configuration for Lorenz96 system with optional overrides.
        Args:
            **kwargs: Any configuration parameters you want to override
        """
        # Define default values first
        default_state_dim = 40
        default_params = 8.0

        # Create default config
        default_config = cls(
            dt=0.01,
            window_size=20,
            total_steps=200,
            obs_std=1.2,
            obs_frequency=5,
            state_dim=default_state_dim,
            obs_dim=20,
            inflation_factor=1,
            seed=0,
            params=default_params,
            true_initial_state=(jnp.ones(default_state_dim) * default_params)
            .at[default_state_dim // 2]
            .set(default_params + 0.05),
            initial_guess=jnp.full(default_state_dim, default_params)
            + ((-1) ** jnp.arange(default_state_dim)),
        )

        if kwargs:

            # If state_dim was updated, update dependent parameters
            if "state_dim" in kwargs:
                new_state_dim = kwargs["state_dim"]
                default_config = dataclasses.replace(
                    default_config,
                    true_initial_state=(jnp.ones(new_state_dim) * default_config.params)
                    .at[new_state_dim // 2]
                    .set(default_config.params + 0.05),
                    initial_guess=jnp.full(new_state_dim, default_config.params)
                    + ((-1) ** jnp.arange(new_state_dim)),
                )

            default_config = dataclasses.replace(default_config, **kwargs)

        return default_config


class ObservationSystem:
    """Handles observation generation and processing"""

    def __init__(self, config: Union[Lorenz63Config, Lorenz96Config]):
        self.config = config
        self.H = jnp.eye(config.obs_dim, config.state_dim)
        self.R = jnp.eye(config.obs_dim) * (config.obs_std**2)
        self.observed_vars = jnp.arange(
            1, config.state_dim, int(config.state_dim / config.obs_dim)
        )
        self._setup_observation_indices()

    def _setup_observation_operator(self):
        """Setup observation operator matrix"""
        self.H = jnp.zeros(
            (self.config.obs_dim, self.config.state_dim), dtype=jnp.float32
        )
        self.H = self.H.at[jnp.arange(self.config.obs_dim), self.observed_vars].set(1.0)

    def _setup_observation_indices(self):
        """Setup observation indices for windows"""
        self.obs_indices_per_window = jnp.arange(
            0, self.config.window_size, self.config.obs_frequency
        )
        self.obs_indices = jnp.arange(
            0, self.config.total_steps, self.config.obs_frequency
        )

    def generate_observations(self, trajectory: jnp.ndarray) -> jnp.ndarray:
        """Generate synthetic observations from true trajectory"""
        seed = jax.random.PRNGKey(self.config.seed)

        # Generate noise in chunks matching obs_indices_per_window length
        noise_windows = []
        for i in range(len(self.obs_indices) // len(self.obs_indices_per_window)):
            noise_window = self.config.obs_std * jax.random.normal(
                seed, (self.config.obs_dim, len(self.obs_indices_per_window))
            )
            noise_windows.append(noise_window)

        obs_noise = jnp.concatenate(noise_windows, axis=1)

        return self.H @ trajectory[self.obs_indices].T + obs_noise


class LorenzSystem:
    """Main class for Lorenz system simulation and data assimilation"""

    def __init__(
        self,
        config: Union[Lorenz63Config, Lorenz96Config],
        ode_system: Callable,
        cost_functions: Dict[str, Callable],
    ):
        self.config = config
        self.ode_system = ode_system
        self.cost_functions = cost_functions
        self.obs_system = ObservationSystem(config)
        self._initialize_covariances()

    def _initialize_covariances(self):
        """Initialize background error covariance matrices"""
        self.B = jnp.eye(self.config.state_dim)
        self.B = self._generate_bcov(
            self.B, self.obs_system.R, self.config.inflation_factor
        )

    @staticmethod
    def _generate_bcov(B: jnp.ndarray, R: jnp.ndarray, inflation_factor) -> jnp.ndarray:
        """
        Adjust the covariance matrix `B` by an inflation factor until its minimum
        eigenvalue is greater than or equal to the maximum eigenvalue of `R`.
        """
        B = inflation_factor * jnp.eye(B.shape[0])
        return B

    def _create_experiment_config(
        self, observations: jnp.ndarray
    ) -> "ExperimentConfig":
        """Create configuration for experiments"""
        num_windows = observations.shape[1] // (
            self.config.window_size // self.config.obs_frequency
        )

        return ExperimentConfig(
            initial_state=self.config.initial_guess,
            y_obs=observations.T,
            obs_indices_per_window=self.obs_system.obs_indices_per_window,
            H=self.obs_system.H,
            B=self.B,
            R=self.obs_system.R,
            inflation_factor=self.config.inflation_factor,
            window_size=self.config.window_size,
            dt=self.config.dt,
            obs_frequency=self.config.obs_frequency,
            num_windows=num_windows,
            params=self.config.params,
            ode_system=self.ode_system,
        )

    def run_experiment(self) -> Dict:
        """Run the complete experiment with all cost functions"""
        # Generate true trajectory
        _, trajectory = integrate(
            self.config.true_initial_state,
            self.config.total_steps,
            self.config.dt,
            self.config.params,
            self.ode_system,
        )

        # Generate observations
        observations = self.obs_system.generate_observations(trajectory)

        # Setup experiment configuration
        self.experiment_config = self._create_experiment_config(observations)

        # Run experiments with different cost functions
        results = {}
        backgrounds = {}
        optimized_states = {}
        analysis_out = {}
        loss_components = {}

        timing_results = {}
        for name, cost_func in self.cost_functions.items():
            start_time = time.perf_counter()
            exp_results, optimized_state, background, analysis = self._run_experiment(
                self.experiment_config, cost_func
            )

            elapsed_time = time.perf_counter() - start_time
            timing_results[name] = elapsed_time

            # Calculate error metrics
            metrics = calculate_metrics(
                analysis,
                trajectory,
                self.obs_system.obs_indices,
                self.experiment_config.y_obs,
                self.experiment_config.H,
            )

            results[name] = metrics
            optimized_states[name] = optimized_state
            backgrounds[name] = background
            analysis_out[name] = analysis
            loss_components[name] = exp_results.cost_components

        return {
            "results": results,
            "optimized_states": optimized_states,
            "backgrounds": backgrounds,
            "analysis": analysis_out,
            "y_obs": observations,
            "trajectory": trajectory,
            "loss_components": loss_components,
            "solver_state_infos": exp_results.solver_state_infos,
            "timing_results": timing_results,
        }

    @staticmethod
    def _run_experiment(
        config: ExperimentConfig, cost_function: Callable
    ) -> Tuple[FourDVarResults, Dict[str, float]]:
        """Run a single 4D-Var experiment with given configuration and cost function."""
        results = FourDVarResults(
            *multi_window_4dvar(
                config.initial_state,
                config.y_obs,
                config.obs_indices_per_window,
                config.H,
                config.B,
                config.R,
                config.window_size,
                config.dt,
                config.obs_frequency,
                config.num_windows,
                config.params,
                cost_function,
                config.ode_system,
            )
        )

        optimized_states, analysis, background = results.get_stacked_results()

        return results, optimized_states, background, analysis
