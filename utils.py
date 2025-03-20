import warnings
from typing import Dict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.flatten_util import ravel_pytree
from omegaconf import DictConfig
from sklearn.manifold import TSNE

from kheperax.final_distance import FinalDistKheperaxTask
from kheperax.target import TargetKheperaxConfig
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.utils.plotting import (
    plot_2d_map_elites_repertoire,
    plot_multidimensional_map_elites_grid,
)
from viz.plot_repertoire import plot_repertoire_embeddings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import wandb  # noqa: E402

from qdax.tasks.environments import create as create_v2  # noqa: E402
from qdax.tasks.environments_v1 import create as create_v1  # noqa: E402


def set_env_params(cfg: DictConfig) -> Dict:
    """Set environment parameters from config."""
    if "env_params" not in cfg.keys():
        return cfg
    if cfg.env.name in cfg.env_params.keys():
        for k, v in cfg.env_params[cfg.env.name].items():
            cfg[k] = v
            print(f"Setting {k} to {v}")
    return cfg


def get_env(cfg):
    """Initialize environment based on configuration."""
    if hasattr(cfg, "qdax_name"):  # qdax
        env_creator = create_v1 if cfg.version == "v1" else create_v2
        return env_creator(
            cfg.qdax_name,
            episode_length=cfg.episode_length,
            exclude_current_positions_from_observation=cfg.exclude_current_positions_from_observation,
            fixed_init_state=True if cfg.version == "v2" else False,
        )
    else:  # kheperax
        kheperax_config = TargetKheperaxConfig.get_map(cfg.map_name)
        kheperax_config.episode_length = cfg.episode_length
        kheperax_config.mlp_policy_hidden_layer_sizes = cfg.policy_hidden_layer_size
        kheperax_config.resolution = cfg.resolution
        kheperax_config.std_noise_wheel_velocities = cfg.std_noise_wheel_velocities
        kheperax_config.action_scale = cfg.action_scale
        kheperax_config.robot = kheperax_config.robot.replace(
            std_noise_sensor_measures=cfg.std_noise_sensor_measures,
            lasers_return_minus_one_if_out_of_range=cfg.lasers_return_minus_one_if_out_of_range,
        )
        return FinalDistKheperaxTask.create_env(kheperax_config)


def init_population(cfg, env, policy_network, random_key):
    """Initialize a population of policies."""
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=cfg.batch_size)
    fake_batch = jnp.zeros(shape=(cfg.batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)
    return init_variables, random_key


def get_repertoire_fig(repertoire, cfg, repertoire_name):
    """Generate a figure for visualizing a repertoire."""
    # plotting for active repertoire
    if repertoire_name.startswith("repertoire"):
        if cfg.me_repertoire:
            if repertoire.descriptors.shape[1] == 2:
                fig, axes = plot_2d_map_elites_repertoire(
                    centroids=repertoire.centroids,
                    repertoire_fitnesses=repertoire.fitnesses,
                    minval=cfg.env.task.min_bd,
                    maxval=cfg.env.task.max_bd,
                    repertoire_descriptors=repertoire.descriptors,
                    vmin=cfg.env.vmin,
                    vmax=cfg.env.vmax,
                )
            else:
                fig, axes = plot_multidimensional_map_elites_grid(
                    repertoire=repertoire,
                    minval=cfg.env.task.min_bd,
                    maxval=cfg.env.task.max_bd,
                    grid_shape=tuple(cfg.env.task.grid_shape),
                    vmin=cfg.env.vmin,
                    vmax=cfg.env.vmax,
                )
        else:
            if repertoire.descriptors.shape[1] == 2:
                # Get the embeddings
                embeddings_2d = repertoire.descriptors
                # Fill NaN values with zeros
                embeddings_2d = jnp.nan_to_num(embeddings_2d)
            else:
                # Get the embeddings
                embeddings = repertoire.descriptors
                # Fill NaN values with zeros
                embeddings = jnp.nan_to_num(embeddings)
                # Perform dimensionality reduction using t-SNE
                model = TSNE(n_components=2, random_state=42)
                embeddings_2d = model.fit_transform(embeddings)

            fig, axes = plot_repertoire_embeddings(
                repertoire_fitnesses=repertoire.fitnesses,
                repertoire_descriptors=repertoire.passive_descriptors,
                embeddings_2d=embeddings_2d,
                minval=cfg.env.min_bd,
                maxval=cfg.env.max_bd,
                vmin=cfg.env.vmin,
                vmax=cfg.env.vmax,
            )

    # plotting for passive repertoire
    elif repertoire_name.startswith("passive_repertoire"):
        # For the moment, we assume all passive grids will have 2D BD space
        if repertoire.descriptors.shape[1] == 2:
            fig, axes = plot_2d_map_elites_repertoire(
                centroids=repertoire.centroids,
                repertoire_fitnesses=repertoire.fitnesses,
                minval=cfg.env.min_bd,
                maxval=cfg.env.max_bd,
                repertoire_descriptors=repertoire.descriptors,
                vmin=cfg.env.vmin,
                vmax=cfg.env.vmax,
            )

    else:
        raise ValueError(
            "repertoire_name {repertoire_name} not recognized. Must be 'repertoire' or 'passive_repertoire'."
        )

    return fig, axes


def log_final_metrics(
    cfg, metrics, total_duration, repertoire, passive_repertoire=None
):
    """Log the final metrics and repertoire visualizations."""
    if cfg.name in ["td3_baseline", "ga_baseline", "map_elites_rand_baseline"]:
        fig, axes = None, None
    else:
        fig, axes = get_repertoire_fig(repertoire, cfg, repertoire_name="repertoire")
    if fig is not None:
        wandb.log(
            {"final/repertoire": wandb.Image(fig)},
        )
        plt.close(fig)

    if passive_repertoire is not None:
        fig, axes = get_repertoire_fig(
            passive_repertoire, cfg, repertoire_name="passive_repertoire"
        )
        if fig is not None:
            wandb.log(
                {"final/passive_repertoire": wandb.Image(fig)},
            )
            plt.close(fig)

    for key, value in metrics.items():
        # take last value
        wandb.log(
            {"final/" + key: value[-1]},
        )
    wandb.log(
        {"final/duration": total_duration},
    )

    return metrics


def log_running_metrics(metrics, logged_metrics, all_metrics, step):
    """Log ongoing metrics to wandb during training."""
    for key, value in metrics.items():
        # take last value
        logged_metrics[key] = value[-1]

        # take all values
        if key in all_metrics.keys():
            all_metrics[key] = jnp.concatenate([all_metrics[key], value])
        else:
            all_metrics[key] = value
    wandb.log(logged_metrics, step=step)
    return logged_metrics, all_metrics


def log_repertoire(repertoire, cfg, step, repertoire_name="passive_repertoire"):
    "Log the given repertoire."
    fig, _ = get_repertoire_fig(repertoire, cfg, repertoire_name=repertoire_name)
    if fig:
        wandb.log({repertoire_name: wandb.Image(fig)}, step=step)
        plt.close(fig)
    return repertoire


def log_repertoire_embeddings(repertoire, cfg, step, log_name="repertoire_embeddings"):
    """
    Visualize the embeddings using dimensionality reduction (t-SNE) and log these to
    wandb.
    """
    # Get the embeddings
    embeddings = repertoire.descriptors
    # Fill NaN values with zeros
    embeddings = jnp.nan_to_num(embeddings)
    # Perform dimensionality reduction using t-SNE
    model = TSNE(n_components=2, random_state=42)
    embeddings_2d = model.fit_transform(embeddings)

    fig, axs = plot_repertoire_embeddings(
        repertoire_fitnesses=repertoire.fitnesses,
        repertoire_descriptors=repertoire.passive_descriptors,
        embeddings_2d=embeddings_2d,
        minval=cfg.env.min_bd,
        maxval=cfg.env.max_bd,
        vmin=cfg.env.vmin,
        vmax=cfg.env.vmax,
    )

    # Log the plot to wandb
    wandb.log({log_name: wandb.Image(fig)}, step=step)
    plt.close(fig)

    return repertoire


def load_repertoire(cfg):
    """Load a stored repertoire."""

    # Init environment
    env = get_env(cfg.env)
    # Init a random key
    random_key = jax.random.PRNGKey(cfg.seed)
    # Init policy network
    policy_layer_sizes = tuple(cfg.policy_hidden_layer_size) + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of policies
    random_key, subkey = jax.random.split(random_key)
    fake_batch = jnp.zeros(shape=(env.observation_size,))
    fake_params = policy_network.init(subkey, fake_batch)

    # Load the repertoire
    _, reconstruction_fn = ravel_pytree(fake_params)
    return MapElitesRepertoire.load(
        reconstruction_fn=reconstruction_fn, path=cfg.repertoire_dir
    )



def get_observation_dims(cfg, observation_size, img_observation_size):
    """Determine observation dimensions based on configuration."""
    obs_cfg = cfg.env.observation_extraction
    obs_dim = jnp.minimum(observation_size, obs_cfg.max_obs_size)
    observations_per_episode = cfg.env.episode_length // obs_cfg.sampling_freq if obs_cfg.subsample else cfg.env.episode_length
    if obs_cfg.observation_option == "sensory_data":
        return jnp.array((observations_per_episode, obs_dim))
    elif obs_cfg.observation_option == "images":
        return img_observation_size
    raise ValueError("Invalid observation_option selected.")
