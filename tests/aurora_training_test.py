import functools
import logging
import os
import time
from pathlib import Path
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf

from ae_utils.model_train import init_autoencoder_model_training
from main.ae_trainer import train
from qdax.core.aurora import AURORA
from qdax.core.aurora_passive_archive import AURORAPassive
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks import environments, environments_v1
from qdax.tasks.brax_envs import get_aurora_scoring_fn
from qdax.utils.metrics import passive_qd_metrics
from tasks import task_behavior_descriptor_extractor
from tasks.scoring import scoring_function
from tasks.step import final_dist_step_fn, play_step_fn
from utils import (
    get_env,
    get_observation_dims,
    init_population,
    log_final_metrics,
    log_repertoire,
    log_running_metrics,
)
from viz.visualization import kheperax_viz_best_individual, viz_best_individual


def recon_loop(dataset, model, params, obs_mean, obs_std, path):
    for sample in dataset:

        sample_plot = sample * obs_std + obs_mean
        plt.imshow(sample_plot)
        plt.show()
        plt.savefig(path + "sample.png")

        sample = jnp.expand_dims(sample, axis=0)
        latent = model.apply(
            {"params": params},
            sample,
            method=model.encode,
        )
        recon = model.apply(
            {"params": params},
            latent,
            method=model.decode,
        )
        recon = np.array((recon[0] * obs_std + obs_mean)).astype(jnp.uint8)
        plt.imshow(recon)
        plt.show()
        plt.savefig("view/recon.png")

        print("Latent: ", latent)


def eval_model_valset(obs_mean, obs_std, model, model_params, random_key):
    random_key, subkey = jax.random.split(random_key)
    path_dir = "dataset/observation.npy"
    dataset = jnp.load(path_dir)
    # remove nan observations
    valid_obs = jnp.all(jnp.isnan(dataset), axis=(1, 2, 3))
    dataset = dataset[~valid_obs, ...]
    normalized_dataset = (dataset - obs_mean) / obs_std
    normalized_dataset = jax.random.choice(
        subkey, normalized_dataset, (10,), replace=False
    )
    obs_std = jnp.where(obs_std == jnp.inf, 0, obs_std)
    recon_loop(normalized_dataset, model, model_params, obs_mean, obs_std, "view/")


def eval_model_trainset(repertoire, obs_mean, obs_std, model, model_params, random_key):
    random_key, subkey = jax.random.split(random_key)
    dataset = repertoire.observations
    # remove nan observations
    valid_obs = jnp.all(jnp.isnan(dataset), axis=(1, 2, 3))
    dataset = dataset[~valid_obs, ...]
    normalized_dataset = (dataset - obs_mean) / obs_std
    if dataset.shape[0] > 10:
        normalized_dataset = jax.random.choice(
            subkey, normalized_dataset, (10,), replace=False
        )
    obs_std = jnp.where(obs_std == jnp.inf, 0, obs_std)
    recon_loop(normalized_dataset, model, model_params, obs_mean, obs_std, "view/t_")


# usage python -m main.trainer --config-name map_elites.yaml env=ant_omni
@hydra.main(config_path="../configs", config_name="aurora.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """Start training run.

    Args:
        cfg (DictConfig): master config for the hydra_launcher
    """

    logging.info(cfg)
    logging.info("Training")
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    with wandb.init(config=wandb_cfg, **cfg.wandb):

        # Create local results directory for visualizations and repertoire
        results_dir = os.path.join(hydra.utils.get_original_cwd(), "results")
        _repertoire_dir = os.path.join(results_dir, "repertoire", cfg.wandb.name, "")
        _passive_archive_dir = os.path.join(
            results_dir, "passive_archive", cfg.wandb.name, ""
        )
        _best_individual_dir = Path(results_dir) / "best_individual"
        os.makedirs(_repertoire_dir, exist_ok=True)
        os.makedirs(_best_individual_dir, exist_ok=True)
        os.makedirs(_passive_archive_dir, exist_ok=True)

        # Init a random key
        random_key = jax.random.PRNGKey(cfg.seed)
        random_key, subkey = jax.random.split(random_key)

        # Init environment
        env = get_env(cfg.env)
        observations_dims = get_observation_dims(
            cfg, env.observation_size, env.aurora_observation_size
        )

        # Init policy network
        policy_layer_sizes = tuple(cfg.env.policy_hidden_layer_size) + (
            env.action_size,
        )
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )

        # Create the initial environment states
        random_key, subkey = jax.random.split(random_key)
        keys = jnp.repeat(
            jnp.expand_dims(subkey, axis=0), repeats=cfg.batch_size, axis=0
        )
        reset_fn = jax.jit(jax.vmap(env.reset))
        init_states = reset_fn(keys)

        if cfg.env.name == "kheperax":
            step_fn = functools.partial(
                final_dist_step_fn, policy_network=policy_network, env=env
            )
        else:
            step_fn = functools.partial(
                play_step_fn, policy_network=policy_network, env=env
            )

        # Prepare the scoring function
        bd_extraction_fn = functools.partial(
            task_behavior_descriptor_extractor[cfg.env.qdax_name]["function"],
            **task_behavior_descriptor_extractor[cfg.env.qdax_name]["args"],
        )
        scoring_fn = functools.partial(
            scoring_function,
            cfg=cfg,
            init_states=init_states,
            play_step_fn=step_fn,
            behavior_descriptor_extractor=bd_extraction_fn,
        )

        def observation_extractor_fn(
            data,
        ):
            """Extract observation from the state."""
            data, final_state = data
            if cfg.env.observation_extraction.observation_option == "images":
                observations = final_state.info["image_obs"]

            elif cfg.env.observation_extraction.observation_option == "sensory_data":
                # state_obs: (batch_size, traj_length//sampling_freq, obs_size (max_obs_size))
                observations = data.obs[
                    :,
                    :: cfg.env.observation_extraction.sampling_freq,
                    : cfg.env.observation_extraction.max_obs_size,
                ]
            else:
                raise ValueError("Unknown observation option.")

            return observations

        aurora_scoring_fn = get_aurora_scoring_fn(
            scoring_fn=scoring_fn,
            observation_extractor_fn=observation_extractor_fn,
        )

        # Init population of controllers
        init_variables, random_key = init_population(
            cfg, env, policy_network, random_key
        )
        params_count = int(
            sum(x.size for x in jax.tree_util.tree_leaves(init_variables))
            / cfg.batch_size
        )
        logging.info(f"Policy params count (search space): {params_count}")

        # Get minimum reward value to make sure qd_score are positive
        if cfg.env.name == "kheperax_omni":
            reward_offset = 0.5
        elif cfg.env.name == "kheperax_target":
            reward_offset = cfg.env.episode_length * jnp.sqrt(2)
        elif cfg.env.name == "kheperax":
            reward_offset = cfg.env.episode_length + jnp.sqrt(2) * 100
        elif cfg.env.name == "aurora_maze":
            reward_offset = 0.0
        else:
            if cfg.env.version == "v1":
                reward_offset = environments_v1.reward_offset[cfg.env.qdax_name]
            else:
                reward_offset = environments.reward_offset[cfg.env.qdax_name]

        # Define a metrics function
        metrics_fn = functools.partial(
            passive_qd_metrics,
            qd_offset=(
                reward_offset * cfg.env.episode_length
                if cfg.env.qdax_name != "kheperax"
                else reward_offset
            ),
        )

        # Define emitter
        variation_fn = functools.partial(
            isoline_variation, iso_sigma=cfg.iso_sigma, line_sigma=cfg.line_sigma
        )
        mixing_emitter = MixingEmitter(
            mutation_fn=lambda x, y: (x, y),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=cfg.batch_size,
        )

        # Auto-encoder init
        encoder_fn, train_fn, train_state, aurora_extra_info, model = (
            init_autoencoder_model_training(cfg, observations_dims, random_key)
        )
        params_count = sum(
            x.size for x in jax.tree_util.tree_leaves(train_state.params)
        )
        logging.info(f"AE params count: {params_count}")

        @jax.jit
        def update_scan_fn(carry: Any, unused: Any) -> Any:
            """Scan the udpate function."""
            (repertoire, passive_repertoire, random_key, aurora_extra_info) = carry

            # update
            (
                repertoire,
                passive_repertoire,
                _,
                metrics,
                random_key,
            ) = aurora.update(
                repertoire,
                passive_repertoire,
                None,
                random_key,
                aurora_extra_info=aurora_extra_info,
            )

            return (
                (repertoire, passive_repertoire, random_key, aurora_extra_info),
                metrics,
            )

        # Init algorithm
        logging.info("Algorithm initialization")
        init_time = time.time()
        centroids = compute_euclidean_centroids(
            grid_shape=cfg.task.grid_shape,
            minval=cfg.task.min_bd,
            maxval=cfg.task.max_bd,
        )

        # Instantiate AURORA
        aurora = AURORAPassive(
            scoring_function=aurora_scoring_fn,
            emitter=mixing_emitter,
            metrics_function=metrics_fn,
            encoder_function=encoder_fn,
            training_function=train_fn,
        )

        # init step of the aurora algorithm
        (
            repertoire,
            passive_repertoire,
            emitter_state,
            aurora_extra_info,
            random_key,
        ) = aurora.init(
            init_variables,
            centroids,
            aurora_extra_info,
            jnp.asarray(cfg.l_value_init),
            cfg.max_size,
            random_key,
        )


        ##### AURORA INITIALIZATION #####
        # # Randomly choose 10 observations to plot
        # random_key, subkey = jax.random.split(random_key)
        # init_observations = jax.random.choice(
        #     subkey, observations, (10,), replace=False
        # )
        # for i, obs in enumerate(init_observations):
        #     fig = plt.imshow(obs)
        #     plt.savefig(f"view/init_obs_{i}.png")
        #     plt.close()

        # # Plot a scatter plot of the descriptors showing the l-value radius around them
        # fig = plt.scatter(descriptors[:, 0], descriptors[:, 1])

        # # Draw circles around each point
        # for i in range(len(descriptors)):
        #     circle = plt.Circle(
        #         (descriptors[i, 0], descriptors[i, 1]),
        #         cfg.l_value_init,
        #         color="red",
        #         fill=False,
        #     )
        #     plt.gca().add_artist(circle)

        # # Set the aspect of the plot to be equal
        # plt.gca().set_aspect("equal", adjustable="box")
        # plt.savefig("view/init_descriptors.png")
        # plt.close()

        # valid_descriptors = jnp.where(repertoire.fitnesses != -jnp.inf)
        # repertoire_descriptors = repertoire.descriptors[valid_descriptors]

        # fig = plt.scatter(repertoire_descriptors[:, 0], repertoire_descriptors[:, 1])

        # # Draw circles around each point
        # for i in range(len(repertoire_descriptors)):
        #     circle = plt.Circle(
        #         (repertoire_descriptors[i, 0], repertoire_descriptors[i, 1]),
        #         cfg.l_value_init,
        #         color="red",
        #         fill=False,
        #     )
        #     plt.gca().add_artist(circle)

        # # Set the aspect of the plot to be equal
        # plt.gca().set_aspect("equal", adjustable="box")
        # plt.savefig("view/init_rep_descriptors.png")
        # plt.close()
        #### END AURORA INITIALIZATION TEST #####

        init_repertoire_time = time.time() - init_time
        logging.info(f"Repertoire initialized in {init_repertoire_time:.2f} seconds")

        # Init means, stds and AURORA
        random_key, subkey = jax.random.split(random_key)
        repertoire, train_state, aurora_extra_info, _ = aurora.train(
            repertoire, train_state, iteration=0, random_key=subkey
        )

        # CHECK THAT CORRECT PARAMETERS ARE LOGGED
        check_params = jax.tree_util.tree_map(
            jnp.array_equal, train_state.params, aurora_extra_info.model_params
        )
        assert jax.tree_util.tree_all(check_params)

        model_params = aurora_extra_info.model_params
        obs_mean = aurora_extra_info.mean_observations
        obs_std = aurora_extra_info.std_observations

        #### EVALUATE AE ON TRAINING DATASET #####
        eval_model_trainset(
            repertoire, obs_mean, obs_std, model, model_params, random_key
        )
        #### EVALUATE AE ON A TEST DATASET #####
        eval_model_valset(obs_mean, obs_std, model, model_params, random_key)


if __name__ == "__main__":
    main()
