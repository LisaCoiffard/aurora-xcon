import functools
import logging
import os
import time
from pathlib import Path
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import wandb
from omegaconf import DictConfig, OmegaConf

from ae_utils.model_train import init_autoencoder_model_training
from qdax.core.aurora_adaptive import AURORAAdaptivePassive
from qdax.core.aurora_threshold import AURORAThresholdPassive
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks import environments, environments_v1
from qdax.tasks.brax_envs import get_aurora_scoring_fn
from qdax.utils.metrics import passive_qd_metrics
from tasks import task_behavior_descriptor_extractor
from tasks.scoring import scoring_function
from tasks.step import play_step_fn
from utils import (
    get_env,
    get_observation_dims,
    init_population,
    log_final_metrics,
    log_running_metrics,
    set_env_params,
)
from viz.visualization import kheperax_viz_best_individual, viz_best_individual


def train(
    cfg: DictConfig,
    aurora_scan_update,
    train_state,
    aurora_extra_info,
    repertoire,
    passive_repertoire,
    aurora,
    random_key,
):

    model_params = aurora_extra_info.model_params
    previous_error = jnp.sum(repertoire.fitnesses != -jnp.inf) - cfg.target_size
    num_generations = cfg.num_iterations // cfg.metrics_log_period
    all_metrics = {}
    min_obs = None
    max_obs = None
    model_metrics = {}
    total_evaluations = 0
    ae_timelapse = 0
    csc_timelapse = 0

    # Design AURORA's schedule
    default_update_base = cfg.default_update_base
    update_base = int(jnp.ceil(default_update_base / cfg.metrics_log_period))
    schedules = jnp.cumsum(jnp.arange(update_base, num_generations, update_base))

    for i in range(num_generations):

        start_time = time.time()

        (
            (repertoire, passive_repertoire, random_key, aurora_extra_info),
            metrics,
        ) = jax.lax.scan(
            aurora_scan_update,
            (repertoire, passive_repertoire, random_key, aurora_extra_info),
            (),
            length=cfg.metrics_log_period,
        )

        timelapse = time.time() - start_time

        if min_obs is not None and max_obs is not None:
            min_obs = jnp.minimum(min_obs, metrics["min_obs"])
            max_obs = jnp.maximum(max_obs, metrics["max_obs"])
            del metrics["min_obs"]
            del metrics["max_obs"]
        else:
            min_obs = jnp.min(metrics["min_obs"], axis=0)
            max_obs = jnp.max(metrics["max_obs"], axis=0)
            del metrics["min_obs"]
            del metrics["max_obs"]

        total_evaluations += cfg.metrics_log_period * cfg.batch_size

        # AE
        if (i + 1) in schedules and not cfg.no_training:
            logging.info("Updating AE...")
            start_time = time.time()
            # train the autoencoder
            random_key, subkey = jax.random.split(random_key)
            if cfg.reinit_params:
                train_state = train_state.replace(params=model_params)
            repertoire, train_state, aurora_extra_info, model_metrics = aurora.train(
                repertoire, train_state, subkey
            )
            ae_timelapse = time.time() - start_time
            metrics_last_iter = jax.tree_util.tree_map(
                lambda metric: "{0:.4f}".format(float(metric[-1])), model_metrics
            )
            logging.info(metrics_last_iter)

        # CSC
        elif i % 2 == 0 and not cfg.no_csc:
            if cfg.repertoire == "adaptive":
                pass
            else:
                start_time = time.time()
                repertoire, previous_error = aurora.container_size_control(
                    repertoire,
                    target_size=cfg.target_size,
                    prop_gain=cfg.prop_gain,
                    previous_error=previous_error,
                    csc_orig=cfg.csc_orig,
                )
                csc_timelapse = time.time() - start_time

        # Log metrics
        logged_metrics = {
            "time": timelapse + ae_timelapse + csc_timelapse,
            "time_qd": timelapse,
            "time_ae": ae_timelapse,
            "time_csc": csc_timelapse,
            "evaluations": total_evaluations,
            "iteration": 1 + i * cfg.metrics_log_period,
            "env_steps": total_evaluations * cfg.env.episode_length,
        }
        if cfg.repertoire == "adaptive":
            logged_metrics["d_min"] = repertoire.d_min
        else:
            logged_metrics["l_value"] = repertoire.l_value

        logging.info(
            f"Generation {i + 1}/{num_generations} - Time: {timelapse:.2f} seconds"
        )
        metrics_last_iter = jax.tree_util.tree_map(
            lambda metric: "{0:.2f}".format(float(metric[-1])), metrics
        )
        logging.info(metrics_last_iter)
        if (i + 1) in schedules and not cfg.no_training:
            # Log all AE training metrics
            logged_metrics = {
                **logged_metrics,
                **{key: jnp.mean(value) for key, value in model_metrics.items()},
            }

        logged_metrics, all_metrics = log_running_metrics(
            metrics, logged_metrics, all_metrics, step=total_evaluations
        )

        # Every n generations, remove all but n% of individuals
        if cfg.extinction:
            if i == num_generations - 1:
                # Don't do an extinction event on the last generation
                pass
            if (i + 1) % cfg.extinction_freq == 0:
                logging.info("Extinction event...")
                random_key, subkey = jax.random.split(random_key)
                repertoire = repertoire.extinction(
                    remaining_prop=cfg.remaining_prop, random_key=subkey
                )

    return all_metrics, repertoire, passive_repertoire, (min_obs, max_obs)


# usage python -m train.aurora_trainer --config-name aurora.yaml env=brax/ant_maze
@hydra.main(config_path="../configs", config_name="aurora.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """Start training run.

    Args:
        cfg (DictConfig): master config for the hydra_launcher
    """

    cfg = set_env_params(cfg)
    logging.info(cfg)
    logging.info("Training")
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    with wandb.init(config=wandb_cfg, **cfg.wandb):

        # Create local results directory for visualizations and repertoire
        results_dir = os.path.join(os.getcwd(), "results")
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
            cfg,
            env.observation_size,
            (env.aurora_observation_size if cfg.env.name == "kheperax" else None),
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

        step_fn = functools.partial(
            play_step_fn, policy_network=policy_network, env=env
        )

        # Prepare the scoring function
        bd_extraction_fn = functools.partial(
            task_behavior_descriptor_extractor[cfg.env.name]["function"],
            **task_behavior_descriptor_extractor[cfg.env.name]["args"],
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
        if cfg.env.name == "kheperax":
            reward_offset = jnp.sqrt(2) * 100
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
                if cfg.env.name != "kheperax"
                else reward_offset
            ),
            target_size=cfg.target_size if cfg.repertoire == "threshold" else None,
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
        encoder_fn, train_fn, train_state, aurora_extra_info = (
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
        random_key, subkey = jax.random.split(random_key)
        centroids, random_key = compute_cvt_centroids(
            num_descriptors=env.state_descriptor_length,
            num_init_cvt_samples=cfg.env.num_init_cvt_samples,
            num_centroids=cfg.env.num_centroids,
            minval=cfg.env.min_bd,
            maxval=cfg.env.max_bd,
            random_key=subkey,
        )

        # Instantiate AURORA
        if cfg.repertoire == "adaptive":
            aurora = AURORAAdaptivePassive(
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
                cfg.max_size,
                random_key,
            )
        elif cfg.repertoire == "threshold":
            aurora = AURORAThresholdPassive(
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

        init_repertoire_time = time.time() - init_time
        logging.info(f"Repertoire initialized in {init_repertoire_time:.2f} seconds")

        # Init means, stds and AURORA
        random_key, subkey = jax.random.split(random_key)
        repertoire, train_state, aurora_extra_info, _ = aurora.train(
            repertoire, train_state, random_key=subkey
        )

        metrics, repertoire, passive_repertoire, obs_data = train(
            cfg=cfg,
            aurora_scan_update=update_scan_fn,
            train_state=train_state,
            aurora_extra_info=aurora_extra_info,
            repertoire=repertoire,
            passive_repertoire=passive_repertoire,
            aurora=aurora,
            random_key=random_key,
        )

        total_duration = time.time() - init_time

        # Log metrics for the final repertoire
        log_final_metrics(
            cfg=cfg,
            metrics=metrics,
            total_duration=total_duration,
            repertoire=repertoire,
            passive_repertoire=passive_repertoire,
        )

        # Save visualisation of best individual
        viz_filename = (
            Path(_best_individual_dir) / (cfg.wandb.name + ".html")
            if cfg.env.name != "kheperax"
            else Path(_best_individual_dir) / (cfg.wandb.name + ".gif")
        )

        if cfg.env.name == "kheperax":
            kheperax_viz_best_individual(
                env,
                policy_network,
                repertoire,
                path=viz_filename,
                bd_extractor=bd_extraction_fn,
            )
        else:
            viz_best_individual(
                env,
                policy_network,
                repertoire,
                path=viz_filename,
                bd_extractor=bd_extraction_fn,
            )

        # Save final repertoire & AE
        repertoire.save(_repertoire_dir)
        passive_repertoire.save(_passive_archive_dir)
        # Log min/max obs data
        jnp.save(
            os.path.join(_repertoire_dir, "min_obs.npy"),
            obs_data[0],
        )
        jnp.save(
            os.path.join(_repertoire_dir, "max_obs.npy"),
            obs_data[1],
        )


if __name__ == "__main__":
    main()
