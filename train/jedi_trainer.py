import functools
import logging
import os
import time
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import wandb
from omegaconf import DictConfig, OmegaConf

from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.custom_map_elites_passive_archive import CustomMAPElitesPassive
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks import environments, environments_v1
from qdax.utils.metrics import passive_qd_metrics
from qdax_es.core.containers.gp_repertoire import GPRepertoire
from qdax_es.core.emitters.jedi_emitter import (
    ConstantScheduler,
    JEDiEmitter,
)
from qdax_es.core.emitters.jedi_pool_emitter import GPJEDiPoolEmitter
from qdax_es.utils.restart import FixedGens
from tasks import behavior_descriptor_extractor, task_behavior_descriptor_extractor
from tasks.scoring import passive_scoring_function, scoring_function
from tasks.step import play_step_fn
from utils import (
    get_env,
    init_population,
    log_final_metrics,
    log_repertoire,
    log_running_metrics,
    set_env_params,
)
from viz.visualization import (
    kheperax_viz_best_individual,
    viz_best_individual,
)


def train(
    cfg: DictConfig,
    map_elites_scan_update,
    repertoire,
    passive_repertoire,
    emitter_state,
    random_key,
):

    num_generations = cfg.num_iterations // cfg.metrics_log_period
    total_evaluations = 0
    all_metrics = {}
    min_obs = None
    max_obs = None

    for i in range(num_generations):
        start_time = time.time()

        (
            repertoire,
            passive_repertoire,
            emitter_state,
            random_key,
        ), metrics = jax.lax.scan(
            map_elites_scan_update,
            (repertoire, passive_repertoire, emitter_state, random_key),
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

        logging.info(
            f"Generation {i + 1}/{num_generations} - Time: {timelapse:.2f} seconds"
        )
        metrics_last_iter = jax.tree_util.tree_map(
            lambda metric: "{0:.2f}".format(float(metric[-1])), metrics
        )
        logging.info(metrics_last_iter)

        # Log metrics
        total_evaluations += cfg.metrics_log_period * cfg.batch_size
        logged_metrics = {
            "time": timelapse,
            "evaluations": total_evaluations,
            "iteration": 1 + i * cfg.metrics_log_period,
            "env_steps": total_evaluations * cfg.env.episode_length,
        }

        logged_metrics, all_metrics = log_running_metrics(
            metrics, logged_metrics, all_metrics, step=total_evaluations
        )

        # Log repertoire
        if (i * cfg.metrics_log_period) % cfg.repertoire_log_period == 0:
            logging.info("Logging repertoire...")
            repertoire = log_repertoire(
                repertoire, cfg, repertoire_name="repertoire", step=total_evaluations
            )
            passive_repertoire = log_repertoire(
                passive_repertoire,
                cfg,
                repertoire_name="passive_repertoire",
                step=total_evaluations,
            )

    return all_metrics, repertoire, passive_repertoire, (min_obs, max_obs)


# usage python -m train.jedi_trainer --config-name jedi.yaml env=brax/ant_maze
@hydra.main(config_path="../configs", config_name="jedi.yaml", version_base=None)
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
        cfg.batch_size = cfg.es_pop * cfg.pool_size
        logging.info(f"Batch size: {cfg.batch_size}")
        logging.info(f"Alpha: {cfg.wtfs_alpha}")
        env = get_env(cfg.env)

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
            behavior_descriptor_extractor[cfg.env.task.bd_extractor]["function"],
            **behavior_descriptor_extractor[cfg.env.task.bd_extractor]["args"],
        )
        passive_bd_extraction_fn = functools.partial(
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
        passive_scoring_fn = passive_scoring_function(
            scoring_fn=scoring_fn,
            passive_behaviour_descriptor_extractor=passive_bd_extraction_fn,
        )

        # Init population of controllers
        init_variables, random_key = init_population(
            cfg, env, policy_network, random_key
        )

        # Get minimum reward value to make sure qd_score are positive
        if cfg.env.name == "kheperax":
            reward_offset = cfg.env.episode_length + jnp.sqrt(2) * 100
        else:
            if cfg.env.version == "v1":
                reward_offset = environments_v1.reward_offset[cfg.env.qdax_name]
            else:
                reward_offset = environments.reward_offset[cfg.env.qdax_name]

        # Define a metrics function
        metrics_function = functools.partial(
            passive_qd_metrics,
            qd_offset=(
                reward_offset * cfg.env.episode_length
                if cfg.env.name != "kheperax"
                else reward_offset
            ),
        )

        # Compute the centroids
        random_key, subkey = jax.random.split(random_key)
        centroids, random_key = compute_cvt_centroids(
            num_descriptors=cfg.env.task.num_descriptors,
            num_init_cvt_samples=cfg.env.task.num_init_cvt_samples,
            num_centroids=cfg.env.task.num_centroids,
            minval=cfg.env.task.min_bd,
            maxval=cfg.env.task.max_bd,
            random_key=subkey,
        )

        random_key, subkey = jax.random.split(random_key)
        passive_centroids, random_key = compute_cvt_centroids(
            num_descriptors=env.behavior_descriptor_length,
            num_init_cvt_samples=cfg.env.num_init_cvt_samples,
            num_centroids=cfg.env.num_centroids,
            minval=cfg.env.min_bd,
            maxval=cfg.env.max_bd,
            random_key=subkey,
        )

        # Define emitter
        restarter = FixedGens(max_gens=cfg.es_gens)

        if cfg.wtfs_alpha == "decay":
            raise NotImplementedError("Decay alpha not implemented yet")
        else:
            # Assert it is int or float
            assert isinstance(
                cfg.wtfs_alpha, (int, float)
            ), f"Alpha should be int or float if constant, got {cfg.wtfs_alpha}"
            alpha_scheduler = ConstantScheduler(cfg.wtfs_alpha)

        emitter = JEDiEmitter(
            centroids=centroids,
            es_hp={"sigma_init": cfg.sigma_g, "popsize": cfg.es_pop},
            es_type=cfg.es_type,
            alpha_scheduler=alpha_scheduler,
            restarter=restarter,
            global_norm=False,
        )
        emitter = GPJEDiPoolEmitter(
            pool_size=cfg.pool_size,
            emitter=emitter,
        )

        # Instantiate MAP-Elites
        logging.info("Algorithm initialization")
        init_time = time.time()
        map_elites = CustomMAPElitesPassive(
            scoring_function=passive_scoring_fn,
            emitter=emitter,
            metrics_function=metrics_function,
            repertoire_type=GPRepertoire,
        )
        # Compute initial repertoire and emitter state
        repertoire, passive_repertoire, emitter_state, random_key = map_elites.init(
            genotypes=init_variables,
            centroids=centroids,
            passive_centroids=passive_centroids,
            random_key=random_key,
            repertoire_kwargs={
                "weighted": cfg.weighted_gp,
                "max_count": 1e3,
            },
        )

        init_repertoire_time = time.time() - init_time
        logging.info(f"Repertoire initialized in {init_repertoire_time:.2f} seconds")

        map_elites_scan_update = map_elites.scan_update

        metrics, repertoire, passive_repertoire, obs_data = train(
            cfg=cfg,
            map_elites_scan_update=map_elites_scan_update,
            repertoire=repertoire,
            passive_repertoire=passive_repertoire,
            emitter_state=emitter_state,
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

        # Save final repertoire
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

        return repertoire, centroids, random_key, metrics


if __name__ == "__main__":
    main()
