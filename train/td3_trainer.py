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

from qdax.baselines.td3 import TD3, TD3Config
from qdax.core.containers.mapelites_repertoire import (
    compute_cvt_centroids,
)
from qdax.core.map_elites import MapElitesRepertoire
from qdax.core.neuroevolution.buffers.buffer import (
    QDTransition,
    ReplayBuffer,
)
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.neuroevolution.sac_td3_utils import do_iteration_fn, warmstart_buffer
from qdax.tasks import environments_v1
from tasks import task_behavior_descriptor_extractor
from utils import (
    log_final_metrics,
    log_running_metrics,
    set_env_params,
)
from viz.visualization import viz_best_individual


# usage python -m train.td3_trainer --config-name td3.yaml env=brax/ant_maze
@hydra.main(config_path="../configs", config_name="td3.yaml", version_base=None)
def main(cfg: DictConfig) -> None:

    cfg = set_env_params(cfg)
    logging.info(cfg)
    logging.info("Training")
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    with wandb.init(config=wandb_cfg, **cfg.wandb):

        # Create local results directory for visualizations and repertoire
        results_dir = os.path.join(os.getcwd(), "results")
        _passive_archive_dir = os.path.join(
            results_dir, "passive_archive", cfg.wandb.name, ""
        )
        _best_individual_dir = Path(results_dir) / "best_individual"
        os.makedirs(_best_individual_dir, exist_ok=True)
        os.makedirs(_passive_archive_dir, exist_ok=True)

        # Init a random key
        random_key = jax.random.PRNGKey(cfg.seed)
        random_key, subkey = jax.random.split(random_key)

        # Init environment
        if cfg.env.name == "ant_maze":
            env = environments_v1.create(
                env_name=cfg.env.qdax_name,
                episode_length=cfg.env.episode_length,
                batch_size=cfg.batch_size,
                use_contact_forces=cfg.env.use_contact_forces,
                exclude_current_positions_from_observation=cfg.env.exclude_current_positions_from_observation,
            )
            eval_env = environments_v1.create(
                env_name=cfg.env.qdax_name,
                episode_length=cfg.env.episode_length,
                batch_size=cfg.qd_batch_size,
                use_contact_forces=cfg.env.use_contact_forces,
                exclude_current_positions_from_observation=cfg.env.exclude_current_positions_from_observation,
            )
        else:
            env = environments_v1.create(
                env_name=cfg.env.qdax_name,
                episode_length=cfg.env.episode_length,
                batch_size=cfg.batch_size,
                exclude_current_positions_from_observation=cfg.env.exclude_current_positions_from_observation,
            )
            eval_env = environments_v1.create(
                env_name=cfg.env.qdax_name,
                episode_length=cfg.env.episode_length,
                batch_size=cfg.qd_batch_size,
                exclude_current_positions_from_observation=cfg.env.exclude_current_positions_from_observation,
            )

        # Initialize buffer
        dummy_transition = QDTransition.init_dummy(
            observation_dim=env.observation_size,
            action_dim=env.action_size,
            descriptor_dim=env.state_descriptor_length,
        )
        replay_buffer = ReplayBuffer.init(
            buffer_size=cfg.buffer_size, transition=dummy_transition
        )

        td3_config = TD3Config(
            batch_size=cfg.transitions_batch_size,
            episode_length=cfg.env.episode_length,
            soft_tau_update=cfg.soft_tau_update,
            expl_noise=cfg.expl_noise,
            policy_delay=cfg.policy_delay,
            discount=cfg.discount,
            noise_clip=cfg.noise_clip,
            policy_noise=cfg.policy_noise,
            reward_scaling=cfg.reward_scaling,
            critic_hidden_layer_size=cfg.critic_hidden_layer_size,
            policy_hidden_layer_size=cfg.env.policy_hidden_layer_size,
            critic_learning_rate=cfg.critic_learning_rate,
            policy_learning_rate=cfg.policy_learning_rate,
        )

        # Init TD3
        init_time = time.time()
        td3 = TD3(config=td3_config, action_size=env.action_size)
        random_key, subkey = jax.random.split(random_key)
        training_state = td3.init(
            random_key=subkey,
            action_size=env.action_size,
            observation_size=env.observation_size,
        )

        # Create the initial environment states
        random_key, subkey = jax.random.split(random_key)
        keys = jnp.repeat(
            jnp.expand_dims(subkey, axis=0), repeats=cfg.batch_size, axis=0
        )
        env_reset_fn = jax.jit(env.reset)
        init_state = env_reset_fn(keys)

        random_key, subkey = jax.random.split(random_key)
        keys = jnp.repeat(
            jnp.expand_dims(subkey, axis=0), repeats=cfg.qd_batch_size, axis=0
        )
        eval_env_reset_fn = jax.jit(eval_env.reset)
        eval_init_state = eval_env_reset_fn(keys)

        # Make play_step functions scannable by passing static args beforehand
        play_eval_step = functools.partial(
            td3.play_qd_step_fn,
            env=eval_env,
            deterministic=True,
        )
        play_step = functools.partial(
            td3.play_qd_step_fn,
            env=env,
            deterministic=False,
        )

        # Prepare eval function
        bd_extraction_fn = functools.partial(
            task_behavior_descriptor_extractor[cfg.env.name]["function"],
            **task_behavior_descriptor_extractor[cfg.env.name]["args"],
        )

        reward_offset = environments_v1.reward_offset[cfg.env.qdax_name]

        eval_qd_policy = functools.partial(
            td3.eval_qd_policy_fn,
            eval_env_first_state=eval_init_state,
            play_step_fn=play_eval_step,
            bd_extraction_fn=bd_extraction_fn,
            reward_type=cfg.env.reward_type,
        )

        # Define metrics function for the passive repertoire
        def metrics_fn(repertoire: MapElitesRepertoire):
            grid_empty = repertoire.fitnesses == -jnp.inf
            qd_score = jnp.sum(repertoire.fitnesses, where=~grid_empty)
            qd_score += (
                reward_offset * cfg.env.episode_length * jnp.sum(1.0 - grid_empty)
            )
            coverage = 100 * jnp.mean(1.0 - grid_empty)
            max_fitness = jnp.max(repertoire.fitnesses)

            return {
                "passive_qd_score": jnp.array([qd_score]),
                "passive_max_fitness": jnp.array([max_fitness]),
                "passive_coverage": jnp.array([coverage]),
            }

        # Warmstart the buffer
        random_key, subkey = jax.random.split(random_key)
        replay_buffer, env_state, training_state = warmstart_buffer(
            replay_buffer=replay_buffer,
            training_state=training_state,
            env_state=init_state,
            num_warmstart_steps=cfg.warmup_steps,
            env_batch_size=cfg.batch_size,
            play_step_fn=play_step,
        )

        # Evaluate untrained policy
        fitnesses, descriptors, mean_fitness, mean_bd = eval_qd_policy(
            training_state=training_state
        )

        # Init the passive repertoire
        random_key, subkey = jax.random.split(random_key)
        centroids, random_key = compute_cvt_centroids(
            num_descriptors=cfg.env.task.num_descriptors,
            num_init_cvt_samples=cfg.env.num_init_cvt_samples,
            num_centroids=cfg.env.num_centroids,
            minval=cfg.env.task.min_bd,
            maxval=cfg.env.task.max_bd,
            random_key=subkey,
        )
        genotypes = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), training_state.policy_params
        )
        fitnesses = mean_fitness
        descriptors = mean_bd
        repertoire, _ = MapElitesRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
        )

        do_iteration = functools.partial(
            do_iteration_fn,
            env_batch_size=cfg.batch_size,
            grad_updates_per_step=cfg.grad_updates_per_step,
            play_step_fn=play_step,
            update_fn=td3.update,
        )

        def scan_do_iteration(
            carry,
            unused,
        ):
            training_state, env_state, replay_buffer = carry
            training_state, env_state, replay_buffer, metrics = do_iteration(
                training_state, env_state, replay_buffer
            )
            return (training_state, env_state, replay_buffer), metrics

        total_num_iterations = cfg.num_total_steps // (
            cfg.batch_size * cfg.env.episode_length
        )
        logging.info("Total number of iterations: {}".format(total_num_iterations))

        all_metrics = {}
        start_time = time.time()

        # Main training loop
        for iteration in range(total_num_iterations):

            (training_state, env_state, replay_buffer), metrics = jax.lax.scan(
                scan_do_iteration,
                (training_state, env_state, replay_buffer),
                None,
                length=cfg.env.episode_length,
            )

            if iteration % cfg.metrics_log_period == 0:

                # Evaluate policy
                random_key, subkey = jax.random.split(random_key)
                fitnesses, descriptors, mean_fitness, mean_bd = eval_qd_policy(
                    training_state=training_state
                )

                # Add to repertoire
                genotypes = jax.tree_util.tree_map(
                    lambda x: jnp.expand_dims(x, axis=0), training_state.policy_params
                )
                fitnesses = mean_fitness
                descriptors = mean_bd
                repertoire, _ = repertoire.add(
                    genotypes,
                    descriptors,
                    fitnesses,
                )

                total_evaluations = cfg.batch_size * (iteration + 1)
                total_steps = total_evaluations * cfg.env.episode_length
                timelapse = time.time() - start_time
                logged_metrics = {
                    "time": timelapse,
                    "evaluations": total_evaluations,
                    "iteration": 1 + iteration,
                    "env_steps": total_steps,
                }
                # Log mean values of actor / critic metrics
                logged_metrics = jax.tree_util.tree_map(
                    lambda metric: jnp.mean(metric), logged_metrics
                )
                metrics = metrics_fn(repertoire)
                logged_metrics, all_metrics = log_running_metrics(
                    metrics,
                    logged_metrics,
                    all_metrics,
                    step=total_evaluations,
                )

                logging.info(
                    f"Iteration: {iteration}/{total_num_iterations}, Num evals: {total_evaluations}, Time: {timelapse:.2f} seconds"
                )
                metrics_last_eval = jax.tree_util.tree_map(
                    lambda metric: "{0:.2f}".format(float(metric[-1])), metrics
                )
                logging.info(metrics_last_eval)

                start_time = time.time()

        # Run final evaluation & log final metrics
        fitnesses, descriptors, mean_fitness, mean_bd = eval_qd_policy(
            training_state=training_state
        )
        # Final add to repertoire
        genotypes = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), training_state.policy_params
        )
        repertoire, _ = repertoire.add(
            genotypes,
            descriptors,
            fitnesses,
        )
        metrics = metrics_fn(repertoire)
        total_duration = time.time() - init_time

        log_final_metrics(
            cfg=cfg,
            metrics=metrics,
            total_duration=total_duration,
            repertoire=None,
            passive_repertoire=repertoire,
        )

        # Save visualisation of best individual
        viz_filename = Path(_best_individual_dir) / (cfg.wandb.name + ".gif")

        # Init policy network
        policy_layer_sizes = tuple(cfg.env.policy_hidden_layer_size) + (
            env.action_size,
        )
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )

        viz_best_individual(
            eval_env,
            policy_network,
            repertoire,
            path=viz_filename,
            bd_extractor=bd_extraction_fn,
        )

        # Save final repertoire
        repertoire.save(_passive_archive_dir)


if __name__ == "__main__":
    main()
