import os

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from brax.io import html
from brax.v1.io import html as html_v1

from kheperax.rendering_tools import RenderingTools
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire


def print_trajectory_info(rollout, bd_extractor):
    """Print information about the trajectory."""
    # fitness = sum(s.reward for s in rollout)
    fitness = rollout[-1].reward
    print(
        f"Fitness: {fitness}\n",
        f"Number of steps: {len(rollout)}\n",
    )


def visualize_individual(env, policy_network, params, path: os.PathLike, bd_extractor):
    """Visualize the behavior of an individual in the environment and save the visualization as an HTML file."""
    # Play some steps in the environment
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_network.apply)

    rollout = []
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)
    reward = 0
    while not state.done:
        rollout.append(state)
        action = jit_inference_fn(params, state.obs)
        state = jit_env_step(state, action)
        reward += state.reward

    print_trajectory_info(rollout, bd_extractor)

    if hasattr(env, "backend"):
        # dt in env = self.sys.dt * self._n_frames, dt in sys = sys.dt
        # This line sets dt in rendering to be dt * n_frames
        # env.sys = env.sys.replace(dt=env.dt)
        html.save(path, env.sys, [s.pipeline_state for s in rollout])
    else:
        html_v1.save_html(path, env.sys, [s.qp for s in rollout])


def kheperax_visualize_individual(
    env, policy_network, params, path: os.PathLike, bd_extractor
):
    """Visualize the behavior of an individual in the environment and save the visualization as a GIF."""
    # Play some steps in the environment
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_network.apply)

    rollout = []
    s_rollout = []
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)
    reward = 0
    base_image = env.create_image(state)
    while not state.done:
        image = env.add_robot(base_image, state)
        image = env.add_lasers(image, state)
        image = env.render_rgb_image(image, flip=True)
        rollout.append(image)
        s_rollout.append(state)

        action = jit_inference_fn(params, state.obs)
        state = jit_env_step(state, action)
        reward += state.reward

    print_trajectory_info(s_rollout, bd_extractor)

    fps = 30
    duration = 1000 / fps
    imageio.mimsave(path, rollout, duration=duration, fps=fps)


def kheperax_visualize_trajectory(
    env,
    policy_network,
    params,
    path: os.PathLike,
    bd_extractor,
):
    """
    Visualize the complete trajectory of an individual in the environment as a single image,
    showing the path taken through the maze.

    Args:
        env: The environment instance
        policy_network: The neural network policy
        params: Network parameters
        path: Path where to save the visualization
        bd_extractor: Behavior descriptor extractor
    """
    # JIT compile environment and policy functions
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_network.apply)

    # Initialize environment
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)
    reward = 0

    # Create base image with initial state
    base_image = env.create_image(state)

    # Store robot positions for trajectory
    positions = []
    s_rollout = []

    # Run simulation and collect positions
    while not state.done:
        # Store current position
        robot_pos = env.get_xy_pos(state.robot)
        positions.append(robot_pos)
        s_rollout.append(state)

        # Get next action and state
        action = jit_inference_fn(params, state.obs)
        state = jit_env_step(state, action)
        reward += state.reward

    # Create final visualization
    trajectory_image = base_image.copy()
    trajectory_image = np.array(trajectory_image, dtype=np.uint8)
    print_trajectory_info(s_rollout, bd_extractor)

    # Draw path using segments
    if len(positions) > 1:
        for i in range(len(positions)):
            pos = positions[i]
            trajectory_image = RenderingTools.place_circle(
                env.kheperax_config,
                trajectory_image,
                center=(pos[0], pos[1]),
                radius=0.005,
                value=5,
            )

    # Add final robot position and lasers
    trajectory_image = env.add_robot(trajectory_image, s_rollout[-1])
    trajectory_image = RenderingTools.place_circle(
        env.kheperax_config,
        trajectory_image,
        center=(positions[0][0], positions[0][1]),
        radius=0.015,
        value=5,
    )

    # Render and save final image
    final_image = env.render_rgb_image(trajectory_image, flip=True)
    imageio.imwrite(path, final_image)


def viz_best_individual(
    env,
    policy_network,
    repertoire: MapElitesRepertoire,
    path: os.PathLike,
    bd_extractor,
):
    """Visualize the behavior of the best individuals in the repertoire."""

    # Get the best individual of the repertoire
    best_idx = jnp.argmax(repertoire.fitnesses)
    best_fitness = jnp.max(repertoire.fitnesses)
    best_bd = repertoire.descriptors[best_idx]
    print(
        f"Best fitness in the repertoire: {best_fitness:.2f}\n",
        f"Behavior descriptor of the best individual in the repertoire: {best_bd}\n",
        f"Index in the repertoire of this individual: {best_idx}\n",
    )
    params = jax.tree_util.tree_map(lambda x: x[best_idx], repertoire.genotypes)

    # Visualize the best individual
    visualize_individual(env, policy_network, params, path, bd_extractor)


def kheperax_viz_best_individual(
    env,
    policy_network,
    repertoire: MapElitesRepertoire,
    path: os.PathLike,
    bd_extractor,
):
    """Visualize the behavior of the best individuals in the repertoire."""

    # Get the best individual of the repertoire
    best_idx = jnp.argmax(repertoire.fitnesses)
    best_fitness = jnp.max(repertoire.fitnesses)
    best_bd = repertoire.descriptors[best_idx]
    print(
        f"Best fitness in the repertoire: {best_fitness:.2f}\n",
        f"Behavior descriptor of the best individual in the repertoire: {best_bd}\n",
        f"Index in the repertoire of this individual: {best_idx}\n",
    )
    params = jax.tree_util.tree_map(lambda x: x[best_idx], repertoire.genotypes)

    # Visualize the best individual
    kheperax_visualize_individual(env, policy_network, params, path, bd_extractor)


def viz_selected_individual(
    env,
    policy_network,
    repertoire: MapElitesRepertoire,
    idx: int,
    path: os.PathLike,
    bd_extractor,
):
    # Get the best individual of the repertoire
    fitness = repertoire.fitnesses[idx]
    bd = repertoire.descriptors[idx]
    print(
        f"Fitness in the repertoire: {fitness:.2f}\n",
        f"Behavior descriptor of the individual in the repertoire: {bd}\n",
    )
    params = jax.tree_util.tree_map(lambda x: x[idx], repertoire.genotypes)

    # Visualize the best individual
    visualize_individual(env, policy_network, params, path, bd_extractor)


def kheperax_viz_selected_individual(
    env,
    policy_network,
    repertoire: MapElitesRepertoire,
    idx: int,
    path: os.PathLike,
    bd_extractor,
):
    """Visualize the behavior of the best individuals in the repertoire."""

    fitness = repertoire.fitnesses[idx]
    bd = repertoire.descriptors[idx]
    print(
        f"Fitness in the repertoire: {fitness:.2f}\n",
        f"Behavior descriptor of the individual in the repertoire: {bd}\n",
    )
    params = jax.tree_util.tree_map(lambda x: x[idx], repertoire.genotypes)

    # Visualize the individual
    kheperax_visualize_individual(env, policy_network, params, path, bd_extractor)


def kheperax_viz_selected_trajectory(
    env,
    policy_network,
    repertoire: MapElitesRepertoire,
    idx: int,
    path: os.PathLike,
    bd_extractor,
):
    """Visualize the behavior of the best individuals in the repertoire."""

    fitness = repertoire.fitnesses[idx]
    bd = repertoire.descriptors[idx]
    print(
        f"Fitness in the repertoire: {fitness:.2f}\n",
        f"Behavior descriptor of the individual in the repertoire: {bd}\n",
    )
    params = jax.tree_util.tree_map(lambda x: x[idx], repertoire.genotypes)

    # Visualize the individual
    kheperax_visualize_trajectory(env, policy_network, params, path, bd_extractor)


if __name__ == "__main__":
    pass
