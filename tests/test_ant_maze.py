import jax
import jax.numpy as jnp
from brax.io import html
from brax.v1.io import html as html_v1

from qdax.tasks import environments, environments_v1
from qdax.tasks.environments_v1.base_wrappers import StateDescriptorResetWrapper


def test_ant_maze_rendering():
    env = environments.create(
        "ant_maze",
        episode_length=1000,
        healthy_z_range=(0.3, 5.0),
        exclude_current_positions_from_observation=False,
        fixed_init_state=True,
        backend="mjx",
    )
    # env = environments_v1.create("ant_maze", episode_length=1000)
    rng = jax.random.PRNGKey(0)
    jit_env_reset = jax.jit(env.reset)
    state = jit_env_reset(rng=rng)
    env.sys = env.sys.replace(dt=env.dt)
    rollout = [state]

    # html_v1.save_html("view/ant_maze.html", env.sys, [s.qp for s in rollout])
    html.save("view/ant_maze.html", env.sys, [s.pipeline_state for s in rollout])
    print("Ant maze rendering saved to view/ant_maze.html")


if __name__ == "__main__":
    test_ant_maze_rendering()
