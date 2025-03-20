import functools

import jax
import jax.numpy as jnp

from qdax import environments
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.environments import bd_extractors
from tasks.scoring import (
    behavior_descriptor_extractor,
    scoring_function,
    task_behavior_descriptor_extractor,
)
from tasks.step import play_step_fn_passive_desc
from viz.visualization import visualize_individual

if __name__ == "__main__":
    env = environments.create(
        "ant_omni",
        episode_length=250,
        healthy_z_range=(0.3, 4.0),
        fixed_init_state=True,
        backend="spring",
    )
    random_key = jax.random.PRNGKey(1234)
    state = env.reset(random_key)
    print(state)
    state = env.step(state, jnp.zeros(env.action_size))
    print(state)
