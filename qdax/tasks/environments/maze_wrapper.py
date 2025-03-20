import warnings
from typing import List, Optional

import brax
import jax
import jax.numpy as jnp
from brax.envs.base import Env, State, Wrapper
from google.protobuf import text_format

from qdax.tasks.environments.base_wrappers import QDEnv  # type: ignore
from qdax.tasks.environments.locomotion_wrappers import (
    COG_NAMES,
    FEET_NAMES,
    FORWARD_REWARD_NAMES,
)


class MazeWrapper(Wrapper):
    """Wraps gym environments to add a maze in the environment
    and a new reward (distance to the goal).

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply add the Maze to the environment,
    along with the new reward.

    This wrapper also adds xy in the observation, as it is an important
    information for an agent. Now that the agent is in a maze, we
    expect its actions to depend on its xy position.

    The xy position is normalised thanks to the decided limits of the env,
    which are [-5, 40] for x and y.

    The only supported envs at the moment are among the classic
    locomotion envs : Ant.

    RMQ: Humanoid is not supported yet.
    RMQ: works for walker2d etc.. but it does not make sens as they
    can only go in one direction.

    Example :

        from brax import envs
        from brax import jumpy as jnp

        # choose in ["ant"]
        ENV_NAME = "ant"
        env = envs.create(env_name=ENV_NAME)
        qd_env = MazeWrapper(env, ENV_NAME)

        state = qd_env.reset(rng=jnp.random_prngkey(seed=0))
        for i in range(10):
            action = jnp.zeros((qd_env.action_size,))
            state = qd_env.step(state, action)


    """

    def __init__(
        self,
        env: Env,
        env_name: str,
    ):

        if env_name not in ["antmaze", "humanoid"]:
            warnings.warn(
                "Make sure your agent can move in two dimensions!",
                stacklevel=2,
            )
        super().__init__(env)

        self.env = env
        self._env_name = env_name

        self._cog_idx = self.env.sys.link_names.index(COG_NAMES[env_name])
        self.target_xy_pos = jnp.array([35.0, 0.0]) #TODO: get this from env

        # we need to normalise x/y position to avoid values to explose
        self._substract = jnp.array([17.5, 17.5])  # come from env limits
        self._divide = jnp.array([22.5, 22.5])  # come from env limits

    @property
    def name(self) -> str:
        return self._env_name

    @property
    def observation_size(self) -> int:
        """The size of the observation vector returned in step and reset."""
        rng = jax.random.PRNGKey(0)
        reset_state = self.reset(rng)
        return int(reset_state.obs.shape[-1])

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        # get xy position of the center of gravity and of the target
        cog_xy_position = state.pipeline_state.x.pos[self._cog_idx][:2]
        # update the reward
        new_reward = -jnp.linalg.norm(self.target_xy_pos - cog_xy_position)
        # add cog xy position to the observation - normalise
        cog_xy_position = (cog_xy_position - self._substract) / self._divide
        return state.replace(reward=new_reward)  # type: ignore

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        # get xy position of the center of gravity and of the target
        cog_xy_position = state.pipeline_state.x.pos[self._cog_idx][:2]
        # update the reward
        new_reward = -jnp.linalg.norm(self.target_xy_pos - cog_xy_position)
        # add cog xy position to the observation - normalise
        cog_xy_position = (cog_xy_position - self._substract) / self._divide
        return state.replace(reward=new_reward)  # type: ignore
