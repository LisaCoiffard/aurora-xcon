from typing import Dict, Optional

from brax.envs.base import Env, State, Wrapper
import flax.struct
import jax
import jax.numpy as jnp


class CompletedEvalMetrics(flax.struct.PyTreeNode):
    current_episode_metrics: Dict[str, jnp.ndarray]
    completed_episodes_metrics: Dict[str, jnp.ndarray]
    completed_episodes: jnp.ndarray
    completed_episodes_steps: jnp.ndarray


class CompletedEvalWrapper(Wrapper):
    """Brax env with eval metrics for completed episodes."""

    STATE_INFO_KEY = "completed_eval_metrics"

    def reset(self, rng: jnp.ndarray) -> State:
        reset_state = self.env.reset(rng)
        reset_state.metrics["reward"] = reset_state.reward
        eval_metrics = CompletedEvalMetrics(
            current_episode_metrics=jax.tree_util.tree_map(
                jnp.zeros_like, reset_state.metrics
            ),
            completed_episodes_metrics=jax.tree_util.tree_map(
                lambda x: jnp.zeros_like(jnp.sum(x)), reset_state.metrics
            ),
            completed_episodes=jnp.zeros(()),
            completed_episodes_steps=jnp.zeros(()),
        )
        reset_state.info[self.STATE_INFO_KEY] = eval_metrics
        return reset_state

    def step(
        self, state: State, action: jnp.ndarray
    ) -> State:
        state_metrics = state.info[self.STATE_INFO_KEY]
        if not isinstance(state_metrics, CompletedEvalMetrics):
            raise ValueError(f"Incorrect type for state_metrics: {type(state_metrics)}")
        del state.info[self.STATE_INFO_KEY]
        nstate = self.env.step(state, action)
        nstate.metrics["reward"] = nstate.reward
        # steps stores the highest step reached when done = True, and then
        # the next steps becomes action_repeat
        completed_episodes_steps = state_metrics.completed_episodes_steps + jnp.sum(
            nstate.info["steps"] * nstate.done
        )
        current_episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b, state_metrics.current_episode_metrics, nstate.metrics
        )
        completed_episodes = state_metrics.completed_episodes + jnp.sum(nstate.done)
        completed_episodes_metrics = jax.tree_util.tree_map(
            lambda a, b: a + jnp.sum(b * nstate.done),
            state_metrics.completed_episodes_metrics,
            current_episode_metrics,
        )
        current_episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a * (1 - nstate.done) + b * nstate.done,
            current_episode_metrics,
            nstate.metrics,
        )

        eval_metrics = CompletedEvalMetrics(
            current_episode_metrics=current_episode_metrics,
            completed_episodes_metrics=completed_episodes_metrics,
            completed_episodes=completed_episodes,
            completed_episodes_steps=completed_episodes_steps,
        )
        nstate.info[self.STATE_INFO_KEY] = eval_metrics
        return nstate


class ClipRewardWrapper(Wrapper):
    """Wraps gym environments to clip the reward to be greater than 0.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply clip the reward to be greater than 0.
    """

    def __init__(
        self,
        env: Env,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> None:
        super().__init__(env)
        self._clip_min = clip_min
        self._clip_max = clip_max

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        return state.replace(
            reward=jnp.clip(state.reward, a_min=self._clip_min, a_max=self._clip_max)
        )

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        return state.replace(
            reward=jnp.clip(state.reward, a_min=self._clip_min, a_max=self._clip_max)
        )


class OffsetRewardWrapper(Wrapper):
    """Wraps gym environments to offset the reward to be greater than 0.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply clip the reward to be greater than 0.
    """

    def __init__(self, env: Env, offset: float = 0.0) -> None:
        super().__init__(env)
        self._offset = offset

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        return state.replace(reward=state.reward + self._offset)

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        return state.replace(reward=state.reward + self._offset)
