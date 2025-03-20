from typing import Any, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from brax.envs.base import Env, State, Wrapper

from qdax.tasks.environments.base_wrappers import QDEnv

FEET_NAMES = {
    "hopper": ["foot"],
    "walker2d": ["foot", "foot_left"],
    "halfcheetah": ["bfoot", "ffoot"],
    "ant": ["", "", "", ""],
    "humanoid": ["left_shin", "right_shin"],
    "antmaze": ["", "", "", ""],
    "humanoidtrap": ["left_shin", "right_shin"],
}


class FeetContactWrapper(QDEnv):
    """Wraps gym environments to add the feet contact data.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply add the feet_contact booleans in
    the information dictionary of the Brax.state.

    The only supported envs at the moment are among the classic
    locomotion envs : Walker2D, Hopper, Ant, Bullet.

    New locomotions envs can easily be added by adding the config name
    of the feet of the corresponding environment in the FEET_NAME dictionary.

    Example :

        from brax import envs
        from brax import jumpy as jnp

        # choose in ["ant", "walker2d", "hopper", "halfcheetah"]
        ENV_NAME = "ant"
        env = envs.create(env_name=ENV_NAME)
        qd_env = FeetContactWrapper(env, ENV_NAME)

        state = qd_env.reset(rng=jnp.random_prngkey(seed=0))
        for i in range(10):
            action = jnp.zeros((qd_env.action_size,))
            state = qd_env.step(state, action)

            # retrieve feet contact
            feet_contact = state.info["state_descriptor"]

            # do whatever you want with feet_contact
            print(f"Feet contact : {feet_contact}")


    """

    def __init__(self, env: Env, env_name: str):
        if env_name not in FEET_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(
            sys=env.sys, backend=env.backend, debug=env._debug, n_frames=env._n_frames
        )
        self.env = env
        self._env_name = env_name

        if hasattr(self.env, "sys"):
            self._feet_idx = jnp.array(
                [
                    i
                    for i, feet_name in enumerate(self.env.sys.link_names)
                    if feet_name in FEET_NAMES[env_name]
                ]
            )
        else:
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

    @property
    def state_descriptor_length(self) -> int:
        return self.behavior_descriptor_length

    @property
    def state_descriptor_name(self) -> str:
        return "feet_contact"

    @property
    def state_descriptor_limits(self) -> Tuple[List, List]:
        return self.behavior_descriptor_limits

    @property
    def behavior_descriptor_length(self) -> int:
        return len(self._feet_idx)

    @property
    def behavior_descriptor_limits(self) -> Tuple[List, List]:
        bd_length = self.behavior_descriptor_length
        return (jnp.zeros((bd_length,)), jnp.ones((bd_length,)))

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info["state_descriptor"] = self._get_feet_contact(state)
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        state.info["state_descriptor"] = self._get_feet_contact(state)
        return state

    def _get_feet_contact(self, state) -> jnp.ndarray:
        return jnp.any(
            jax.vmap(
                lambda x: (state.pipeline_state.contact.link_idx[1] == x)
                & (state.pipeline_state.contact.dist <= 0)
            )(self._feet_idx),
            axis=-1,
        ).astype(jnp.float32)

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)


# name of the center of gravity
COG_NAMES = {
    "hopper": "torso",
    "walker2d": "torso",
    "halfcheetah": "torso",
    "ant": "torso",
    "antmaze": "torso",
    "humanoid": "torso",
    "humanoidtrap": "torso",
}


class XYPositionWrapper(QDEnv):
    """Wraps gym environments to add the position data.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply add the actual position in
    the information dictionary of the Brax.state.

    One can also add values to clip the state descriptors.

    The only supported envs at the moment are among the classic
    locomotion envs : Ant, Humanoid.

    New locomotions envs can easily be added by adding the config name
    of the feet of the corresponding environment in the STATE_POSITION
    dictionary.

    RMQ: this can be used with Hopper, Walker2d, Halfcheetah but it makes
    less sens as those are limited to one direction.

    Example :

        from brax import envs
        from brax import jumpy as jnp

        # choose in ["ant", "walker2d", "hopper", "halfcheetah", "humanoid"]
        ENV_NAME = "ant"
        env = envs.create(env_name=ENV_NAME)
        qd_env = XYPositionWrapper(env, ENV_NAME)

        state = qd_env.reset(rng=jnp.random_prngkey(seed=0))
        for i in range(10):
            action = jnp.zeros((qd_env.action_size,))
            state = qd_env.step(state, action)

            # retrieve feet contact
            xy_position = state.info["xy_position"]

            # do whatever you want with xy_position
            print(f"xy position : {xy_position}")


    """

    def __init__(
        self,
        env: Env,
        env_name: str,
        minval: Optional[List[float]] = None,
        maxval: Optional[List[float]] = None,
    ):
        if env_name not in COG_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(
            sys=env.sys, backend=env.backend, debug=env._debug, n_frames=env._n_frames
        )
        self.env = env
        self._env_name = env_name

        if hasattr(self.env, "sys"):
            self._cog_idx = self.env.sys.link_names.index(COG_NAMES[env_name])
        else:
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        if minval is None:
            minval = jnp.ones((2,)) * (-jnp.inf)

        if maxval is None:
            maxval = jnp.ones((2,)) * jnp.inf

        if len(minval) == 2 and len(maxval) == 2:
            self._minval = jnp.array(minval)
            self._maxval = jnp.array(maxval)
        else:
            raise NotImplementedError(
                "Please make sure to give two values for each limits."
            )

    @property
    def state_descriptor_length(self) -> int:
        return 2

    @property
    def state_descriptor_name(self) -> str:
        return "xy_position"

    @property
    def state_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return self._minval, self._maxval

    @property
    def behavior_descriptor_length(self) -> int:
        return self.state_descriptor_length

    @property
    def behavior_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return self.state_descriptor_limits

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info["state_descriptor"] = jnp.clip(
            state.pipeline_state.x.pos[self._cog_idx][:2],
            min=self._minval,
            max=self._maxval,
        )
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        state.info["state_descriptor"] = jnp.clip(
            state.pipeline_state.x.pos[self._cog_idx][:2],
            min=self._minval,
            max=self._maxval,
        )
        return state

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)


# name of the forward/velocity reward
FORWARD_REWARD_NAMES = {
    "hopper": "reward_forward",
    "walker2d": "reward_forward",
    "halfcheetah": "reward_run",
    "ant": "reward_forward",
    "antmaze": "reward_forward",
    "humanoid": "reward_linvel",
    "humanoidtrap": "reward_linvel",
}


class NoForwardRewardWrapper(Wrapper):
    """Wraps gym environments to remove forward reward.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply remove the forward speed term
    of the reward.

    Example :

        from brax import envs
        from brax import jumpy as jnp

        # choose in ["ant", "walker2d", "hopper", "halfcheetah", "humanoid"]
        ENV_NAME = "ant"
        env = envs.create(env_name=ENV_NAME)
        qd_env = NoForwardRewardWrapper(env, ENV_NAME)

        state = qd_env.reset(rng=jnp.random_prngkey(seed=0))
        for i in range(10):
            action = jnp.zeros((qd_env.action_size,))
            state = qd_env.step(state, action)
    """

    def __init__(self, env: Env, env_name: str) -> None:
        if env_name not in FORWARD_REWARD_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(env)
        self._env_name = env_name
        self._forward_reward_name = FORWARD_REWARD_NAMES[env_name]

    @property
    def name(self) -> str:
        return self._env_name

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        new_reward = state.reward - state.metrics[self._forward_reward_name]
        return state.replace(reward=new_reward)  # type: ignore


class XYPositionFeetInfoWrapper(QDEnv):
    """Wraps gym environments to add the position data.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply add the actual position in
    the information dictionary of the Brax.state.

    One can also add values to clip the state descriptors.

    The only supported envs at the moment are among the classic
    locomotion envs : Ant, Humanoid.

    New locomotions envs can easily be added by adding the config name
    of the feet of the corresponding environment in the STATE_POSITION
    dictionary.

    RMQ: this can be used with Hopper, Walker2d, Halfcheetah but it makes
    less sens as those are limited to one direction.

    Example :

        from brax import envs
        from brax import jumpy as jnp

        # choose in ["ant", "walker2d", "hopper", "halfcheetah", "humanoid"]
        ENV_NAME = "ant"
        env = envs.create(env_name=ENV_NAME)
        qd_env = XYPositionWrapper(env, ENV_NAME)

        state = qd_env.reset(rng=jnp.random_prngkey(seed=0))
        for i in range(10):
            action = jnp.zeros((qd_env.action_size,))
            state = qd_env.step(state, action)

            # retrieve feet contact
            xy_position = state.info["xy_position"]

            # do whatever you want with xy_position
            print(f"xy position : {xy_position}")


    """

    def __init__(
        self,
        env: Env,
        env_name: str,
        minval: Optional[List[float]] = None,
        maxval: Optional[List[float]] = None,
    ):
        if env_name not in COG_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(
            sys=env.sys, backend=env.backend, debug=env._debug, n_frames=env._n_frames
        )

        self.env = env
        self._env_name = env_name
        if hasattr(self.env, "sys"):
            self._cog_idx = self.env.sys.link_names.index(COG_NAMES[env_name])
            self._feet_idx = jnp.array(
                [
                    i
                    for i, feet_name in enumerate(self.env.sys.link_names)
                    if feet_name in FEET_NAMES[env_name]
                ]
            )
        else:
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        if minval is None:
            minval = jnp.ones((2,)) * (-jnp.inf)

        if maxval is None:
            maxval = jnp.ones((2,)) * jnp.inf

        if len(minval) == 2 and len(maxval) == 2:
            self._minval = jnp.array(minval)
            self._maxval = jnp.array(maxval)
        else:
            raise NotImplementedError(
                "Please make sure to give two values for each limits."
            )

    @property
    def state_descriptor_length(self) -> int:
        return 2

    @property
    def state_descriptor_name(self) -> str:
        return "xy_position"

    @property
    def state_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return self._minval, self._maxval

    @property
    def behavior_descriptor_length(self) -> int:
        return self.state_descriptor_length

    @property
    def behavior_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return self.state_descriptor_limits

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info["state_descriptor"] = jnp.clip(
            state.pipeline_state.x.pos[self._cog_idx][:2],
            min=self._minval,
            max=self._maxval,
        )
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        # get xy position of the center of gravity
        state.info["state_descriptor"] = jnp.clip(
            state.pipeline_state.x.pos[self._cog_idx][:2],
            min=self._minval,
            max=self._maxval,
        )
        return state

    def _get_feet_contact(self, state) -> jnp.ndarray:
        return jnp.any(
            jax.vmap(
                lambda x: (state.pipeline_state.contact.link_idx[1] == x)
                & (state.pipeline_state.contact.dist <= 0)
            )(self._feet_idx),
            axis=-1,
        ).astype(jnp.float32)

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)


class PassiveDescWrapper(Wrapper):
    """
    Wraps gym environment to add the passive behavior descriptor and vector of rewards.
    """

    def __init__(
        self,
        env: Env,
        env_name: str,
    ):

        if (
            env_name not in FEET_NAMES.keys()
            or env_name not in COG_NAMES.keys()
            or env_name not in FORWARD_REWARD_NAMES.keys()
        ):
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(env)

        self._env_name = env_name

        self._fd_reward_field = FORWARD_REWARD_NAMES[env_name]

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info["passive_descriptor"] = self._get_feet_contact(state)
        new_reward = jnp.zeros((3,))
        return state.replace(reward=new_reward)

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        state.info["passive_descriptor"] = self._get_feet_contact(state)
        # update the reward (vector of full ant reward, no forward reward and forward reward only)
        new_reward = jnp.concatenate(
            (
                jnp.array((state.reward,)),
                jnp.array((state.reward - state.metrics[self._fd_reward_field],)),
                jnp.array((state.metrics[self._fd_reward_field],)),
            ),
            axis=-1,
        )
        return state.replace(reward=new_reward)  # type: ignore


class StandDescWrapper(QDEnv):

    def __init__(
        self,
        env: Env,
        env_name: str,
        minval: Optional[List[float]] = None,
        maxval: Optional[List[float]] = None,
    ):

        if env_name not in FEET_NAMES.keys() or env_name not in COG_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(
            sys=env.sys, backend=env.backend, debug=env._debug, n_frames=env._n_frames
        )

        self.env = env
        self._env_name = env_name
        if hasattr(self.env, "sys"):
            self._cog_idx = self.env.sys.link_names.index(COG_NAMES[env_name])
            self._feet_idx = jnp.array(
                [
                    i
                    for i, feet_name in enumerate(self.env.sys.link_names)
                    if feet_name in FEET_NAMES[env_name]
                ]
            )
        else:
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        if minval is None:
            minval = jnp.ones((4,)) * (-jnp.inf)

        if maxval is None:
            maxval = jnp.ones((4,)) * jnp.inf

        if len(minval) == 4 and len(maxval) == 4:
            self._minval = jnp.array(minval)
            self._maxval = jnp.array(maxval)
        else:
            raise NotImplementedError(
                "Please make sure to give two values for each limits."
            )

    @property
    def state_descriptor_length(self) -> int:
        return self.behavior_descriptor_length

    @property
    def state_descriptor_name(self) -> str:
        return "stand_descriptor"

    @property
    def state_descriptor_limits(self) -> Tuple[List, List]:
        return self.behavior_descriptor_limits

    @property
    def behavior_descriptor_length(self) -> int:
        return len(self._feet_idx) + 4

    @property
    def behavior_descriptor_limits(self) -> Tuple[List, List]:
        bd_length = self.behavior_descriptor_length
        limits = (jnp.zeros((bd_length,)), jnp.ones((bd_length,)))
        limits[0][4:], limits[1][4:] = self._minval, self._maxval
        return limits

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        feet_contacts = self._get_feet_contact(state)
        state.info["state_descriptor"] = jnp.concatenate(
            (
                feet_contacts,
                self._get_z_torso_position(state),
                self._get_z_torso_orientation(state),
                self._get_xy_torso_position(state),
            ),
            axis=-1,
        )
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        feet_contacts = self._get_feet_contact(state)
        state.info["state_descriptor"] = jnp.concatenate(
            (
                feet_contacts,
                self._get_z_torso_position(state),
                self._get_z_torso_orientation(state),
                self._get_xy_torso_position(state),
            ),
            axis=-1,
        )
        return state

    def _get_xy_torso_position(self, state) -> jnp.ndarray:
        xy_pos = jnp.clip(
            state.pipeline_state.x.pos[self._cog_idx][:2],
            min=self._minval[2:],
            max=self._maxval[2:],
        )
        return xy_pos

    def _get_feet_contact(self, state) -> jnp.ndarray:
        return jnp.any(
            jax.vmap(
                lambda x: (state.pipeline_state.contact.link_idx[1] == x)
                & (state.pipeline_state.contact.dist <= 0)
            )(self._feet_idx),
            axis=-1,
        ).astype(jnp.float32)

    def _get_z_torso_position(self, state) -> jnp.ndarray:
        z_pos = jnp.clip(
            state.pipeline_state.x.pos[self._cog_idx][2],
            min=self._minval[0],
            max=self._maxval[0],
        )
        return jnp.expand_dims(
            z_pos,
            axis=-1,
        )

    def _get_z_torso_orientation(self, state) -> jnp.ndarray:
        z_rot = jnp.clip(
            state.pipeline_state.x.rot[self._cog_idx][2],
            min=self._minval[1],
            max=self._maxval[1],
        )
        return jnp.expand_dims(
            z_rot,
            axis=-1,
        )

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)


class MultiDescRewardWrapper(QDEnv):

    def __init__(
        self,
        env: Env,
        env_name: str,
        minval: Optional[List[float]] = None,
        maxval: Optional[List[float]] = None,
    ):

        if (
            env_name not in FEET_NAMES.keys()
            or env_name not in COG_NAMES.keys()
            or env_name not in FORWARD_REWARD_NAMES.keys()
        ):
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(
            sys=env.sys, backend=env.backend, debug=env._debug, n_frames=env._n_frames
        )

        self.env = env
        self._env_name = env_name
        if hasattr(self.env, "sys"):
            self._cog_idx = self.env.sys.link_names.index(COG_NAMES[env_name])
            self._feet_idx = jnp.array(
                [
                    i
                    for i, feet_name in enumerate(self.env.sys.link_names)
                    if feet_name in FEET_NAMES[env_name]
                ]
            )
        else:
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        # TODO: set a variable for the dim of the limits
        if minval is None:
            minval = jnp.ones((6,)) * (-jnp.inf)

        if maxval is None:
            maxval = jnp.ones((6,)) * jnp.inf

        if len(minval) == 6 and len(maxval) == 6:
            self._minval = jnp.array(minval)
            self._maxval = jnp.array(maxval)
        else:
            raise NotImplementedError(
                "Please make sure to give six values for each limits."
            )

        self._fd_reward_field = FORWARD_REWARD_NAMES[env_name]

    @property
    def state_descriptor_length(self) -> int:
        return self.behavior_descriptor_length

    @property
    def state_descriptor_name(self) -> str:
        return "multi_descriptor"

    @property
    def state_descriptor_limits(self) -> Tuple[List, List]:
        return self.behavior_descriptor_limits

    @property
    def behavior_descriptor_length(self) -> int:
        return 2 * len(self._feet_idx) + 6

    @property
    def behavior_descriptor_limits(self) -> Tuple[List, List]:
        bd_length = self.behavior_descriptor_length
        limits = (jnp.zeros((bd_length,)), jnp.ones((bd_length,)))
        limits[0][4:], limits[1][4:] = self._minval, self._maxval
        return limits

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        feet_contacts = self._get_feet_contact(state)
        joint_angles = self._get_joint_angles(state)

        state.info["state_descriptor"] = jnp.concatenate(
            (
                feet_contacts,
                self._get_torso_position(state),
                self._get_torso_orientation(state),
                joint_angles,
            ),
            axis=-1,
        )
        new_reward = jnp.zeros((3,))
        return state.replace(reward=new_reward)  # type: ignore

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        feet_contacts = self._get_feet_contact(state)
        joint_angles = self._get_joint_angles(state)

        state.info["state_descriptor"] = jnp.concatenate(
            (
                feet_contacts,
                self._get_torso_position(state),
                self._get_torso_orientation(state),
                joint_angles,
            ),
            axis=-1,
        )
        # update the reward (vector of full ant reward, no forward reward and forward reward only)
        new_reward = jnp.concatenate(
            (
                jnp.array((state.reward,)),
                jnp.array((state.reward - state.metrics[self._fd_reward_field],)),
                jnp.array((state.metrics[self._fd_reward_field],)),
            ),
            axis=-1,
        )
        return state.replace(reward=new_reward)  # type: ignore

    def _get_torso_position(self, state) -> jnp.ndarray:
        torso_pos = jnp.clip(
            state.pipeline_state.x.pos[self._cog_idx][:3],
            min=self._minval[0:3],
            max=self._maxval[0:3],
        )
        return torso_pos

    def _get_joint_angles(self, state) -> jnp.ndarray:
        return jnp.zeros(
            4,
        )

    def _get_feet_contact(self, state) -> jnp.ndarray:
        return jnp.any(
            jax.vmap(
                lambda x: (state.pipeline_state.contact.link_idx[1] == x)
                & (state.pipeline_state.contact.dist <= 0)
            )(self._feet_idx),
            axis=-1,
        ).astype(jnp.float32)

    def _get_torso_orientation(self, state) -> jnp.ndarray:
        torso_rot = jnp.clip(
            state.pipeline_state.x.rot[self._cog_idx][:3],
            min=self._minval[3:6],
            max=self._maxval[3:6],
        )
        return torso_rot

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)
