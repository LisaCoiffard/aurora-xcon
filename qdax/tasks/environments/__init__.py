import functools
from typing import Any, Callable, List, Optional, Union

import brax
from brax.envs.base import Env
from brax.envs.wrappers import training

from qdax.tasks.environments.base_wrappers import QDEnv, StateDescriptorResetWrapper
from qdax.tasks.environments.bd_extractors import (
    get_feet_contact_proportion,
    get_final_xy_position,
)
from qdax.tasks.environments.init_state_wrapper import FixedInitialStateWrapper
from qdax.tasks.environments.locomotion_wrappers import (
    FeetContactWrapper,
    NoForwardRewardWrapper,
    XYPositionWrapper,
)
from qdax.tasks.environments.wrappers import (
    ClipRewardWrapper,
    CompletedEvalWrapper,
)

# experimentally determinated offset
# should be sufficient to have only positive rewards but no guarantee
reward_offset = {
    "hopper_uni": 0.9,
    "walker2d_uni": 1.413,
    "halfcheetah_uni": 9.231,
    "ant_uni": 3.24,
    "ant_omni": 3.0,
    "anttrap": 3.38,
    "humanoid_uni": 0.0,
    "humanoid_omni": 0.0,
}

behavior_descriptor_extractor = {
    "hopper_uni": get_feet_contact_proportion,
    "walker2d_uni": get_feet_contact_proportion,
    "halfcheetah_uni": get_feet_contact_proportion,
    "ant_uni": get_feet_contact_proportion,
    "ant_omni": get_final_xy_position,
    "anttrap_uni": get_final_xy_position,
    "anttrap_omni": get_final_xy_position,
    "humanoid_uni": get_feet_contact_proportion,
    "humanoid_omni": get_final_xy_position,
}

_qdax_custom_envs = {
    "hopper_uni": {
        "env": "hopper",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
    },
    "walker2d_uni": {
        "env": "walker2d",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
    },
    "halfcheetah_uni": {
        "env": "halfcheetah",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
    },
    "ant_uni": {"env": "ant", "wrappers": [FeetContactWrapper], "kwargs": [{}]},
    "ant_omni": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, NoForwardRewardWrapper],
        "kwargs": [{"minval": [-30.0, -30.0], "maxval": [30.0, 30.0]}, {}, {}],
    },
    "anttrap_omni": {
        "env": "anttrap",
        "wrappers": [XYPositionWrapper, NoForwardRewardWrapper],
        "kwargs": [{"minval": [-30.0, -30.0], "maxval": [30.0, 30.0]}, {}, {}],
    },
    "humanoid_uni": {
        "env": "humanoid",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
    },
    "humanoid_omni": {
        "env": "humanoid",
        "wrappers": [XYPositionWrapper, NoForwardRewardWrapper],
        "kwargs": [{"minval": [-10.0, -10.0], "maxval": [10.0, 10.0]}, {}],
    },
}


def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    eval_metrics: bool = False,
    fixed_init_state: bool = False,
    qdax_wrappers_kwargs: Optional[List] = None,
    **kwargs: Any,
) -> Union[Env, QDEnv]:
    """Creates an Env with a specified brax system.
    Please use namespace to avoid confusion between this function and
    brax.envs.create.
    """
    if env_name in brax.envs._envs.keys():
        base_env_name = env_name
        env = brax.envs._envs[env_name](debug=True, **kwargs)
    elif env_name in _qdax_custom_envs.keys():
        # Create env
        base_env_name = _qdax_custom_envs[env_name]["env"]
        env = brax.envs._envs[base_env_name](debug=True, **kwargs)

        # Apply wrappers
        wrappers = _qdax_custom_envs[env_name]["wrappers"]
        if qdax_wrappers_kwargs is None:
            kwargs_list = _qdax_custom_envs[env_name]["kwargs"]
        else:
            kwargs_list = qdax_wrappers_kwargs
        for wrapper, kwargs in zip(wrappers, kwargs_list):  # type: ignore
            env = wrapper(env, base_env_name, **kwargs)  # type: ignore
    else:
        raise NotImplementedError("This environment name does not exist!")

    if episode_length is not None:
        env = training.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = training.VmapWrapper(env, batch_size)
    if auto_reset:
        env = training.AutoResetWrapper(env)
        if env_name in _qdax_custom_envs.keys():
            env = StateDescriptorResetWrapper(env)
    if fixed_init_state:
        env = FixedInitialStateWrapper(env, base_env_name)  # type: ignore
    if eval_metrics:
        env = training.EvalWrapper(env)
        env = CompletedEvalWrapper(env)
    env = ClipRewardWrapper(env)
    return env


def create_fn(env_name: str, **kwargs: Any) -> Callable[..., brax.envs.Env]:
    """Returns a function that when called, creates an Env.
    Please use namespace to avoid confusion between this function and
    brax.envs.create_fn.
    """
    return functools.partial(create, env_name, **kwargs)
