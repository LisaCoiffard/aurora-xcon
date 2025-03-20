"""Core class of the AURORA algorithm with Threshold Repertoire."""

from __future__ import annotations

from ast import Dict
from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import ArrayTree
from flax.training.train_state import TrainState

from qdax.core.containers.l_value_archive import (
    UnstructuredRepertoire,
    UnstructuredRepertoirePassiveDesc,
)
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import (
    Centroid,
    Descriptor,
    Fitness,
    Genotype,
    Metrics,
    Observation,
    Params,
    RNGKey,
)
from qdax.tasks.environments.bd_extractors import AuroraExtraInfo


class AURORAThreshold:
    """Core elements of the AURORA algorithm.

    Args:
        scoring_function: a function that takes a batch of genotypes and compute
            their fitnesses and descriptors
        emitter: an emitter is used to suggest offsprings given a MAPELites
            repertoire. It has two compulsory functions. A function that takes
            emits a new population, and a function that update the internal state
            of the emitter.
        metrics_function: a function that takes a repertoire and computes
            any useful metric to track its evolution
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey],
            Tuple[Fitness, Descriptor, ArrayTree, RNGKey],
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        encoder_function: Callable[[Observation, AuroraExtraInfo], Descriptor],
        training_function: Callable[
            [RNGKey, UnstructuredRepertoire, Params, int], AuroraExtraInfo
        ],
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._encoder_fn = encoder_function
        self._train_fn = training_function

    def train(
        self,
        repertoire: UnstructuredRepertoire,
        train_state: TrainState,
        random_key: RNGKey,
    ) -> Tuple[UnstructuredRepertoire, AuroraExtraInfo]:
        random_key, subkey = jax.random.split(random_key)
        train_state, aurora_extra_info, metrics = self._train_fn(
            repertoire,
            train_state,
            random_key,
        )

        # re-addition of all the new behavioural descriptors with the new ae
        new_descriptors = self._encoder_fn(repertoire.observations, aurora_extra_info)

        return (
            repertoire.init(
                genotypes=repertoire.genotypes,
                fitnesses=repertoire.fitnesses,
                descriptors=new_descriptors,
                observations=repertoire.observations,
                l_value=repertoire.l_value,
                max_size=repertoire.max_size,
            ),
            train_state,
            aurora_extra_info,
            metrics,
        )

    @partial(jax.jit, static_argnames=("self", "csc_orig"))
    def container_size_control(
        self,
        repertoire: UnstructuredRepertoire,
        target_size: int,
        prop_gain: float,
        previous_error: jnp.ndarray,
        csc_orig: bool,
    ) -> Tuple[UnstructuredRepertoire, jnp.ndarray]:
        # update the l value
        num_indivs = jnp.sum(repertoire.fitnesses != -jnp.inf)

        # CVC Implementation to keep a constant number of individuals in the archive
        current_error = num_indivs - target_size
        change_rate = current_error - previous_error
        # prop_gain = 1 * 10e-6
        if csc_orig:
            l_value = repertoire.l_value + (prop_gain * current_error)
        else:
            l_value = (
                repertoire.l_value
                + (prop_gain * current_error)
                + (prop_gain * change_rate)
            )

        repertoire = repertoire.init(
            genotypes=repertoire.genotypes,
            fitnesses=repertoire.fitnesses,
            descriptors=repertoire.descriptors,
            observations=repertoire.observations,
            l_value=l_value,
            max_size=repertoire.max_size,
        )

        return repertoire, current_error

    def init(
        self,
        genotypes: Genotype,
        aurora_extra_info: AuroraExtraInfo,
        l_value: jnp.ndarray,
        max_size: int,
        random_key: RNGKey,
    ) -> Tuple[UnstructuredRepertoire, Optional[EmitterState], AuroraExtraInfo, RNGKey]:
        """Initialize an unstructured repertoire with an initial population of
        genotypes. Also performs the first training of the AURORA encoder.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            aurora_extra_info: information to perform AURORA encodings,
                such as the encoder parameters
            l_value: threshold distance for the unstructured repertoire
            max_size: maximum size of the repertoire
            random_key: a random key used for stochastic operations.

        Returns:
            an initialized unstructured repertoire, with the initial state of
            the emitter, and the updated information to perform AURORA encodings
        """
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes,
            random_key,
        )

        observations = extra_scores["observations"]

        descriptors = self._encoder_fn(observations, aurora_extra_info)

        repertoire = UnstructuredRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            observations=observations,
            l_value=l_value,
            max_size=max_size,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            random_key=random_key,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        random_key, subkey = jax.random.split(random_key)

        return repertoire, emitter_state, aurora_extra_info, random_key

    @partial(jax.jit, static_argnames=("self",))
    def ask(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, Dict[str, ArrayTree], RNGKey]:
        """Ask the AURORA algorithm to generate a new batch of genotypes.

        Args:
            repertoire: unstructured repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns:
            a batch of genotypes
            extra information about the genotypes
            a new key
        """
        genotypes, extra_info, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        return genotypes, extra_info, random_key

    @partial(jax.jit, static_argnames=("self",))
    def tell(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        observations: Observation,
        extra_scores: ArrayTree,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState]]:
        """Tell the AURORA algorithm the fitnesses of a batch of genotypes.

        Args:
            repertoire: unstructured repertoire
            emitter_state: state of the emitter
            genotypes: a batch of genotypes
            fitnesses: the fitnesses of the genotypes
            descriptors: the descriptors of the genotypes
            extra_scores: extra information about the genotypes

        Returns:
            the updated repertoire
            the updated emitter state
        """
        # add genotypes and observations in the repertoire
        repertoire = repertoire.add(
            genotypes,
            descriptors,
            fitnesses,
            observations,
        )

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return repertoire, emitter_state

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
        aurora_extra_info: AuroraExtraInfo,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        """Main step of the AURORA algorithm.


        Performs one iteration of the AURORA algorithm.
        1. A batch of genotypes is sampled in the archive and the genotypes are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the archive.

        Args:
            repertoire: unstructured repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key
            aurora_extra_info: extra info for computing encodings

        Results:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new key
        """
        # generate offsprings with the emitter
        genotypes, extra_info, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # scores the offsprings
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes,
            random_key,
        )

        observations = extra_scores["observations"]

        descriptors = self._encoder_fn(observations, aurora_extra_info)

        # add genotypes and observations in the repertoire
        repertoire = repertoire.add(
            genotypes,
            descriptors,
            fitnesses,
            observations,
        )

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores | extra_info,
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key
    
class AURORAThresholdPassive:
    """Core elements of the AURORA algorithm which maintains a passive repertoire of 
    solutions according to hand-coded BDs.

    Args:
        scoring_function: a function that takes a batch of genotypes and compute
            their fitnesses and descriptors
        emitter: an emitter is used to suggest offsprings given a MAPELites
            repertoire. It has two compulsory functions. A function that takes
            emits a new population, and a function that update the internal state
            of the emitter.
        metrics_function: a function that takes a repertoire and computes
            any useful metric to track its evolution
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey],
            Tuple[Fitness, Descriptor, ArrayTree, RNGKey],
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        encoder_function: Callable[[Observation, AuroraExtraInfo], Descriptor],
        training_function: Callable[
            [RNGKey, UnstructuredRepertoirePassiveDesc, Params, int], AuroraExtraInfo
        ],
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._encoder_fn = encoder_function
        self._train_fn = training_function

    def train(
        self,
        repertoire: UnstructuredRepertoirePassiveDesc,
        train_state: TrainState,
        random_key: RNGKey,
    ) -> Tuple[UnstructuredRepertoirePassiveDesc, AuroraExtraInfo]:
        random_key, subkey = jax.random.split(random_key)
        train_state, aurora_extra_info, metrics = self._train_fn(
            repertoire,
            train_state,
            random_key,
        )

        # re-addition of all the new behavioural descriptors with the new ae
        new_descriptors = self._encoder_fn(repertoire.observations, aurora_extra_info)

        return (
            repertoire.init(
                genotypes=repertoire.genotypes,
                fitnesses=repertoire.fitnesses,
                descriptors=new_descriptors,
                passive_descriptors=repertoire.passive_descriptors,
                observations=repertoire.observations,
                l_value=repertoire.l_value,
                max_size=repertoire.max_size,
            ),
            train_state,
            aurora_extra_info,
            metrics,
        )

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "csc_orig",
        ),
    )
    def container_size_control(
        self,
        repertoire: UnstructuredRepertoirePassiveDesc,
        target_size: int,
        prop_gain: float,
        previous_error: jnp.ndarray,
        csc_orig: bool,
    ) -> Tuple[UnstructuredRepertoirePassiveDesc, jnp.ndarray]:
        # update the l value
        num_indivs = jnp.sum(repertoire.fitnesses != -jnp.inf)

        # CVC Implementation to keep a constant number of individuals in the archive
        current_error = num_indivs - target_size
        change_rate = current_error - previous_error
        # prop_gain = 1 * 10e-5
        if csc_orig:
            l_value = repertoire.l_value + (prop_gain * current_error)
        else:
            l_value = (
                repertoire.l_value
                + (prop_gain * current_error)
                + (prop_gain * change_rate)
            )

        repertoire = repertoire.init(
            genotypes=repertoire.genotypes,
            fitnesses=repertoire.fitnesses,
            descriptors=repertoire.descriptors,
            passive_descriptors=repertoire.passive_descriptors,
            observations=repertoire.observations,
            l_value=l_value,
            max_size=repertoire.max_size,
        )

        return repertoire, current_error

    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        aurora_extra_info: AuroraExtraInfo,
        l_value: jnp.ndarray,
        max_size: int,
        random_key: RNGKey,
    ) -> Tuple[UnstructuredRepertoirePassiveDesc, Optional[EmitterState], AuroraExtraInfo, RNGKey]:
        """Initialize an unstructured repertoire with an initial population of
        genotypes. Also performs the first training of the AURORA encoder.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            aurora_extra_info: information to perform AURORA encodings,
                such as the encoder parameters
            l_value: threshold distance for the unstructured repertoire
            max_size: maximum size of the repertoire
            random_key: a random key used for stochastic operations.

        Returns:
            an initialized unstructured and passive MAP-Elites repertoire, 
            with the initial state of the emitter, 
            and the updated information to perform AURORA encodings
        """
        fitnesses, passive_descriptors, extra_scores, random_key = (
            self._scoring_function(
                genotypes,
                random_key,
            )
        )

        observations = extra_scores["observations"]

        descriptors = self._encoder_fn(observations, aurora_extra_info)

        repertoire = UnstructuredRepertoirePassiveDesc.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            passive_descriptors=passive_descriptors,
            observations=observations,
            l_value=l_value,
            max_size=max_size,
        )

        passive_repertoire, _ = MapElitesRepertoire.init(
            genotypes=repertoire.genotypes,
            fitnesses=repertoire.fitnesses,
            descriptors=repertoire.passive_descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            random_key=random_key,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        random_key, subkey = jax.random.split(random_key)

        return (
            repertoire,
            passive_repertoire,
            emitter_state,
            aurora_extra_info,
            random_key,
        )

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: MapElitesRepertoire,
        passive_repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
        aurora_extra_info: AuroraExtraInfo,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        """Main step of the AURORA algorithm.


        Performs one iteration of the AURORA algorithm.
        1. A batch of genotypes is sampled in the archive and the genotypes are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the archive.

        Args:
            repertoire: unstructured repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key
            aurora_extra_info: extra info for computing encodings

        Results:
            the updated unsutructured repertoire
            the updated MAP-Elites passive repertoire            
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new key
        """
        # generate offsprings with the emitter
        genotypes, extra_info, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )
        # scores the offsprings
        fitnesses, passive_descriptors, extra_scores, random_key = (
            self._scoring_function(
                genotypes,
                random_key,
            )
        )

        observations = extra_scores["observations"]

        descriptors = self._encoder_fn(observations, aurora_extra_info)

        # add genotypes and observations in the repertoire
        repertoire = repertoire.add(
            genotypes,
            descriptors,
            passive_descriptors,
            fitnesses,
            observations,
        )

        passive_repertoire, _ = passive_repertoire.add(
            repertoire.genotypes,
            repertoire.passive_descriptors,
            repertoire.fitnesses,
            extra_scores,
        )

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update the metrics
        metrics = self._metrics_function(repertoire, passive_repertoire)

        return repertoire, passive_repertoire, emitter_state, metrics, random_key

