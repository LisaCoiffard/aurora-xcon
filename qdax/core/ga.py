"""Core components of the vanilla Genetic Algorithm with passive MAP-Elites archive."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)


class GAPassive:
    """Core elements of the MAP-Elites algorithm.

    Note: Although very similar to the GeneticAlgorithm, we decided to keep the
    MAPElites class independent of the GeneticAlgorithm class at the moment to keep
    elements explicit.

    Args:
        scoring_function: a function that takes a batch of genotypes and compute
            their fitnesses and descriptors
        emitter: an emitter is used to suggest offsprings given a MAPELites
            repertoire. It has two compulsory functions. A function that takes
            emits a new population, and a function that update the internal state
            of the emitter.
        metrics_function: a function that takes a MAP-Elites repertoire and compute
            any useful metric to track its evolution
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function

    @partial(jax.jit, static_argnames=("self", "population_size"))
    def init(
        self,
        genotypes: Genotype,
        population_size: int,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[
        MapElitesRepertoire,
        Optional[MapElitesRepertoire],
        Optional[EmitterState],
        RNGKey,
    ]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tessellation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter,
            and a random key.
        """
        # score initial genotypes
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # init the repertoire
        repertoire, survivor_indices = GARepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            population_size=population_size,
        )

        # add passive scores in the passive archive
        passive_genotypes = jax.tree_util.tree_map(
            lambda x: x[survivor_indices], genotypes
        )
        passive_fitnesses = fitnesses[survivor_indices]
        passive_descriptors = extra_scores["passive_descriptors"][survivor_indices]

        passive_repertoire, _ = MapElitesRepertoire.init(
            genotypes=passive_genotypes,
            fitnesses=passive_fitnesses,
            descriptors=passive_descriptors,
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
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return repertoire, passive_repertoire, emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: MapElitesRepertoire,
        passive_repertoire: Optional[MapElitesRepertoire],
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[
        MapElitesRepertoire,
        Optional[MapElitesRepertoire],
        Optional[EmitterState],
        Metrics,
        RNGKey,
    ]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.

        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """
        # generate offsprings with the emitter
        genotypes, extra_info, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )
        # scores the offsprings
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # add genotypes in the repertoire
        repertoire, survivor_indices = repertoire.add(genotypes, fitnesses)

        # add passive scores in the passive archive
        passive_genotypes = jax.tree_util.tree_map(
            lambda x: x[survivor_indices], genotypes
        )
        passive_fitnesses = fitnesses[survivor_indices]
        passive_descriptors = extra_scores["passive_descriptors"][survivor_indices]

        # add genotypes in the passive repertoire
        passive_repertoire, _ = passive_repertoire.add(
            passive_genotypes, passive_descriptors, passive_fitnesses, extra_scores
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

        # update the metrics
        metrics = self._metrics_function(repertoire, passive_repertoire, extra_scores)

        return (
            repertoire,
            passive_repertoire,
            emitter_state,
            metrics,
            random_key,
        )

    @partial(jax.jit, static_argnames=("self",))
    def scan_update(
        self,
        carry: Tuple[
            MapElitesRepertoire,
            Optional[MapElitesRepertoire],
            Optional[EmitterState],
            RNGKey,
        ],
        unused: Any,
    ) -> Tuple[
        Tuple[
            MapElitesRepertoire,
            Optional[MapElitesRepertoire],
            Optional[EmitterState],
            RNGKey,
        ],
        Metrics,
    ]:
        """Rewrites the update function in a way that makes it compatible with the
        jax.lax.scan primitive.

        Args:
            carry: a tuple containing the repertoire, the emitter state and a
                random key.
            unused: unused element, necessary to respect jax.lax.scan API.

        Returns:
            The updated repertoire and emitter state, with a new random key and metrics.
        """
        repertoire, passive_repertoire, emitter_state, random_key = carry
        (
            repertoire,
            passive_repertoire,
            emitter_state,
            metrics,
            random_key,
        ) = self.update(
            repertoire,
            passive_repertoire,
            emitter_state,
            random_key,
        )

        return (repertoire, passive_repertoire, emitter_state, random_key), metrics
