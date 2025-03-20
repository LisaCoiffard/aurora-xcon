"""Defines functions to retrieve metrics from training processes."""

from __future__ import annotations

import csv
from functools import partial
from typing import Dict, List

import jax
from jax import numpy as jnp

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.custom_types import Metrics
from qdax.utils.pareto_front import compute_hypervolume


class CSVLogger:
    """Logger to save metrics of an experiment in a csv file
    during the training process.
    """

    def __init__(self, filename: str, header: List) -> None:
        """Create the csv logger, create a file and write the
        header.

        Args:
            filename: path to which the file will be saved.
            header: header of the csv file.
        """
        self._filename = filename
        self._header = header
        with open(self._filename, "w") as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            # write the header
            writer.writeheader()

    def log(self, metrics: Dict[str, float]) -> None:
        """Log new metrics to the csv file.

        Args:
            metrics: A dictionary containing the metrics that
                need to be saved.
        """
        with open(self._filename, "a") as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            # write new metrics in a raw
            writer.writerow(metrics)


def default_ga_metrics(
    repertoire: GARepertoire,
) -> Metrics:
    """Compute the usual GA metrics that one can retrieve
    from a GA repertoire.

    Args:
        repertoire: a GA repertoire

    Returns:
        a dictionary containing the max fitness of the
            repertoire.
    """

    # get metrics
    max_fitness = jnp.max(repertoire.fitnesses, axis=0)

    return {
        "max_fitness": max_fitness,
    }


def default_qd_metrics(repertoire: MapElitesRepertoire, qd_offset: float) -> Metrics:
    """Compute the usual QD metrics that one can retrieve
    from a MAP Elites repertoire.

    Args:
        repertoire: a MAP-Elites repertoire
        qd_offset: an offset used to ensure that the QD score
            will be positive and increasing with the number
            of individuals.

    Returns:
        a dictionary containing the QD score (sum of fitnesses
            modified to be all positive), the max fitness of the
            repertoire, the coverage (number of niche filled in
            the repertoire).
    """

    # get metrics
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
    qd_score += qd_offset * jnp.sum(1.0 - repertoire_empty)
    coverage = 100 * jnp.mean(1.0 - repertoire_empty)
    max_fitness = jnp.max(repertoire.fitnesses)

    return {"qd_score": qd_score, "max_fitness": max_fitness, "coverage": coverage}


def default_moqd_metrics(
    repertoire: MOMERepertoire, reference_point: jnp.ndarray
) -> Metrics:
    """Compute the MOQD metric given a MOME repertoire and a reference point.

    Args:
        repertoire: a MOME repertoire.
        reference_point: the hypervolume of a pareto front has to be computed
            relatively to a reference point.

    Returns:
        A dictionary containing all the computed metrics.
    """
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    repertoire_empty = jnp.all(repertoire_empty, axis=-1)
    repertoire_not_empty = ~repertoire_empty
    repertoire_not_empty = jnp.any(repertoire_not_empty, axis=-1)
    coverage = 100 * jnp.mean(repertoire_not_empty)
    hypervolume_function = partial(compute_hypervolume, reference_point=reference_point)
    moqd_scores = jax.vmap(hypervolume_function)(repertoire.fitnesses)
    moqd_scores = jnp.where(repertoire_not_empty, moqd_scores, -jnp.inf)
    max_hypervolume = jnp.max(moqd_scores)
    max_scores = jnp.max(repertoire.fitnesses, axis=(0, 1))
    max_sum_scores = jnp.max(jnp.sum(repertoire.fitnesses, axis=-1), axis=(0, 1))
    num_solutions = jnp.sum(~repertoire_empty)
    (
        pareto_front,
        _,
    ) = repertoire.compute_global_pareto_front()

    global_hypervolume = compute_hypervolume(
        pareto_front, reference_point=reference_point
    )
    metrics = {
        "moqd_score": moqd_scores,
        "max_hypervolume": max_hypervolume,
        "max_scores": max_scores,
        "max_sum_scores": max_sum_scores,
        "coverage": coverage,
        "number_solutions": num_solutions,
        "global_hypervolume": global_hypervolume,
    }

    return metrics


def multi_passive_qd_metrics(
    repertoire: MapElitesRepertoire,
    passive_repertoire_1: MapElitesRepertoire,
    passive_repertoire_2: MapElitesRepertoire,
    passive_repertoire_3: MapElitesRepertoire,
    qd_offset: float,
) -> Metrics:
    """Compute the usual QD metrics that one can retrieve
    from a MAP Elites repertoire.

    Args:
        repertoire: a MAP-Elites repertoire
        qd_offset: an offset used to ensure that the QD score
            will be positive and increasing with the number
            of individuals.

    Returns:
        a dictionary containing the QD score (sum of fitnesses
            modified to be all positive), the max fitness of the
            repertoire, the coverage (number of niche filled in
            the repertoire).
    """

    # get metrics
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
    qd_score += qd_offset * jnp.sum(1.0 - repertoire_empty)
    coverage = 100 * jnp.mean(1.0 - repertoire_empty)
    max_fitness = jnp.max(repertoire.fitnesses)

    # metrics for passive repertoires
    passive_repertoire_1_empty = passive_repertoire_1.fitnesses == -jnp.inf
    p_qd_score_1 = jnp.sum(
        passive_repertoire_1.fitnesses, where=~passive_repertoire_1_empty
    )
    p_qd_score_1 += qd_offset * jnp.sum(1.0 - passive_repertoire_1_empty)
    p_coverage_1 = 100 * jnp.mean(1.0 - passive_repertoire_1_empty)
    p_max_fitness_1 = jnp.max(passive_repertoire_1.fitnesses)

    passive_repertoire_2_empty = passive_repertoire_2.fitnesses == -jnp.inf
    p_qd_score_2 = jnp.sum(
        passive_repertoire_2.fitnesses, where=~passive_repertoire_2_empty
    )
    p_qd_score_2 += qd_offset * jnp.sum(1.0 - passive_repertoire_2_empty)
    p_coverage_2 = 100 * jnp.mean(1.0 - passive_repertoire_2_empty)
    p_max_fitness_2 = jnp.max(passive_repertoire_2.fitnesses)

    passive_repertoire_3_empty = passive_repertoire_3.fitnesses == -jnp.inf
    p_qd_score_3 = jnp.sum(
        passive_repertoire_3.fitnesses, where=~passive_repertoire_3_empty
    )
    p_qd_score_3 += qd_offset * jnp.sum(1.0 - passive_repertoire_3_empty)
    p_coverage_3 = 100 * jnp.mean(1.0 - passive_repertoire_3_empty)
    p_max_fitness_3 = jnp.max(passive_repertoire_3.fitnesses)

    return {
        "qd_score": qd_score,
        "max_fitness": max_fitness,
        "coverage": coverage,
        "passive_qd_score_1": p_qd_score_1,
        "passive_max_fitness_1": p_max_fitness_1,
        "passive_coverage_1": p_coverage_1,
        "passive_qd_score_2": p_qd_score_2,
        "passive_max_fitness_2": p_max_fitness_2,
        "passive_coverage_2": p_coverage_2,
        "passive_qd_score_3": p_qd_score_3,
        "passive_max_fitness_3": p_max_fitness_3,
        "passive_coverage_3": p_coverage_3,
    }


def passive_qd_metrics(
    repertoire: MapElitesRepertoire,
    passive_repertoire: MapElitesRepertoire,
    extra_scores: jnp.ndarray,
    qd_offset: float,
    target_size: bool = None,
) -> Metrics:
    """Compute the usual QD metrics that one can retrieve
    from a MAP Elites repertoire.

    Args:
        repertoire: a MAP-Elites repertoire
        qd_offset: an offset used to ensure that the QD score
            will be positive and increasing with the number
            of individuals.

    Returns:
        a dictionary containing the QD score (sum of fitnesses
            modified to be all positive), the max fitness of the
            repertoire, the coverage (number of niche filled in
            the repertoire).
    """

    # get metrics
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
    qd_score += qd_offset * jnp.sum(1.0 - repertoire_empty)
    coverage = 100 * jnp.mean(1.0 - repertoire_empty)
    max_fitness = jnp.max(repertoire.fitnesses)

    if target_size is not None:
        coverage = (jnp.sum(~repertoire_empty)/target_size) * 100

    # metrics for passive repertoires
    passive_repertoire_empty = passive_repertoire.fitnesses == -jnp.inf
    p_qd_score = jnp.sum(passive_repertoire.fitnesses, where=~passive_repertoire_empty)
    p_qd_score += qd_offset * jnp.sum(1.0 - passive_repertoire_empty)
    p_coverage = 100 * jnp.mean(1.0 - passive_repertoire_empty)
    p_max_fitness = jnp.max(passive_repertoire.fitnesses)
    p_mean_fitness = jnp.mean(
        passive_repertoire.fitnesses, where=~passive_repertoire_empty
    )

    observations = extra_scores["transitions"].obs
    min_obs = jnp.min(observations, axis=(0, 1))
    max_obs = jnp.max(observations, axis=(0, 1))

    return {
        "qd_score": qd_score,
        "max_fitness": max_fitness,
        "coverage": coverage,
        "passive_qd_score": p_qd_score,
        "passive_max_fitness": p_max_fitness,
        "passive_coverage": p_coverage,
        "passive_mean_fitness": p_mean_fitness,
        "min_obs": min_obs,
        "max_obs": max_obs,
    }
