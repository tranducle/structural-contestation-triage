"""Statistics for the v3 instruction-model study.

The SALAD sample has four strata of equal size drawn from a test split whose stratum sizes are
known. A population rate is therefore the population-weighted mean of the per-stratum sample
rates. Uncertainty comes from a stratified bootstrap that resamples alerts within each stratum.
One resample matrix is shared by every policy and, when the alert set is the same, by every model,
so the bootstrap distribution of a difference between two policies is a paired one.

Model answers are treated as fixed (temperature 0, fixed seed; determinism is measured separately),
so the intervals describe alert-sampling uncertainty only.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import binomtest

Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> list[float] | None:
    """Wilson score interval for a binomial proportion; None when there are no trials."""
    if trials <= 0:
        return None
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def proportion(successes: int, trials: int) -> dict[str, object]:
    return {
        "successes": int(successes),
        "trials": int(trials),
        "rate": (successes / trials) if trials else None,
        "wilson95": wilson_interval(successes, trials),
    }


def paired_mcnemar(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    """Exact two-sided McNemar test on paired binary outcomes.

    first_only counts pairs where only the first vector is 1, second_only the reverse.
    """
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"paired vectors differ in shape: {a.shape} vs {b.shape}")
    first_only = int(np.sum(a & ~b))
    second_only = int(np.sum(~a & b))
    discordant = first_only + second_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(min(first_only, second_only), discordant, p=0.5, alternative="two-sided").pvalue
    )
    return {
        "n_pairs": int(a.size),
        "first_only": first_only,
        "second_only": second_only,
        "discordant": discordant,
        "exact_p": p_value,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjustment; returns adjusted p-values in the input order."""
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(np.argsort(values, kind="mergesort")):
        running = max(running, min(1.0, float(values[index]) * (len(values) - rank)))
        adjusted[index] = running
    return adjusted.tolist()


class StratifiedDesign:
    """Alerts in a fixed order, their strata, and the population size of each stratum."""

    def __init__(self, strata: Sequence[str], population: Mapping[str, int]) -> None:
        self.strata = np.asarray(list(strata), dtype=object)
        self.population = {name: int(size) for name, size in population.items()}
        missing = set(self.strata.tolist()) - set(self.population)
        if missing:
            raise ValueError(f"strata without a population size: {sorted(missing)}")
        self.sample_counts = {name: int(np.sum(self.strata == name)) for name in self.population}
        self.weights = np.array(
            [self.population[name] / self.sample_counts[name] for name in self.strata],
            dtype=np.float64,
        )

    def mask(self, stratum_names: Sequence[str]) -> np.ndarray:
        return np.isin(self.strata, list(stratum_names))

    def weighted_rate(self, decisions: np.ndarray, stratum_names: Sequence[str]) -> float:
        """Population-weighted mean of decisions over the named strata."""
        weights = self.weights * self.mask(stratum_names)
        if weights.sum() == 0.0:
            return float("nan")
        return float(np.dot(weights, np.asarray(decisions, dtype=np.float64)) / weights.sum())

    def resample_counts(self, resamples: int, seed: int) -> np.ndarray:
        """Multiplicity matrix (resamples x alerts) of a within-stratum bootstrap."""
        rng = np.random.default_rng(seed)
        counts = np.zeros((resamples, self.strata.size), dtype=np.int32)
        for name in sorted(self.population):
            members = np.flatnonzero(self.strata == name)
            if members.size == 0:
                continue
            draws = rng.integers(0, members.size, size=(resamples, members.size))
            for position, member in enumerate(members):
                counts[:, member] = np.sum(draws == position, axis=1)
        return counts

    def bootstrap_rates(
        self, counts: np.ndarray, decisions: np.ndarray, stratum_names: Sequence[str]
    ) -> np.ndarray:
        """Weighted rate under every bootstrap resample; decisions has shape (alerts,)."""
        weighted = counts * (self.weights * self.mask(stratum_names))[None, :]
        return (weighted @ np.asarray(decisions, dtype=np.float64)) / weighted.sum(axis=1)


def percentile_interval(samples: np.ndarray) -> list[float]:
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def rate_interval(samples: np.ndarray) -> list[float]:
    """Percentile interval of a rate, kept in [0, 1].

    A resample in which every weighted decision is 1 sums numerator and denominator in different orders, so
    its rate can exceed 1 by one ulp. Only the reported bounds are clipped; the samples are left as they are.
    """
    return [min(max(bound, 0.0), 1.0) for bound in percentile_interval(samples)]


def expected_loss(
    recall: np.ndarray | float, fpr: np.ndarray | float,
    prevalence: np.ndarray | float, cost_ratio: np.ndarray | float,
) -> np.ndarray:
    """Expected cost per alert in units of one analyst review.

    Every escalated alert costs one review; every missed hostile alert costs cost_ratio reviews:
    loss = prevalence * (1 - recall) * cost_ratio + prevalence * recall + (1 - prevalence) * fpr.
    """
    recall = np.asarray(recall, dtype=np.float64)
    fpr = np.asarray(fpr, dtype=np.float64)
    prevalence = np.asarray(prevalence, dtype=np.float64)
    cost_ratio = np.asarray(cost_ratio, dtype=np.float64)
    return prevalence * (1.0 - recall) * cost_ratio + prevalence * recall + (1.0 - prevalence) * fpr
