from dataclasses import dataclass
from itertools import product
import math
import random

from .model import (
    CostDistribution,
    MonteCarloMetricSummary,
    Regime,
    Scenario,
    Stability,
    _validate_cost_distribution,
)

OBSERVED_TYPES = tuple(product((0, 1), (0, 1)))
TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class TwoCriterionScenario:
    benefit: float
    university_quota: float
    diversity_weight: float
    underlying_share_00: float
    underlying_share_01: float
    underlying_share_10: float
    underlying_share_11: float


@dataclass(frozen=True)
class TwoCriterionOutcomes:
    tutoring_threshold_0: float
    tutoring_threshold_1: float
    tutoring_participation_rate_0: float
    tutoring_participation_rate_1: float
    population_tutoring_mass: float
    admission_probability_00: float
    admission_probability_01: float
    admission_probability_10: float
    admission_probability_11: float
    credibility_0: float
    credibility_1: float
    posterior_evaluation_00: float
    posterior_evaluation_01: float
    posterior_evaluation_10: float
    posterior_evaluation_11: float
    admitted_high_ability_share: float
    admitted_diversity_share: float
    aggregate_tutoring_expenditure: float


@dataclass(frozen=True)
class TwoCriterionEquilibrium:
    regime: Regime
    stability: Stability
    outcomes: TwoCriterionOutcomes


@dataclass(frozen=True)
class TwoCriterionAnalysis:
    equilibria: tuple[TwoCriterionEquilibrium, ...]
    selected_equilibrium: TwoCriterionEquilibrium | None


@dataclass(frozen=True)
class TwoCriterionMonteCarloPopulationSummary:
    population_size: int
    trials: int
    regime_agreement_rate: float
    tutoring_threshold_0: MonteCarloMetricSummary
    tutoring_threshold_1: MonteCarloMetricSummary
    tutoring_participation_rate_0: MonteCarloMetricSummary
    tutoring_participation_rate_1: MonteCarloMetricSummary
    population_tutoring_mass: MonteCarloMetricSummary
    admission_probability_00: MonteCarloMetricSummary
    admission_probability_01: MonteCarloMetricSummary
    admission_probability_10: MonteCarloMetricSummary
    admission_probability_11: MonteCarloMetricSummary
    credibility_0: MonteCarloMetricSummary
    credibility_1: MonteCarloMetricSummary
    posterior_evaluation_00: MonteCarloMetricSummary
    posterior_evaluation_01: MonteCarloMetricSummary
    posterior_evaluation_10: MonteCarloMetricSummary
    posterior_evaluation_11: MonteCarloMetricSummary
    admitted_high_ability_share: MonteCarloMetricSummary
    admitted_diversity_share: MonteCarloMetricSummary
    aggregate_tutoring_expenditure: MonteCarloMetricSummary


@dataclass(frozen=True)
class TwoCriterionMonteCarloValidation:
    continuum_equilibrium: TwoCriterionEquilibrium
    population_summaries: tuple[TwoCriterionMonteCarloPopulationSummary, ...]


def _validate_two_criterion_scenario(scenario: TwoCriterionScenario) -> None:
    if scenario.benefit <= 0.0:
        raise ValueError("benefit must be positive")
    if not 0.0 < scenario.university_quota < 1.0:
        raise ValueError("university_quota must be strictly between 0 and 1")
    if not 0.0 <= scenario.diversity_weight <= 1.0:
        raise ValueError("diversity_weight must be between 0 and 1")
    shares = (
        scenario.underlying_share_00,
        scenario.underlying_share_01,
        scenario.underlying_share_10,
        scenario.underlying_share_11,
    )
    if any(not math.isfinite(share) or share < 0.0 for share in shares):
        raise ValueError("underlying shares must be finite and nonnegative")
    if not math.isclose(sum(shares), 1.0, abs_tol=1e-9):
        raise ValueError("underlying shares must sum to 1")


def _shares(scenario: TwoCriterionScenario) -> dict[tuple[int, int], float]:
    return {
        (0, 0): scenario.underlying_share_00,
        (0, 1): scenario.underlying_share_01,
        (1, 0): scenario.underlying_share_10,
        (1, 1): scenario.underlying_share_11,
    }


def _posterior_classes(
    posterior: dict[tuple[int, int], float],
) -> list[list[tuple[int, int]]]:
    remaining = list(OBSERVED_TYPES)
    classes: list[list[tuple[int, int]]] = []
    while remaining:
        peak = max(posterior[observed] for observed in remaining)
        group = [
            observed
            for observed in remaining
            if abs(posterior[observed] - peak) <= TIE_TOLERANCE
        ]
        remaining = [observed for observed in remaining if observed not in group]
        classes.append(group)
    return classes


def _is_better(
    left: tuple[int, int],
    right: tuple[int, int],
    posterior: dict[tuple[int, int], float],
) -> bool:
    return posterior[left] > posterior[right] + TIE_TOLERANCE


def _hat_candidate(
    types_in_class: set[tuple[int, int]],
    posterior: dict[tuple[int, int], float],
    benefit: float,
    tutoring_threshold_0: float,
    tutoring_threshold_1: float,
) -> dict[tuple[int, int], float]:
    thresholds = {0: tutoring_threshold_0, 1: tutoring_threshold_1}
    hat: dict[tuple[int, int], float] = {}
    for diversity in (0, 1):
        high = (1, diversity)
        low = (0, diversity)
        offset = thresholds[diversity] / benefit
        high_in = high in types_in_class
        low_in = low in types_in_class
        if high_in and low_in:
            return {}
        if high_in:
            low_probability = 0.0 if not _is_better(low, high, posterior) else 1.0
            hat[high] = min(1.0, max(0.0, low_probability + offset))
        if low_in:
            high_probability = 1.0 if _is_better(high, low, posterior) else 0.0
            hat[low] = min(1.0, max(0.0, high_probability - offset))
    return hat


def _rationalizing_split(
    rationing_class: list[tuple[int, int]],
    masses: dict[tuple[int, int], float],
    posterior: dict[tuple[int, int], float],
    residual: float,
    benefit: float,
    tutoring_threshold_0: float,
    tutoring_threshold_1: float,
) -> dict[tuple[int, int], float]:
    positive = [
        observed
        for observed in rationing_class
        if masses[observed] > TIE_TOLERANCE
    ]
    if len(positive) == 1:
        probability = residual / masses[positive[0]]
        return {observed: probability for observed in rationing_class}

    hat = _hat_candidate(
        set(rationing_class),
        posterior,
        benefit,
        tutoring_threshold_0,
        tutoring_threshold_1,
    )
    if not positive or any(observed not in hat for observed in positive):
        return {}
    budget = sum(masses[observed] * hat[observed] for observed in positive)
    if abs(budget - residual) > TIE_TOLERANCE:
        return {}
    representative = hat[positive[0]]
    return {
        observed: (hat[observed] if observed in positive else representative)
        for observed in rationing_class
    }


def _common_lottery(
    rationing_class: list[tuple[int, int]],
    masses: dict[tuple[int, int], float],
    residual: float,
) -> dict[tuple[int, int], float]:
    class_mass = sum(masses[observed] for observed in rationing_class)
    probability = residual / class_mass
    return {observed: probability for observed in rationing_class}


def _admission_probabilities(
    masses: dict[tuple[int, int], float],
    posterior: dict[tuple[int, int], float],
    quota: float,
    benefit: float,
    tutoring_threshold_0: float,
    tutoring_threshold_1: float,
) -> dict[tuple[int, int], float]:
    classes = _posterior_classes(posterior)
    assigned: dict[tuple[int, int], float] = {}
    residual = quota
    rationing_class: list[tuple[int, int]] | None = None
    rationing_residual = 0.0

    for index, group in enumerate(classes):
        class_mass = sum(masses[observed] for observed in group)
        if class_mass <= TIE_TOLERANCE:
            continue
        if residual <= TIE_TOLERANCE:
            for later in classes[index:]:
                for observed in later:
                    assigned.setdefault(observed, 0.0)
            break
        if residual + TIE_TOLERANCE < class_mass:
            rationing_class = group
            rationing_residual = residual
            for later in classes[index + 1 :]:
                for observed in later:
                    assigned.setdefault(observed, 0.0)
            break
        for observed in group:
            assigned[observed] = 1.0
        if abs(residual - class_mass) <= TIE_TOLERANCE:
            for later in classes[index + 1 :]:
                for observed in later:
                    assigned.setdefault(observed, 0.0)
            break
        residual -= class_mass

    if rationing_class is not None:
        split = _rationalizing_split(
            rationing_class,
            masses,
            posterior,
            rationing_residual,
            benefit,
            tutoring_threshold_0,
            tutoring_threshold_1,
        )
        if not split:
            split = _common_lottery(rationing_class, masses, rationing_residual)
        assigned.update(split)

    for observed in OBSERVED_TYPES:
        if observed in assigned:
            continue
        if any(
            assigned.get(other, 0.0) > TIE_TOLERANCE
            and _is_better(observed, other, posterior)
            for other in OBSERVED_TYPES
        ):
            assigned[observed] = 1.0
        else:
            assigned[observed] = 0.0

    return assigned


def one_stage_image(scenario: TwoCriterionScenario) -> Scenario:
    _validate_two_criterion_scenario(scenario)
    if not math.isclose(scenario.diversity_weight, 0.0, abs_tol=1e-9):
        raise ValueError("one-stage image requires diversity_weight = 0")
    if not math.isclose(
        scenario.underlying_share_10 * scenario.underlying_share_01,
        scenario.underlying_share_11 * scenario.underlying_share_00,
        abs_tol=1e-9,
    ):
        raise ValueError("one-stage image requires equal high-ability odds")
    return Scenario(
        benefit=scenario.benefit,
        high_ability_share=(
            scenario.underlying_share_10 + scenario.underlying_share_11
        ),
        university_quota=scenario.university_quota,
    )


def evaluate_threshold_pair(
    scenario: TwoCriterionScenario,
    cost_distribution: CostDistribution,
    tutoring_threshold_0: float,
    tutoring_threshold_1: float,
) -> TwoCriterionOutcomes:
    _validate_two_criterion_scenario(scenario)
    _validate_cost_distribution(cost_distribution, scenario.benefit)
    return _evaluate_threshold_pair_core(
        scenario,
        cost_distribution,
        tutoring_threshold_0,
        tutoring_threshold_1,
    )


def _evaluate_threshold_pair_core(
    scenario: TwoCriterionScenario,
    cost_distribution: CostDistribution,
    tutoring_threshold_0: float,
    tutoring_threshold_1: float,
) -> TwoCriterionOutcomes:
    masses, clipped_0, clipped_1, participation_0, participation_1 = _masses_at(
        scenario,
        cost_distribution,
        tutoring_threshold_0,
        tutoring_threshold_1,
    )
    shares = _shares(scenario)
    credibility_0 = (
        1.0 if masses[1, 0] <= TIE_TOLERANCE else shares[1, 0] / masses[1, 0]
    )
    credibility_1 = (
        1.0 if masses[1, 1] <= TIE_TOLERANCE else shares[1, 1] / masses[1, 1]
    )
    weight = scenario.diversity_weight
    posterior = {
        (0, 0): 0.0,
        (0, 1): weight,
        (1, 0): (1.0 - weight) * credibility_0,
        (1, 1): weight + (1.0 - weight) * credibility_1,
    }
    admission = _admission_probabilities(
        masses,
        posterior,
        scenario.university_quota,
        scenario.benefit,
        clipped_0,
        clipped_1,
    )
    quota = scenario.university_quota
    return TwoCriterionOutcomes(
        tutoring_threshold_0=tutoring_threshold_0,
        tutoring_threshold_1=tutoring_threshold_1,
        tutoring_participation_rate_0=participation_0,
        tutoring_participation_rate_1=participation_1,
        population_tutoring_mass=(
            shares[0, 0] * participation_0 + shares[0, 1] * participation_1
        ),
        admission_probability_00=admission[0, 0],
        admission_probability_01=admission[0, 1],
        admission_probability_10=admission[1, 0],
        admission_probability_11=admission[1, 1],
        credibility_0=credibility_0,
        credibility_1=credibility_1,
        posterior_evaluation_00=posterior[0, 0],
        posterior_evaluation_01=posterior[0, 1],
        posterior_evaluation_10=posterior[1, 0],
        posterior_evaluation_11=posterior[1, 1],
        admitted_high_ability_share=(
            (shares[1, 0] * admission[1, 0] + shares[1, 1] * admission[1, 1])
            / quota
        ),
        admitted_diversity_share=(
            (
                shares[1, 1] * admission[1, 1]
                + shares[0, 1] * participation_1 * admission[1, 1]
                + shares[0, 1] * (1.0 - participation_1) * admission[0, 1]
            )
            / quota
        ),
        aggregate_tutoring_expenditure=(
            shares[0, 0] * cost_distribution.partial_first_moment(clipped_0)
            + shares[0, 1] * cost_distribution.partial_first_moment(clipped_1)
        ),
    )


def _best_response(
    scenario: TwoCriterionScenario,
    cost_distribution: CostDistribution,
    tutoring_threshold_0: float,
    tutoring_threshold_1: float,
) -> tuple[float, float, TwoCriterionOutcomes]:
    outcomes = _evaluate_threshold_pair_core(
        scenario,
        cost_distribution,
        tutoring_threshold_0,
        tutoring_threshold_1,
    )
    benefit = scenario.benefit
    tutoring_benefit_0 = benefit * (
        outcomes.admission_probability_10 - outcomes.admission_probability_00
    )
    tutoring_benefit_1 = benefit * (
        outcomes.admission_probability_11 - outcomes.admission_probability_01
    )
    return (
        min(benefit, max(0.0, tutoring_benefit_0)),
        min(benefit, max(0.0, tutoring_benefit_1)),
        outcomes,
    )


def _regime_from_outcomes(outcomes: TwoCriterionOutcomes) -> Regime:
    leftover_academic = [
        academic
        for academic, probability in (
            (1, outcomes.admission_probability_10),
            (1, outcomes.admission_probability_11),
            (0, outcomes.admission_probability_00),
            (0, outcomes.admission_probability_01),
        )
        if TIE_TOLERANCE < probability < 1.0 - TIE_TOLERANCE
    ]
    if leftover_academic:
        if all(academic == 1 for academic in leftover_academic):
            return Regime.TIGHT
        if all(academic == 0 for academic in leftover_academic):
            return Regime.LOOSE
        return Regime.TIGHT
    if (
        outcomes.admission_probability_00 <= TIE_TOLERANCE
        and outcomes.admission_probability_01 <= TIE_TOLERANCE
    ):
        return Regime.TIGHT
    return Regime.LOOSE


def _hat_x_on_class(
    rationing_class: list[tuple[int, int]],
    posterior: dict[tuple[int, int], float],
    benefit: float,
    tutoring_threshold_0: float,
    tutoring_threshold_1: float,
) -> dict[tuple[int, int], float]:
    return _hat_candidate(
        set(rationing_class),
        posterior,
        benefit,
        tutoring_threshold_0,
        tutoring_threshold_1,
    )


def _rationing_class_and_residual(
    masses: dict[tuple[int, int], float],
    posterior: dict[tuple[int, int], float],
    quota: float,
) -> tuple[list[tuple[int, int]] | None, float]:
    residual = quota
    for group in _posterior_classes(posterior):
        class_mass = sum(masses[observed] for observed in group)
        if class_mass <= TIE_TOLERANCE:
            continue
        if residual <= TIE_TOLERANCE:
            return None, 0.0
        if residual + TIE_TOLERANCE < class_mass:
            return group, residual
        if abs(residual - class_mass) <= TIE_TOLERANCE:
            return None, 0.0
        residual -= class_mass
    return None, 0.0


def _masses_at(
    scenario: TwoCriterionScenario,
    cost_distribution: CostDistribution,
    tutoring_threshold_0: float,
    tutoring_threshold_1: float,
) -> tuple[dict[tuple[int, int], float], float, float, float, float]:
    clipped_0 = min(scenario.benefit, max(0.0, tutoring_threshold_0))
    clipped_1 = min(scenario.benefit, max(0.0, tutoring_threshold_1))
    participation_0 = cost_distribution.cdf(clipped_0)
    participation_1 = cost_distribution.cdf(clipped_1)
    shares = _shares(scenario)
    masses = {
        (1, 0): shares[1, 0] + shares[0, 0] * participation_0,
        (1, 1): shares[1, 1] + shares[0, 1] * participation_1,
        (0, 0): shares[0, 0] * (1.0 - participation_0),
        (0, 1): shares[0, 1] * (1.0 - participation_1),
    }
    return masses, clipped_0, clipped_1, participation_0, participation_1


def _quota_gap(
    scenario: TwoCriterionScenario,
    cost_distribution: CostDistribution,
    tutoring_threshold_0: float,
    tutoring_threshold_1: float,
) -> float:
    tutoring_benefit_0, tutoring_benefit_1, outcomes = _best_response(
        scenario,
        cost_distribution,
        tutoring_threshold_0,
        tutoring_threshold_1,
    )
    masses, clipped_0, clipped_1, _, _ = _masses_at(
        scenario,
        cost_distribution,
        tutoring_threshold_0,
        tutoring_threshold_1,
    )
    posterior = {
        (0, 0): outcomes.posterior_evaluation_00,
        (0, 1): outcomes.posterior_evaluation_01,
        (1, 0): outcomes.posterior_evaluation_10,
        (1, 1): outcomes.posterior_evaluation_11,
    }
    rationing_class, residual = _rationing_class_and_residual(
        masses, posterior, scenario.university_quota
    )
    if rationing_class is None:
        return (tutoring_benefit_0 - tutoring_threshold_0) + (
            tutoring_benefit_1 - tutoring_threshold_1
        )
    hat = _hat_x_on_class(
        rationing_class,
        posterior,
        scenario.benefit,
        clipped_0,
        clipped_1,
    )
    if not hat:
        return (tutoring_benefit_0 - tutoring_threshold_0) + (
            tutoring_benefit_1 - tutoring_threshold_1
        )
    budget = sum(
        masses[observed] * hat[observed]
        for observed in rationing_class
        if masses[observed] > TIE_TOLERANCE
    )
    return residual - budget


def _inverse_cdf(
    cost_distribution: CostDistribution, probability: float, benefit: float
) -> float:
    if probability <= TIE_TOLERANCE:
        return 0.0
    if probability >= 1.0 - TIE_TOLERANCE:
        return benefit
    lower = 0.0
    upper = benefit
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if cost_distribution.cdf(midpoint) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _mixed_partner(
    scenario: TwoCriterionScenario,
    cost_distribution: CostDistribution,
    tutoring_threshold_0: float,
) -> float:
    weight = scenario.diversity_weight
    if weight >= 1.0 - TIE_TOLERANCE:
        return math.nan
    offset = weight / (1.0 - weight)
    shares = _shares(scenario)
    clipped_0 = min(scenario.benefit, max(0.0, tutoring_threshold_0))
    participation_0 = cost_distribution.cdf(clipped_0)
    mass_10 = shares[1, 0] + shares[0, 0] * participation_0
    credibility_0 = 1.0 if mass_10 <= TIE_TOLERANCE else shares[1, 0] / mass_10
    credibility_1 = credibility_0 - offset
    if credibility_1 <= TIE_TOLERANCE or credibility_1 > 1.0 + TIE_TOLERANCE:
        return math.nan
    if shares[0, 1] <= TIE_TOLERANCE:
        if abs(credibility_1 - 1.0) <= TIE_TOLERANCE:
            return 0.0
        return math.nan
    participation_1 = (
        shares[1, 1] * (1.0 / credibility_1 - 1.0) / shares[0, 1]
    )
    if participation_1 < -TIE_TOLERANCE or participation_1 > 1.0 + TIE_TOLERANCE:
        return math.nan
    return _inverse_cdf(
        cost_distribution,
        min(1.0, max(0.0, participation_1)),
        scenario.benefit,
    )


def _mu10_equals_mu01_threshold(
    scenario: TwoCriterionScenario, cost_distribution: CostDistribution
) -> float:
    weight = scenario.diversity_weight
    if weight <= TIE_TOLERANCE or weight >= 1.0 - TIE_TOLERANCE:
        return math.nan
    target = weight / (1.0 - weight)
    if target > 1.0 + TIE_TOLERANCE:
        return math.nan
    shares = _shares(scenario)
    if shares[0, 0] <= TIE_TOLERANCE:
        return math.nan
    if target <= TIE_TOLERANCE:
        return math.nan
    participation = shares[1, 0] * (1.0 / target - 1.0) / shares[0, 0]
    if participation < -TIE_TOLERANCE or participation > 1.0 + TIE_TOLERANCE:
        return math.nan
    return _inverse_cdf(
        cost_distribution,
        min(1.0, max(0.0, participation)),
        scenario.benefit,
    )


def _find_1d_roots(residual, lower: float, upper: float) -> list[float]:
    if abs(upper - lower) <= 1e-15:
        value = residual(lower)
        if math.isfinite(value) and abs(value) < 1e-10:
            return [lower]
        return []

    def bisect_root(left: float, right: float) -> float:
        left_value = residual(left)
        for _ in range(80):
            midpoint = (left + right) / 2.0
            midpoint_value = residual(midpoint)
            if not math.isfinite(left_value) or not math.isfinite(midpoint_value):
                return midpoint
            if (left_value > 0.0) == (midpoint_value > 0.0):
                left = midpoint
                left_value = midpoint_value
            else:
                right = midpoint
        return (left + right) / 2.0

    def minimize_absolute_excess(left: float, right: float) -> float:
        inverse_phi = (5.0**0.5 - 1.0) / 2.0
        left_probe = right - inverse_phi * (right - left)
        right_probe = left + inverse_phi * (right - left)
        left_value = abs(residual(left_probe))
        right_value = abs(residual(right_probe))
        for _ in range(80):
            if not math.isfinite(left_value):
                left_value = float("inf")
            if not math.isfinite(right_value):
                right_value = float("inf")
            if left_value <= right_value:
                right = right_probe
                right_probe = left_probe
                right_value = left_value
                left_probe = right - inverse_phi * (right - left)
                left_value = abs(residual(left_probe))
            else:
                left = left_probe
                left_probe = right_probe
                left_value = right_value
                right_probe = left + inverse_phi * (right - left)
                right_value = abs(residual(right_probe))
        return (left + right) / 2.0

    grid_size = 8192
    roots: list[float] = []

    def point(index: int) -> float:
        return lower + (upper - lower) * index / grid_size

    previous_previous = point(0)
    previous_previous_value = residual(previous_previous)
    previous = point(1)
    previous_value = residual(previous)
    if math.isfinite(previous_previous_value) and abs(previous_previous_value) < 1e-10:
        roots.append(previous_previous)
    if (
        math.isfinite(previous_previous_value)
        and math.isfinite(previous_value)
        and previous_previous_value * previous_value < 0.0
    ):
        roots.append(bisect_root(previous_previous, previous))

    for index in range(2, grid_size + 1):
        cutoff = point(index)
        value = residual(cutoff)
        if (
            math.isfinite(previous_value)
            and math.isfinite(previous_previous_value)
            and math.isfinite(value)
            and abs(previous_value) <= abs(previous_previous_value)
            and abs(previous_value) <= abs(value)
        ):
            candidate = minimize_absolute_excess(previous_previous, cutoff)
            if abs(residual(candidate)) < 1e-9:
                roots.append(candidate)
        if (
            math.isfinite(previous_value)
            and math.isfinite(value)
            and previous_value * value < 0.0
        ):
            roots.append(bisect_root(previous, cutoff))
        previous_previous = previous
        previous_previous_value = previous_value
        previous = cutoff
        previous_value = value
    if math.isfinite(previous_value) and abs(previous_value) < 1e-10:
        roots.append(upper)

    distinct_roots: list[float] = []
    for root in sorted(roots):
        if not distinct_roots or abs(root - distinct_roots[-1]) > 1e-7:
            distinct_roots.append(root)
    return distinct_roots


def _classify_1d(residual, parameter: float, lower: float, upper: float, benefit: float) -> Stability:
    step = max(benefit * 1e-5, 1e-8)
    direction_tolerance = 1e-8

    def probe(candidate: float) -> float | None:
        if candidate < lower - 1e-15 or candidate > upper + 1e-15:
            return None
        value = residual(candidate)
        if not math.isfinite(value):
            return None
        return value

    lower_excess = probe(max(lower, parameter - step))
    upper_excess = probe(min(upper, parameter + step))
    if lower_excess is None or upper_excess is None:
        return Stability.NEUTRAL
    if lower_excess > direction_tolerance and upper_excess < -direction_tolerance:
        return Stability.STABLE
    if lower_excess < -direction_tolerance and upper_excess > direction_tolerance:
        return Stability.UNSTABLE
    return Stability.NEUTRAL


def _ranking_signature(outcomes: TwoCriterionOutcomes) -> tuple[tuple[tuple[int, int], ...], ...]:
    posterior = {
        (0, 0): outcomes.posterior_evaluation_00,
        (0, 1): outcomes.posterior_evaluation_01,
        (1, 0): outcomes.posterior_evaluation_10,
        (1, 1): outcomes.posterior_evaluation_11,
    }
    return tuple(tuple(group) for group in _posterior_classes(posterior))


def _classify_jacobian(
    scenario: TwoCriterionScenario,
    cost_distribution: CostDistribution,
    tutoring_threshold_0: float,
    tutoring_threshold_1: float,
) -> Stability:
    benefit = scenario.benefit
    delta = max(benefit * 1e-5, 1e-8)
    _, _, root_outcomes = _best_response(
        scenario, cost_distribution, tutoring_threshold_0, tutoring_threshold_1
    )
    root_signature = _ranking_signature(root_outcomes)

    def excess_at(threshold_0: float, threshold_1: float) -> tuple[float, float] | None:
        clipped_0 = min(benefit, max(0.0, threshold_0))
        clipped_1 = min(benefit, max(0.0, threshold_1))
        tutoring_benefit_0, tutoring_benefit_1, outcomes = _best_response(
            scenario, cost_distribution, clipped_0, clipped_1
        )
        if _ranking_signature(outcomes) != root_signature:
            return None
        return tutoring_benefit_0 - clipped_0, tutoring_benefit_1 - clipped_1

    def derivative(axis: int) -> tuple[float, float] | None:
        forward_0 = tutoring_threshold_0 + (delta if axis == 0 else 0.0)
        forward_1 = tutoring_threshold_1 + (delta if axis == 1 else 0.0)
        backward_0 = tutoring_threshold_0 - (delta if axis == 0 else 0.0)
        backward_1 = tutoring_threshold_1 - (delta if axis == 1 else 0.0)
        forward = excess_at(forward_0, forward_1)
        backward = excess_at(backward_0, backward_1)
        center = excess_at(tutoring_threshold_0, tutoring_threshold_1)
        if center is None:
            return None
        if forward is not None and backward is not None:
            span = 2.0 * delta
            return (forward[0] - backward[0]) / span, (forward[1] - backward[1]) / span
        if forward is not None:
            return (forward[0] - center[0]) / delta, (forward[1] - center[1]) / delta
        if backward is not None:
            return (center[0] - backward[0]) / delta, (center[1] - backward[1]) / delta
        return None

    column_0 = derivative(0)
    column_1 = derivative(1)
    if column_0 is None or column_1 is None:
        return Stability.NEUTRAL
    trace = column_0[0] + column_1[1]
    determinant = column_0[0] * column_1[1] - column_0[1] * column_1[0]
    if abs(determinant) < 1e-8:
        return Stability.NEUTRAL
    if trace < 0.0 and determinant > 0.0:
        return Stability.STABLE
    return Stability.UNSTABLE


def _equal_high_ability_odds(scenario: TwoCriterionScenario) -> bool:
    return math.isclose(
        scenario.underlying_share_10 * scenario.underlying_share_01,
        scenario.underlying_share_11 * scenario.underlying_share_00,
        abs_tol=1e-9,
    )


def _square_is_continuum(
    scenario: TwoCriterionScenario, cost_distribution: CostDistribution
) -> bool:
    benefit = scenario.benefit
    hits = 0
    count = 0
    for index_0 in range(1, 6):
        for index_1 in range(1, 6):
            threshold_0 = benefit * index_0 / 6.0
            threshold_1 = benefit * index_1 / 6.0
            tutoring_benefit_0, tutoring_benefit_1, _ = _best_response(
                scenario, cost_distribution, threshold_0, threshold_1
            )
            count += 1
            if (
                math.hypot(
                    tutoring_benefit_0 - threshold_0,
                    tutoring_benefit_1 - threshold_1,
                )
                < 1e-6
            ):
                hits += 1
    return hits > 0.4 * count


def _append_fixed_point(
    points: list[tuple[float, float]],
    scenario: TwoCriterionScenario,
    cost_distribution: CostDistribution,
    tutoring_threshold_0: float,
    tutoring_threshold_1: float,
) -> None:
    tutoring_benefit_0, tutoring_benefit_1, _ = _best_response(
        scenario,
        cost_distribution,
        tutoring_threshold_0,
        tutoring_threshold_1,
    )
    if (
        math.hypot(
            tutoring_benefit_0 - tutoring_threshold_0,
            tutoring_benefit_1 - tutoring_threshold_1,
        )
        < 1e-7
    ):
        points.append((tutoring_threshold_0, tutoring_threshold_1))


def analyze_two_criterion_scenario(
    scenario: TwoCriterionScenario,
    cost_distribution: CostDistribution,
) -> TwoCriterionAnalysis:
    _validate_two_criterion_scenario(scenario)
    _validate_cost_distribution(cost_distribution, scenario.benefit)
    benefit = scenario.benefit
    points: list[tuple[float, float]] = []

    def mixed_residual(threshold_0: float) -> float:
        threshold_1 = _mixed_partner(scenario, cost_distribution, threshold_0)
        if not math.isfinite(threshold_1):
            return math.nan
        return _quota_gap(scenario, cost_distribution, threshold_0, threshold_1)

    for threshold_0 in _find_1d_roots(mixed_residual, 0.0, benefit):
        threshold_1 = _mixed_partner(scenario, cost_distribution, threshold_0)
        if math.isfinite(threshold_1):
            _append_fixed_point(
                points, scenario, cost_distribution, threshold_0, threshold_1
            )

    if _equal_high_ability_odds(scenario) and math.isclose(
        scenario.diversity_weight, 0.0, abs_tol=1e-9
    ):

        def diagonal_residual(cutoff: float) -> float:
            tutoring_benefit_0, _, _ = _best_response(
                scenario, cost_distribution, cutoff, cutoff
            )
            return tutoring_benefit_0 - cutoff

        for cutoff in _find_1d_roots(diagonal_residual, 0.0, benefit):
            _append_fixed_point(points, scenario, cost_distribution, cutoff, cutoff)

    vertical = _mu10_equals_mu01_threshold(scenario, cost_distribution)
    if math.isfinite(vertical):

        def vertical_residual(threshold_1: float) -> float:
            return _quota_gap(
                scenario, cost_distribution, vertical, threshold_1
            )

        for threshold_1 in _find_1d_roots(vertical_residual, 0.0, benefit):
            _append_fixed_point(
                points, scenario, cost_distribution, vertical, threshold_1
            )

    def edge_residual_1(fixed_0: float):
        def residual(threshold_1: float) -> float:
            _, tutoring_benefit_1, _ = _best_response(
                scenario, cost_distribution, fixed_0, threshold_1
            )
            return tutoring_benefit_1 - threshold_1

        return residual

    def edge_residual_0(fixed_1: float):
        def residual(threshold_0: float) -> float:
            tutoring_benefit_0, _, _ = _best_response(
                scenario, cost_distribution, threshold_0, fixed_1
            )
            return tutoring_benefit_0 - threshold_0

        return residual

    endpoint_tolerance = 1e-10
    for fixed_0 in (0.0, benefit):
        for threshold_1 in _find_1d_roots(edge_residual_1(fixed_0), 0.0, benefit):
            tutoring_benefit_0, tutoring_benefit_1, _ = _best_response(
                scenario, cost_distribution, fixed_0, threshold_1
            )
            if abs(tutoring_benefit_0 - fixed_0) < endpoint_tolerance:
                _append_fixed_point(
                    points, scenario, cost_distribution, fixed_0, threshold_1
                )
    for fixed_1 in (0.0, benefit):
        for threshold_0 in _find_1d_roots(edge_residual_0(fixed_1), 0.0, benefit):
            tutoring_benefit_0, tutoring_benefit_1, _ = _best_response(
                scenario, cost_distribution, threshold_0, fixed_1
            )
            if abs(tutoring_benefit_1 - fixed_1) < endpoint_tolerance:
                _append_fixed_point(
                    points, scenario, cost_distribution, threshold_0, fixed_1
                )

    for corner_0, corner_1 in ((0.0, 0.0), (0.0, benefit), (benefit, 0.0), (benefit, benefit)):
        _append_fixed_point(points, scenario, cost_distribution, corner_0, corner_1)

    if _square_is_continuum(scenario, cost_distribution):
        points = [
            (threshold_0, threshold_1)
            for threshold_0, threshold_1 in points
            if abs(threshold_0 - threshold_1) <= 1e-7
        ]

    points.sort()
    distinct_points: list[tuple[float, float]] = []
    for point in points:
        if not distinct_points or math.hypot(
            point[0] - distinct_points[-1][0],
            point[1] - distinct_points[-1][1],
        ) > 1e-7:
            distinct_points.append(point)

    nested = math.isclose(scenario.diversity_weight, 0.0, abs_tol=1e-9) and (
        _equal_high_ability_odds(scenario)
    )

    def classify(threshold_0: float, threshold_1: float) -> Stability:
        if nested and abs(threshold_0 - threshold_1) <= 1e-7:
            cutoff = (threshold_0 + threshold_1) / 2.0

            def diagonal_residual(value: float) -> float:
                tutoring_benefit_0, _, _ = _best_response(
                    scenario, cost_distribution, value, value
                )
                return tutoring_benefit_0 - value

            return _classify_1d(diagonal_residual, cutoff, 0.0, benefit, benefit)
        partner = _mixed_partner(scenario, cost_distribution, threshold_0)
        if math.isfinite(partner) and abs(partner - threshold_1) <= 1e-6:
            return _classify_1d(mixed_residual, threshold_0, 0.0, benefit, benefit)
        if abs(threshold_0) <= 1e-9 or abs(threshold_0 - benefit) <= 1e-9:
            return _classify_1d(
                edge_residual_1(threshold_0), threshold_1, 0.0, benefit, benefit
            )
        if abs(threshold_1) <= 1e-9 or abs(threshold_1 - benefit) <= 1e-9:
            return _classify_1d(
                edge_residual_0(threshold_1), threshold_0, 0.0, benefit, benefit
            )
        return _classify_jacobian(
            scenario, cost_distribution, threshold_0, threshold_1
        )

    equilibria = []
    for threshold_0, threshold_1 in distinct_points:
        _, _, outcomes = _best_response(
            scenario, cost_distribution, threshold_0, threshold_1
        )
        equilibria.append(
            TwoCriterionEquilibrium(
                regime=_regime_from_outcomes(outcomes),
                stability=classify(threshold_0, threshold_1),
                outcomes=outcomes,
            )
        )
    selected_equilibrium = max(
        (
            equilibrium
            for equilibrium in equilibria
            if equilibrium.stability is Stability.STABLE
        ),
        key=lambda equilibrium: equilibrium.outcomes.population_tutoring_mass,
        default=None,
    )
    return TwoCriterionAnalysis(
        equilibria=tuple(equilibria),
        selected_equilibrium=selected_equilibrium,
    )


def _exact_count(share: float, population_size: int, name: str) -> int:
    count = share * population_size
    rounded_count = round(count)
    if not math.isclose(count, rounded_count, abs_tol=1e-9):
        raise ValueError(f"population size must make {name} an integer count")
    return rounded_count


def _sample_cost(
    cost_distribution: CostDistribution, benefit: float, generator: random.Random
) -> float:
    probability = generator.random()
    lower = 0.0
    upper = benefit
    for _ in range(48):
        midpoint = (lower + upper) / 2.0
        if cost_distribution.cdf(midpoint) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _summarize_metric(values: list[float], target: float) -> MonteCarloMetricSummary:
    mean = math.fsum(values) / len(values)
    squared_errors = [(value - target) ** 2 for value in values]
    if len(values) == 1:
        standard_error = 0.0
    else:
        sample_variance = math.fsum((value - mean) ** 2 for value in values) / (
            len(values) - 1
        )
        standard_error = math.sqrt(sample_variance / len(values))
    return MonteCarloMetricSummary(
        mean=mean,
        bias=mean - target,
        root_mean_squared_error=math.sqrt(math.fsum(squared_errors) / len(values)),
        standard_error=standard_error,
    )


def _largest_remainder_targets(
    types: list[tuple[int, int]],
    probabilities: dict[tuple[int, int], float],
    counts: dict[tuple[int, int], int],
    seats: int,
) -> dict[tuple[int, int], int]:
    positive = [observed for observed in types if counts.get(observed, 0) > 0]
    available = sum(counts[observed] for observed in positive)
    if available <= seats:
        return {observed: counts.get(observed, 0) for observed in types}
    raw = {
        observed: probabilities[observed] * counts[observed] for observed in positive
    }
    total_raw = sum(raw.values())
    if total_raw <= TIE_TOLERANCE:
        raw = {observed: float(counts[observed]) for observed in positive}
        total_raw = sum(raw.values())
    scaled = {observed: raw[observed] / total_raw * seats for observed in raw}
    floors = {
        observed: min(counts[observed], math.floor(scaled[observed]))
        for observed in scaled
    }
    leftover = seats - sum(floors.values())
    order = sorted(
        scaled,
        key=lambda observed: (
            scaled[observed] - math.floor(scaled[observed]),
            observed[0],
            observed[1],
        ),
        reverse=True,
    )
    targets = {observed: int(floors[observed]) for observed in scaled}
    guard = 0
    while leftover > 0 and guard < seats + len(order):
        observed = order[guard % len(order)]
        if targets[observed] < counts[observed]:
            targets[observed] += 1
            leftover -= 1
        guard += 1
    return {observed: targets.get(observed, 0) for observed in types}


def _continuum_probabilities(
    outcomes: TwoCriterionOutcomes,
) -> dict[tuple[int, int], float]:
    return {
        (0, 0): outcomes.admission_probability_00,
        (0, 1): outcomes.admission_probability_01,
        (1, 0): outcomes.admission_probability_10,
        (1, 1): outcomes.admission_probability_11,
    }


def _continuum_posterior(
    outcomes: TwoCriterionOutcomes,
) -> dict[tuple[int, int], float]:
    return {
        (0, 0): outcomes.posterior_evaluation_00,
        (0, 1): outcomes.posterior_evaluation_01,
        (1, 0): outcomes.posterior_evaluation_10,
        (1, 1): outcomes.posterior_evaluation_11,
    }


def _admit_applicants(
    applicants: list[tuple[int, int, int, float | None]],
    outcomes: TwoCriterionOutcomes,
    seats: int,
    generator: random.Random,
) -> set[int]:
    probabilities = _continuum_probabilities(outcomes)
    classes = _posterior_classes(_continuum_posterior(outcomes))
    remaining = seats
    admitted: set[int] = set()

    def members_of(group: list[tuple[int, int]]) -> list[int]:
        types_in_group = set(group)
        return [
            index
            for index, (_, diversity, observed_academic, _) in enumerate(applicants)
            if (observed_academic, diversity) in types_in_group and index not in admitted
        ]

    def take(indices: list[int], count: int) -> None:
        nonlocal remaining
        if count <= 0 or not indices or remaining <= 0:
            return
        chosen_count = min(count, len(indices), remaining)
        chosen = (
            generator.sample(indices, chosen_count)
            if chosen_count < len(indices)
            else indices
        )
        admitted.update(chosen)
        remaining -= len(chosen)

    for group in classes:
        if remaining <= 0:
            break
        members = members_of(group)
        if not members:
            continue
        group_probabilities = [probabilities[observed] for observed in group]
        peak = max(group_probabilities)
        floor_probability = min(group_probabilities)
        if floor_probability >= 1.0 - TIE_TOLERANCE:
            take(members, remaining)
        elif peak <= TIE_TOLERANCE:
            continue
        elif peak - floor_probability <= TIE_TOLERANCE:
            take(members, remaining)
        else:
            counts = {
                observed: sum(
                    1
                    for index in members
                    if (applicants[index][2], applicants[index][1]) == observed
                )
                for observed in group
            }
            targets = _largest_remainder_targets(
                group, probabilities, counts, remaining
            )
            for observed, target in targets.items():
                pool = [
                    index
                    for index in members
                    if (applicants[index][2], applicants[index][1]) == observed
                ]
                take(pool, target)

    if remaining > 0:
        for group in classes:
            if remaining <= 0:
                break
            take(members_of(group), remaining)
    return admitted


def run_two_criterion_monte_carlo_validation(
    scenario: TwoCriterionScenario,
    cost_distribution: CostDistribution,
    population_sizes: tuple[int, ...],
    trials: int,
    seed: int,
) -> TwoCriterionMonteCarloValidation:
    if not population_sizes:
        raise ValueError("at least one population size is required")
    continuum_analysis = analyze_two_criterion_scenario(
        scenario, cost_distribution
    )
    continuum_equilibrium = continuum_analysis.selected_equilibrium
    if continuum_equilibrium is None:
        raise ValueError("scenario must have a selected stable equilibrium")
    if not isinstance(trials, int) or isinstance(trials, bool) or trials <= 0:
        raise ValueError("trials must be a positive integer")

    outcomes = continuum_equilibrium.outcomes
    targets = {
        "tutoring_threshold_0": outcomes.tutoring_threshold_0,
        "tutoring_threshold_1": outcomes.tutoring_threshold_1,
        "tutoring_participation_rate_0": outcomes.tutoring_participation_rate_0,
        "tutoring_participation_rate_1": outcomes.tutoring_participation_rate_1,
        "population_tutoring_mass": outcomes.population_tutoring_mass,
        "admission_probability_00": outcomes.admission_probability_00,
        "admission_probability_01": outcomes.admission_probability_01,
        "admission_probability_10": outcomes.admission_probability_10,
        "admission_probability_11": outcomes.admission_probability_11,
        "credibility_0": outcomes.credibility_0,
        "credibility_1": outcomes.credibility_1,
        "posterior_evaluation_00": outcomes.posterior_evaluation_00,
        "posterior_evaluation_01": outcomes.posterior_evaluation_01,
        "posterior_evaluation_10": outcomes.posterior_evaluation_10,
        "posterior_evaluation_11": outcomes.posterior_evaluation_11,
        "admitted_high_ability_share": outcomes.admitted_high_ability_share,
        "admitted_diversity_share": outcomes.admitted_diversity_share,
        "aggregate_tutoring_expenditure": outcomes.aggregate_tutoring_expenditure,
    }
    generator = random.Random(seed)
    population_summaries: list[TwoCriterionMonteCarloPopulationSummary] = []

    for population_size in population_sizes:
        if (
            not isinstance(population_size, int)
            or isinstance(population_size, bool)
            or population_size <= 0
        ):
            raise ValueError("population sizes must be positive integers")
        count_00 = _exact_count(
            scenario.underlying_share_00, population_size, "underlying share 00"
        )
        count_01 = _exact_count(
            scenario.underlying_share_01, population_size, "underlying share 01"
        )
        count_10 = _exact_count(
            scenario.underlying_share_10, population_size, "underlying share 10"
        )
        count_11 = _exact_count(
            scenario.underlying_share_11, population_size, "underlying share 11"
        )
        seats = _exact_count(
            scenario.university_quota, population_size, "the university quota"
        )
        type_counts = {(0, 0): count_00, (0, 1): count_01, (1, 0): count_10, (1, 1): count_11}
        values = {name: [] for name in targets}
        regime_agreements = 0
        threshold_0 = outcomes.tutoring_threshold_0
        threshold_1 = outcomes.tutoring_threshold_1
        thresholds = {0: threshold_0, 1: threshold_1}
        weight = scenario.diversity_weight
        benefit = scenario.benefit
        continuum_x = _continuum_probabilities(outcomes)

        for _ in range(trials):
            applicants: list[tuple[int, int, int, float | None]] = []
            tutored_costs = {0: [], 1: []}
            untutored_costs = {0: [], 1: []}
            for diversity in (0, 1):
                for _ in range(type_counts[1, diversity]):
                    applicants.append((1, diversity, 1, None))
                for _ in range(type_counts[0, diversity]):
                    cost = _sample_cost(cost_distribution, benefit, generator)
                    if cost <= thresholds[diversity]:
                        applicants.append((0, diversity, 1, cost))
                        tutored_costs[diversity].append(cost)
                    else:
                        applicants.append((0, diversity, 0, cost))
                        untutored_costs[diversity].append(cost)

            admitted = _admit_applicants(applicants, outcomes, seats, generator)
            observed_counts = {observed: 0 for observed in OBSERVED_TYPES}
            admitted_observed = {observed: 0 for observed in OBSERVED_TYPES}
            admitted_high_ability = 0
            admitted_diversity = 0
            for index, (underlying_academic, diversity, observed_academic, _) in enumerate(
                applicants
            ):
                observed = (observed_academic, diversity)
                observed_counts[observed] += 1
                if index not in admitted:
                    continue
                admitted_observed[observed] += 1
                admitted_high_ability += underlying_academic
                admitted_diversity += diversity

            trial_x = {
                observed: (
                    continuum_x[observed]
                    if observed_counts[observed] == 0
                    else admitted_observed[observed] / observed_counts[observed]
                )
                for observed in OBSERVED_TYPES
            }
            credibility = {}
            for diversity in (0, 1):
                high_observed = observed_counts[1, diversity]
                credibility[diversity] = (
                    1.0
                    if high_observed == 0
                    else type_counts[1, diversity] / high_observed
                )
            posterior = {
                (0, 0): 0.0,
                (0, 1): weight,
                (1, 0): (1.0 - weight) * credibility[0],
                (1, 1): weight + (1.0 - weight) * credibility[1],
            }
            trial_outcomes = TwoCriterionOutcomes(
                tutoring_threshold_0=(
                    (
                        max(tutored_costs[0], default=0.0)
                        + min(untutored_costs[0], default=benefit)
                    )
                    / 2.0
                ),
                tutoring_threshold_1=(
                    (
                        max(tutored_costs[1], default=0.0)
                        + min(untutored_costs[1], default=benefit)
                    )
                    / 2.0
                ),
                tutoring_participation_rate_0=(
                    0.0
                    if type_counts[0, 0] == 0
                    else len(tutored_costs[0]) / type_counts[0, 0]
                ),
                tutoring_participation_rate_1=(
                    0.0
                    if type_counts[0, 1] == 0
                    else len(tutored_costs[1]) / type_counts[0, 1]
                ),
                population_tutoring_mass=(
                    (len(tutored_costs[0]) + len(tutored_costs[1])) / population_size
                ),
                admission_probability_00=trial_x[0, 0],
                admission_probability_01=trial_x[0, 1],
                admission_probability_10=trial_x[1, 0],
                admission_probability_11=trial_x[1, 1],
                credibility_0=credibility[0],
                credibility_1=credibility[1],
                posterior_evaluation_00=posterior[0, 0],
                posterior_evaluation_01=posterior[0, 1],
                posterior_evaluation_10=posterior[1, 0],
                posterior_evaluation_11=posterior[1, 1],
                admitted_high_ability_share=admitted_high_ability / seats,
                admitted_diversity_share=admitted_diversity / seats,
                aggregate_tutoring_expenditure=(
                    (
                        math.fsum(tutored_costs[0])
                        + math.fsum(tutored_costs[1])
                    )
                    / population_size
                ),
            )
            regime_agreements += (
                _regime_from_outcomes(trial_outcomes) is continuum_equilibrium.regime
            )
            values["tutoring_threshold_0"].append(trial_outcomes.tutoring_threshold_0)
            values["tutoring_threshold_1"].append(trial_outcomes.tutoring_threshold_1)
            values["tutoring_participation_rate_0"].append(
                trial_outcomes.tutoring_participation_rate_0
            )
            values["tutoring_participation_rate_1"].append(
                trial_outcomes.tutoring_participation_rate_1
            )
            values["population_tutoring_mass"].append(
                trial_outcomes.population_tutoring_mass
            )
            values["admission_probability_00"].append(
                trial_outcomes.admission_probability_00
            )
            values["admission_probability_01"].append(
                trial_outcomes.admission_probability_01
            )
            values["admission_probability_10"].append(
                trial_outcomes.admission_probability_10
            )
            values["admission_probability_11"].append(
                trial_outcomes.admission_probability_11
            )
            values["credibility_0"].append(trial_outcomes.credibility_0)
            values["credibility_1"].append(trial_outcomes.credibility_1)
            values["posterior_evaluation_00"].append(
                trial_outcomes.posterior_evaluation_00
            )
            values["posterior_evaluation_01"].append(
                trial_outcomes.posterior_evaluation_01
            )
            values["posterior_evaluation_10"].append(
                trial_outcomes.posterior_evaluation_10
            )
            values["posterior_evaluation_11"].append(
                trial_outcomes.posterior_evaluation_11
            )
            values["admitted_high_ability_share"].append(
                trial_outcomes.admitted_high_ability_share
            )
            values["admitted_diversity_share"].append(
                trial_outcomes.admitted_diversity_share
            )
            values["aggregate_tutoring_expenditure"].append(
                trial_outcomes.aggregate_tutoring_expenditure
            )

        metric_summaries = {
            name: _summarize_metric(values[name], target)
            for name, target in targets.items()
        }
        population_summaries.append(
            TwoCriterionMonteCarloPopulationSummary(
                population_size=population_size,
                trials=trials,
                regime_agreement_rate=regime_agreements / trials,
                **metric_summaries,
            )
        )

    return TwoCriterionMonteCarloValidation(
        continuum_equilibrium=continuum_equilibrium,
        population_summaries=tuple(population_summaries),
    )
