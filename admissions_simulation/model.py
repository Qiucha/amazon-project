from dataclasses import dataclass
from enum import Enum
import math
import random
from typing import Protocol


class CostDistribution(Protocol):
    """Continuous tutoring-cost distribution supported on ``[0, B]``."""

    def cdf(self, cost: float) -> float: ...

    def partial_first_moment(self, cost: float) -> float: ...


class Regime(Enum):
    LOOSE = "loose"
    TIGHT = "tight"


class Stability(Enum):
    STABLE = "stable"
    UNSTABLE = "unstable"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class Scenario:
    benefit: float
    high_ability_share: float
    university_quota: float


@dataclass(frozen=True)
class UniformCostDistribution:
    upper: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.upper) or self.upper <= 0.0:
            raise ValueError("upper must be positive and finite")

    def cdf(self, cost: float) -> float:
        return min(1.0, max(0.0, cost / self.upper))

    def partial_first_moment(self, cost: float) -> float:
        clipped_cost = min(self.upper, max(0.0, cost))
        return clipped_cost**2 / (2.0 * self.upper)


@dataclass(frozen=True)
class OutcomeMetrics:
    low_ability_tutoring_participation_rate: float
    population_tutoring_mass: float
    credibility: float
    admitted_high_ability_share: float
    aggregate_tutoring_expenditure: float


@dataclass(frozen=True)
class Equilibrium:
    tutoring_cost_cutoff: float
    regime: Regime
    high_score_admission_probability: float
    low_score_admission_probability: float
    stability: Stability
    outcomes: OutcomeMetrics


@dataclass(frozen=True)
class ScenarioAnalysis:
    equilibria: tuple[Equilibrium, ...]
    selected_equilibrium: Equilibrium | None


@dataclass(frozen=True)
class MonteCarloMetricSummary:
    mean: float
    bias: float
    root_mean_squared_error: float
    standard_error: float


@dataclass(frozen=True)
class MonteCarloPopulationSummary:
    population_size: int
    trials: int
    regime_agreement_rate: float
    tutoring_cost_cutoff: MonteCarloMetricSummary
    high_score_admission_probability: MonteCarloMetricSummary
    low_score_admission_probability: MonteCarloMetricSummary
    low_ability_tutoring_participation_rate: MonteCarloMetricSummary
    population_tutoring_mass: MonteCarloMetricSummary
    credibility: MonteCarloMetricSummary
    admitted_high_ability_share: MonteCarloMetricSummary
    aggregate_tutoring_expenditure: MonteCarloMetricSummary


@dataclass(frozen=True)
class MonteCarloValidation:
    continuum_equilibrium: Equilibrium
    population_summaries: tuple[MonteCarloPopulationSummary, ...]


def _validate_cost_distribution(
    cost_distribution: CostDistribution, benefit: float
) -> None:
    tolerance = 1e-9
    previous_cdf = -tolerance
    previous_moment = -tolerance
    for index in range(257):
        cost = benefit * index / 256
        cdf = cost_distribution.cdf(cost)
        moment = cost_distribution.partial_first_moment(cost)
        if not math.isfinite(cdf) or not -tolerance <= cdf <= 1.0 + tolerance:
            raise ValueError("cost distribution CDF must stay between 0 and 1")
        if cdf + tolerance < previous_cdf:
            raise ValueError("cost distribution CDF must be nondecreasing")
        if not math.isfinite(moment) or not -tolerance <= moment <= benefit + tolerance:
            raise ValueError("partial first moment must stay between 0 and B")
        if moment + tolerance < previous_moment:
            raise ValueError("partial first moment must be nondecreasing")
        previous_cdf = cdf
        previous_moment = moment

    if not math.isclose(cost_distribution.cdf(0.0), 0.0, abs_tol=tolerance):
        raise ValueError("cost distribution CDF must satisfy F(0) = 0")
    if not math.isclose(cost_distribution.cdf(benefit), 1.0, abs_tol=tolerance):
        raise ValueError("cost distribution CDF must satisfy F(B) = 1")
    if not math.isclose(
        cost_distribution.partial_first_moment(0.0), 0.0, abs_tol=tolerance
    ):
        raise ValueError("partial first moment must be zero at cost 0")


def analyze_scenario(
    scenario: Scenario, cost_distribution: CostDistribution
) -> ScenarioAnalysis:
    if scenario.benefit <= 0.0:
        raise ValueError("benefit must be positive")
    if not 0.0 < scenario.high_ability_share < 1.0:
        raise ValueError("high_ability_share must be strictly between 0 and 1")
    if not 0.0 < scenario.university_quota < 1.0:
        raise ValueError("university_quota must be strictly between 0 and 1")
    _validate_cost_distribution(cost_distribution, scenario.benefit)

    def admissions_policy(cutoff: float) -> tuple[Regime, float, float]:
        participation = cost_distribution.cdf(cutoff)
        high_scorer_mass = scenario.high_ability_share + (
            1.0 - scenario.high_ability_share
        ) * participation

        if scenario.university_quota > high_scorer_mass:
            low_scorer_mass = 1.0 - high_scorer_mass
            return (
                Regime.LOOSE,
                1.0,
                (scenario.university_quota - high_scorer_mass)
                / low_scorer_mass,
            )

        return Regime.TIGHT, scenario.university_quota / high_scorer_mass, 0.0

    def tutoring_benefit(cutoff: float) -> float:
        _, high_score_admission, low_score_admission = admissions_policy(cutoff)
        return scenario.benefit * (
            high_score_admission - low_score_admission
        )

    def excess_tutoring_benefit(cutoff: float) -> float:
        return tutoring_benefit(cutoff) - cutoff

    def bisect_root(lower: float, upper: float) -> float:
        lower_value = excess_tutoring_benefit(lower)
        for _ in range(80):
            midpoint = (lower + upper) / 2.0
            midpoint_value = excess_tutoring_benefit(midpoint)
            if (lower_value > 0.0) == (midpoint_value > 0.0):
                lower = midpoint
                lower_value = midpoint_value
            else:
                upper = midpoint
        return (lower + upper) / 2.0

    def minimize_absolute_excess(lower: float, upper: float) -> float:
        inverse_phi = (5.0**0.5 - 1.0) / 2.0
        left_probe = upper - inverse_phi * (upper - lower)
        right_probe = lower + inverse_phi * (upper - lower)
        left_value = abs(excess_tutoring_benefit(left_probe))
        right_value = abs(excess_tutoring_benefit(right_probe))
        for _ in range(80):
            if left_value <= right_value:
                upper = right_probe
                right_probe = left_probe
                right_value = left_value
                left_probe = upper - inverse_phi * (upper - lower)
                left_value = abs(excess_tutoring_benefit(left_probe))
            else:
                lower = left_probe
                left_probe = right_probe
                left_value = right_value
                right_probe = lower + inverse_phi * (upper - lower)
                right_value = abs(excess_tutoring_benefit(right_probe))
        return (lower + upper) / 2.0

    roots: list[float] = []
    grid_size = 8192
    previous_previous_cutoff = 0.0
    previous_previous_value = excess_tutoring_benefit(
        previous_previous_cutoff
    )
    previous_cutoff = scenario.benefit / grid_size
    previous_value = excess_tutoring_benefit(previous_cutoff)
    if abs(previous_previous_value) < 1e-10:
        roots.append(previous_previous_cutoff)
    if previous_previous_value * previous_value < 0.0:
        roots.append(
            bisect_root(previous_previous_cutoff, previous_cutoff)
        )

    for index in range(2, grid_size + 1):
        cutoff = scenario.benefit * index / grid_size
        value = excess_tutoring_benefit(cutoff)
        if (
            abs(previous_value) <= abs(previous_previous_value)
            and abs(previous_value) <= abs(value)
        ):
            candidate = minimize_absolute_excess(
                previous_previous_cutoff, cutoff
            )
            if abs(excess_tutoring_benefit(candidate)) < 1e-9:
                roots.append(candidate)
        if previous_value * value < 0.0:
            roots.append(bisect_root(previous_cutoff, cutoff))
        previous_previous_cutoff = previous_cutoff
        previous_previous_value = previous_value
        previous_cutoff = cutoff
        previous_value = value
    if abs(previous_value) < 1e-10:
        roots.append(scenario.benefit)

    distinct_roots: list[float] = []
    for root in sorted(roots):
        if not distinct_roots or abs(root - distinct_roots[-1]) > 1e-7:
            distinct_roots.append(root)

    def classify_stability(cutoff: float) -> Stability:
        step = max(scenario.benefit * 1e-5, 1e-8)
        lower = max(0.0, cutoff - step)
        upper = min(scenario.benefit, cutoff + step)
        lower_excess = excess_tutoring_benefit(lower)
        upper_excess = excess_tutoring_benefit(upper)
        direction_tolerance = 1e-8
        if (
            lower_excess > direction_tolerance
            and upper_excess < -direction_tolerance
        ):
            return Stability.STABLE
        if (
            lower_excess < -direction_tolerance
            and upper_excess > direction_tolerance
        ):
            return Stability.UNSTABLE
        return Stability.NEUTRAL

    def build_equilibrium(cutoff: float) -> Equilibrium:
        regime, high_score_admission, low_score_admission = admissions_policy(
            cutoff
        )
        participation = cost_distribution.cdf(cutoff)
        population_tutoring_mass = (
            1.0 - scenario.high_ability_share
        ) * participation
        high_scorer_mass = scenario.high_ability_share + population_tutoring_mass
        outcomes = OutcomeMetrics(
            low_ability_tutoring_participation_rate=participation,
            population_tutoring_mass=population_tutoring_mass,
            credibility=scenario.high_ability_share / high_scorer_mass,
            admitted_high_ability_share=(
                scenario.high_ability_share
                * high_score_admission
                / scenario.university_quota
            ),
            aggregate_tutoring_expenditure=(
                (1.0 - scenario.high_ability_share)
                * cost_distribution.partial_first_moment(cutoff)
            ),
        )
        return Equilibrium(
            tutoring_cost_cutoff=cutoff,
            regime=regime,
            high_score_admission_probability=high_score_admission,
            low_score_admission_probability=low_score_admission,
            stability=classify_stability(cutoff),
            outcomes=outcomes,
        )

    equilibria = tuple(build_equilibrium(cutoff) for cutoff in distinct_roots)
    selected_equilibrium = next(
        (
            equilibrium
            for equilibrium in reversed(equilibria)
            if equilibrium.stability is Stability.STABLE
        ),
        None,
    )
    return ScenarioAnalysis(
        equilibria=equilibria,
        selected_equilibrium=selected_equilibrium,
    )


def run_monte_carlo_validation(
    scenario: Scenario,
    cost_distribution: CostDistribution,
    population_sizes: tuple[int, ...],
    trials: int,
    seed: int,
) -> MonteCarloValidation:
    if not population_sizes:
        raise ValueError("at least one population size is required")
    continuum_analysis = analyze_scenario(scenario, cost_distribution)
    continuum_equilibrium = continuum_analysis.selected_equilibrium
    if continuum_equilibrium is None:
        raise ValueError("scenario must have a selected stable equilibrium")
    if not isinstance(trials, int) or isinstance(trials, bool) or trials <= 0:
        raise ValueError("trials must be a positive integer")

    def exact_count(share: float, population_size: int, name: str) -> int:
        count = share * population_size
        rounded_count = round(count)
        if not math.isclose(count, rounded_count, abs_tol=1e-9):
            raise ValueError(
                f"population size must make {name} an integer count"
            )
        return rounded_count

    def sample_cost(generator: random.Random) -> float:
        probability = generator.random()
        lower = 0.0
        upper = scenario.benefit
        for _ in range(48):
            midpoint = (lower + upper) / 2.0
            if cost_distribution.cdf(midpoint) < probability:
                lower = midpoint
            else:
                upper = midpoint
        return (lower + upper) / 2.0

    def summarize(
        values: list[float], target: float
    ) -> MonteCarloMetricSummary:
        mean = math.fsum(values) / len(values)
        squared_errors = [(value - target) ** 2 for value in values]
        if len(values) == 1:
            standard_error = 0.0
        else:
            sample_variance = math.fsum(
                (value - mean) ** 2 for value in values
            ) / (len(values) - 1)
            standard_error = math.sqrt(sample_variance / len(values))
        return MonteCarloMetricSummary(
            mean=mean,
            bias=mean - target,
            root_mean_squared_error=math.sqrt(
                math.fsum(squared_errors) / len(values)
            ),
            standard_error=standard_error,
        )

    targets = {
        "tutoring_cost_cutoff": continuum_equilibrium.tutoring_cost_cutoff,
        "high_score_admission_probability": (
            continuum_equilibrium.high_score_admission_probability
        ),
        "low_score_admission_probability": (
            continuum_equilibrium.low_score_admission_probability
        ),
        "low_ability_tutoring_participation_rate": (
            continuum_equilibrium.outcomes.low_ability_tutoring_participation_rate
        ),
        "population_tutoring_mass": (
            continuum_equilibrium.outcomes.population_tutoring_mass
        ),
        "credibility": continuum_equilibrium.outcomes.credibility,
        "admitted_high_ability_share": (
            continuum_equilibrium.outcomes.admitted_high_ability_share
        ),
        "aggregate_tutoring_expenditure": (
            continuum_equilibrium.outcomes.aggregate_tutoring_expenditure
        ),
    }
    generator = random.Random(seed)
    population_summaries: list[MonteCarloPopulationSummary] = []
    for population_size in population_sizes:
        if (
            not isinstance(population_size, int)
            or isinstance(population_size, bool)
            or population_size <= 0
        ):
            raise ValueError("population sizes must be positive integers")
        high_ability_count = exact_count(
            scenario.high_ability_share,
            population_size,
            "the high-ability share",
        )
        university_seats = exact_count(
            scenario.university_quota,
            population_size,
            "the university quota",
        )
        low_ability_count = population_size - high_ability_count
        values = {name: [] for name in targets}
        regime_agreements = 0

        for _ in range(trials):
            tutored_costs: list[float] = []
            untutored_costs: list[float] = []
            for _ in range(low_ability_count):
                cost = sample_cost(generator)
                if cost <= continuum_equilibrium.tutoring_cost_cutoff:
                    tutored_costs.append(cost)
                else:
                    untutored_costs.append(cost)

            tutoring_count = len(tutored_costs)
            high_scorer_count = high_ability_count + tutoring_count
            low_scorer_count = population_size - high_scorer_count
            if university_seats > high_scorer_count:
                regime = Regime.LOOSE
                admitted_high_ability_count = high_ability_count
                high_score_admission_probability = 1.0
                low_score_admission_probability = (
                    (university_seats - high_scorer_count) / low_scorer_count
                )
            else:
                regime = Regime.TIGHT
                admitted_high_ability_count = sum(
                    applicant < high_ability_count
                    for applicant in generator.sample(
                        range(high_scorer_count), university_seats
                    )
                )
                high_score_admission_probability = (
                    university_seats / high_scorer_count
                )
                low_score_admission_probability = 0.0

            regime_agreements += regime is continuum_equilibrium.regime
            lower_cutoff_bound = max(tutored_costs, default=0.0)
            upper_cutoff_bound = min(
                untutored_costs, default=scenario.benefit
            )
            values["tutoring_cost_cutoff"].append(
                (lower_cutoff_bound + upper_cutoff_bound) / 2.0
            )
            values["high_score_admission_probability"].append(
                high_score_admission_probability
            )
            values["low_score_admission_probability"].append(
                low_score_admission_probability
            )
            values["low_ability_tutoring_participation_rate"].append(
                tutoring_count / low_ability_count
            )
            values["population_tutoring_mass"].append(
                tutoring_count / population_size
            )
            values["credibility"].append(
                high_ability_count / high_scorer_count
            )
            values["admitted_high_ability_share"].append(
                admitted_high_ability_count / university_seats
            )
            values["aggregate_tutoring_expenditure"].append(
                math.fsum(tutored_costs) / population_size
            )

        metric_summaries = {
            name: summarize(values[name], target)
            for name, target in targets.items()
        }
        population_summaries.append(
            MonteCarloPopulationSummary(
                population_size=population_size,
                trials=trials,
                regime_agreement_rate=regime_agreements / trials,
                **metric_summaries,
            )
        )

    return MonteCarloValidation(
        continuum_equilibrium=continuum_equilibrium,
        population_summaries=tuple(population_summaries),
    )
