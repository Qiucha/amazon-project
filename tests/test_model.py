import math
import unittest

from admissions_simulation import (
    CostDistribution,
    Regime,
    Scenario,
    Stability,
    UniformCostDistribution,
    analyze_scenario,
    run_monte_carlo_validation,
)


class QuadraticCostDistribution(CostDistribution):
    def cdf(self, cost: float) -> float:
        return min(1.0, max(0.0, cost)) ** 2

    def partial_first_moment(self, cost: float) -> float:
        clipped_cost = min(1.0, max(0.0, cost))
        return 2 * clipped_cost**3 / 3


class MalformedCostDistribution(CostDistribution):
    def cdf(self, cost: float) -> float:
        return 0.5

    def partial_first_moment(self, cost: float) -> float:
        return 0.0


class TenthPowerCostDistribution(CostDistribution):
    def cdf(self, cost: float) -> float:
        return min(1.0, max(0.0, cost)) ** 10

    def partial_first_moment(self, cost: float) -> float:
        clipped_cost = min(1.0, max(0.0, cost))
        return 10 * clipped_cost**11 / 11


class AnalyzeScenarioTests(unittest.TestCase):
    def test_finds_the_uniform_cost_equilibrium_for_the_paper_baseline(self) -> None:
        cases = (
            (0.33, (-0.3 + math.sqrt(1.014)) / 1.4),
            (0.4, 4 / 7),
        )

        for university_quota, expected_cutoff in cases:
            with self.subTest(university_quota=university_quota):
                analysis = analyze_scenario(
                    Scenario(
                        benefit=1.0,
                        high_ability_share=0.3,
                        university_quota=university_quota,
                    ),
                    UniformCostDistribution(upper=1.0),
                )

                self.assertEqual(len(analysis.equilibria), 1)
                self.assertAlmostEqual(
                    analysis.equilibria[0].tutoring_cost_cutoff,
                    expected_cutoff,
                )

    def test_reports_the_admissions_policy_and_quota_regime(self) -> None:
        analysis = analyze_scenario(
            Scenario(benefit=1.0, high_ability_share=0.3, university_quota=0.4),
            UniformCostDistribution(upper=1.0),
        )

        equilibrium = analysis.equilibria[0]
        self.assertEqual(equilibrium.regime, Regime.TIGHT)
        self.assertAlmostEqual(equilibrium.high_score_admission_probability, 4 / 7)
        self.assertEqual(equilibrium.low_score_admission_probability, 0.0)

    def test_finds_classifies_and_selects_among_multiple_equilibria(self) -> None:
        analysis = analyze_scenario(
            Scenario(benefit=1.0, high_ability_share=0.3, university_quota=0.85),
            UniformCostDistribution(upper=1.0),
        )

        expected_cutoffs = (
            (1 - math.sqrt(1 / 7)) / 2,
            (1 + math.sqrt(1 / 7)) / 2,
            (-0.3 + math.sqrt(2.47)) / 1.4,
        )
        self.assertEqual(len(analysis.equilibria), 3)
        for equilibrium, expected_cutoff in zip(
            analysis.equilibria, expected_cutoffs, strict=True
        ):
            self.assertAlmostEqual(
                equilibrium.tutoring_cost_cutoff, expected_cutoff, places=7
            )
        self.assertEqual(
            tuple(equilibrium.stability for equilibrium in analysis.equilibria),
            (Stability.STABLE, Stability.UNSTABLE, Stability.STABLE),
        )
        self.assertIs(analysis.selected_equilibrium, analysis.equilibria[-1])

    def test_reports_distinct_tutoring_and_admissions_outcomes(self) -> None:
        analysis = analyze_scenario(
            Scenario(benefit=1.0, high_ability_share=0.3, university_quota=0.4),
            UniformCostDistribution(upper=1.0),
        )

        outcomes = analysis.equilibria[0].outcomes
        self.assertAlmostEqual(outcomes.low_ability_tutoring_participation_rate, 4 / 7)
        self.assertAlmostEqual(outcomes.population_tutoring_mass, 0.4)
        self.assertAlmostEqual(outcomes.credibility, 3 / 7)
        self.assertAlmostEqual(outcomes.admitted_high_ability_share, 3 / 7)
        self.assertAlmostEqual(outcomes.aggregate_tutoring_expenditure, 4 / 35)

    def test_accepts_a_configurable_cost_distribution(self) -> None:
        analysis = analyze_scenario(
            Scenario(
                benefit=1.0,
                high_ability_share=0.3,
                university_quota=0.2375,
            ),
            QuadraticCostDistribution(),
        )

        equilibrium = analysis.equilibria[0]
        self.assertAlmostEqual(equilibrium.tutoring_cost_cutoff, 0.5)
        self.assertAlmostEqual(
            equilibrium.outcomes.low_ability_tutoring_participation_rate, 0.25
        )
        self.assertAlmostEqual(
            equilibrium.outcomes.aggregate_tutoring_expenditure, 7 / 120
        )

    def test_rejects_scenarios_outside_the_model_domain(self) -> None:
        invalid_scenarios = (
            Scenario(benefit=0.0, high_ability_share=0.3, university_quota=0.4),
            Scenario(benefit=1.0, high_ability_share=0.0, university_quota=0.4),
            Scenario(benefit=1.0, high_ability_share=1.0, university_quota=0.4),
            Scenario(benefit=1.0, high_ability_share=0.3, university_quota=0.0),
            Scenario(benefit=1.0, high_ability_share=0.3, university_quota=1.0),
        )

        for scenario in invalid_scenarios:
            with self.subTest(scenario=scenario), self.assertRaises(ValueError):
                analyze_scenario(scenario, UniformCostDistribution(upper=1.0))

    def test_finds_a_neutral_equilibrium_that_only_touches_the_fixed_point(self) -> None:
        neutral_cutoff = 1 / math.sqrt(3)
        university_quota = 1 - 0.7 * neutral_cutoff * (
            1 - neutral_cutoff**2
        )
        analysis = analyze_scenario(
            Scenario(
                benefit=1.0,
                high_ability_share=0.3,
                university_quota=university_quota,
            ),
            QuadraticCostDistribution(),
        )

        neutral_equilibria = tuple(
            equilibrium
            for equilibrium in analysis.equilibria
            if equilibrium.stability is Stability.NEUTRAL
        )
        self.assertEqual(len(neutral_equilibria), 1)
        self.assertAlmostEqual(
            neutral_equilibria[0].tutoring_cost_cutoff,
            neutral_cutoff,
            places=7,
        )

    def test_rejects_malformed_cost_distributions(self) -> None:
        scenario = Scenario(
            benefit=1.0,
            high_ability_share=0.3,
            university_quota=0.4,
        )

        with self.assertRaises(ValueError):
            UniformCostDistribution(upper=0.0)
        with self.assertRaises(ValueError):
            analyze_scenario(scenario, MalformedCostDistribution())

    def test_tight_root_is_directionally_stable_even_with_a_steep_response(self) -> None:
        cutoff = 0.8
        high_ability_share = 0.3
        university_quota = cutoff * (
            high_ability_share
            + (1 - high_ability_share) * cutoff**10
        )
        analysis = analyze_scenario(
            Scenario(
                benefit=1.0,
                high_ability_share=high_ability_share,
                university_quota=university_quota,
            ),
            TenthPowerCostDistribution(),
        )

        equilibrium = min(
            analysis.equilibria,
            key=lambda candidate: abs(candidate.tutoring_cost_cutoff - cutoff),
        )
        self.assertAlmostEqual(equilibrium.tutoring_cost_cutoff, cutoff)
        self.assertEqual(equilibrium.stability, Stability.STABLE)


class MonteCarloValidationTests(unittest.TestCase):
    def test_repeats_a_seeded_finite_population_validation_exactly(self) -> None:
        scenario = Scenario(
            benefit=1.0,
            high_ability_share=0.3,
            university_quota=0.4,
        )

        first = run_monte_carlo_validation(
            scenario,
            UniformCostDistribution(upper=1.0),
            population_sizes=(100,),
            trials=5,
            seed=1729,
        )
        second = run_monte_carlo_validation(
            scenario,
            UniformCostDistribution(upper=1.0),
            population_sizes=(100,),
            trials=5,
            seed=1729,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.population_summaries[0].population_size, 100)
        self.assertEqual(first.population_summaries[0].trials, 5)

    def test_recovers_the_paper_baseline_continuum_outcomes(self) -> None:
        validation = run_monte_carlo_validation(
            Scenario(
                benefit=1.0,
                high_ability_share=0.3,
                university_quota=0.4,
            ),
            UniformCostDistribution(upper=1.0),
            population_sizes=(1000,),
            trials=200,
            seed=1729,
        )

        summary = validation.population_summaries[0]
        self.assertEqual(summary.regime_agreement_rate, 1.0)
        expected_means = (
            (summary.tutoring_cost_cutoff.mean, 4 / 7),
            (
                summary.low_ability_tutoring_participation_rate.mean,
                4 / 7,
            ),
            (summary.population_tutoring_mass.mean, 0.4),
            (summary.high_score_admission_probability.mean, 4 / 7),
            (summary.low_score_admission_probability.mean, 0.0),
            (summary.credibility.mean, 3 / 7),
            (summary.admitted_high_ability_share.mean, 3 / 7),
            (summary.aggregate_tutoring_expenditure.mean, 4 / 35),
        )
        for actual, expected in expected_means:
            with self.subTest(expected=expected):
                self.assertAlmostEqual(actual, expected, delta=0.015)

    def test_requires_at_least_one_finite_population_size(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "at least one population size"
        ):
            run_monte_carlo_validation(
                Scenario(
                    benefit=1.0,
                    high_ability_share=0.3,
                    university_quota=0.4,
                ),
                UniformCostDistribution(upper=1.0),
                population_sizes=(),
                trials=5,
                seed=1729,
            )


if __name__ == "__main__":
    unittest.main()
