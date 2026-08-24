from dataclasses import replace
import math
import unittest

from admissions_simulation import (
    CostDistribution,
    Regime,
    Scenario,
    Stability,
    TwoCriterionScenario,
    UniformCostDistribution,
    analyze_scenario,
    analyze_two_criterion_scenario,
    evaluate_threshold_pair,
    one_stage_image,
    run_two_criterion_monte_carlo_validation,
)


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


def illustration_scenario(university_quota: float, diversity_weight: float) -> TwoCriterionScenario:
    return TwoCriterionScenario(
        benefit=1.0,
        university_quota=university_quota,
        diversity_weight=diversity_weight,
        underlying_share_00=0.448,
        underlying_share_01=0.252,
        underlying_share_10=0.192,
        underlying_share_11=0.108,
    )


class EvaluateThresholdPairTests(unittest.TestCase):
    def test_nested_tight_pair_recovers_the_one_stage_paper_baseline(self) -> None:
        cost_distribution = UniformCostDistribution(upper=1.0)
        scenario = illustration_scenario(0.40, 0.0)
        one_stage = analyze_scenario(
            Scenario(benefit=1.0, high_ability_share=0.3, university_quota=0.40),
            cost_distribution,
        ).selected_equilibrium
        assert one_stage is not None

        outcomes = evaluate_threshold_pair(
            scenario,
            cost_distribution,
            one_stage.tutoring_cost_cutoff,
            one_stage.tutoring_cost_cutoff,
        )

        self.assertAlmostEqual(outcomes.tutoring_threshold_0, 4 / 7)
        self.assertAlmostEqual(outcomes.tutoring_threshold_1, 4 / 7)
        self.assertAlmostEqual(outcomes.tutoring_participation_rate_0, 4 / 7)
        self.assertAlmostEqual(outcomes.tutoring_participation_rate_1, 4 / 7)
        self.assertAlmostEqual(outcomes.population_tutoring_mass, 0.4)
        self.assertAlmostEqual(outcomes.admission_probability_10, 4 / 7)
        self.assertAlmostEqual(outcomes.admission_probability_11, 4 / 7)
        self.assertAlmostEqual(outcomes.admission_probability_00, 0.0)
        self.assertAlmostEqual(outcomes.admission_probability_01, 0.0)
        self.assertAlmostEqual(outcomes.credibility_0, 3 / 7)
        self.assertAlmostEqual(outcomes.credibility_1, 3 / 7)
        self.assertAlmostEqual(outcomes.posterior_evaluation_00, 0.0)
        self.assertAlmostEqual(outcomes.posterior_evaluation_01, 0.0)
        self.assertAlmostEqual(outcomes.posterior_evaluation_10, 3 / 7)
        self.assertAlmostEqual(outcomes.posterior_evaluation_11, 3 / 7)
        self.assertAlmostEqual(outcomes.admitted_high_ability_share, 3 / 7)
        self.assertAlmostEqual(outcomes.admitted_diversity_share, 0.36)
        self.assertAlmostEqual(outcomes.aggregate_tutoring_expenditure, 4 / 35)

    def test_nested_pair_at_the_one_stage_cutoff_matches_analyze_scenario(self) -> None:
        cost_distribution = UniformCostDistribution(upper=1.0)
        university_quota = 0.33
        scenario = illustration_scenario(university_quota, 0.0)
        one_stage = analyze_scenario(
            Scenario(
                benefit=1.0,
                high_ability_share=0.3,
                university_quota=university_quota,
            ),
            cost_distribution,
        ).selected_equilibrium
        assert one_stage is not None
        cutoff = one_stage.tutoring_cost_cutoff
        expected_cutoff = (-0.3 + math.sqrt(1.014)) / 1.4

        outcomes = evaluate_threshold_pair(
            scenario,
            cost_distribution,
            cutoff,
            cutoff,
        )

        self.assertAlmostEqual(cutoff, expected_cutoff)
        self.assertAlmostEqual(outcomes.tutoring_threshold_0, cutoff)
        self.assertAlmostEqual(outcomes.tutoring_threshold_1, cutoff)
        self.assertAlmostEqual(
            outcomes.tutoring_participation_rate_0,
            one_stage.outcomes.low_ability_tutoring_participation_rate,
        )
        self.assertAlmostEqual(
            outcomes.tutoring_participation_rate_1,
            one_stage.outcomes.low_ability_tutoring_participation_rate,
        )
        self.assertAlmostEqual(
            outcomes.population_tutoring_mass,
            one_stage.outcomes.population_tutoring_mass,
        )
        self.assertAlmostEqual(
            outcomes.admission_probability_10,
            one_stage.high_score_admission_probability,
        )
        self.assertAlmostEqual(
            outcomes.admission_probability_11,
            one_stage.high_score_admission_probability,
        )
        self.assertAlmostEqual(
            outcomes.admission_probability_00,
            one_stage.low_score_admission_probability,
        )
        self.assertAlmostEqual(
            outcomes.admission_probability_01,
            one_stage.low_score_admission_probability,
        )
        self.assertAlmostEqual(
            outcomes.credibility_0, one_stage.outcomes.credibility
        )
        self.assertAlmostEqual(
            outcomes.credibility_1, one_stage.outcomes.credibility
        )
        self.assertAlmostEqual(
            outcomes.posterior_evaluation_10, one_stage.outcomes.credibility
        )
        self.assertAlmostEqual(
            outcomes.posterior_evaluation_11, one_stage.outcomes.credibility
        )
        self.assertAlmostEqual(outcomes.posterior_evaluation_00, 0.0)
        self.assertAlmostEqual(outcomes.posterior_evaluation_01, 0.0)
        self.assertAlmostEqual(
            outcomes.admitted_high_ability_share,
            one_stage.outcomes.admitted_high_ability_share,
        )
        self.assertAlmostEqual(
            outcomes.aggregate_tutoring_expenditure,
            one_stage.outcomes.aggregate_tutoring_expenditure,
        )

    def test_stratified_ranking_at_zero_thresholds_is_tutoring_independent(self) -> None:
        outcomes = evaluate_threshold_pair(
            illustration_scenario(0.33, 0.75),
            UniformCostDistribution(upper=1.0),
            0.0,
            0.0,
        )

        self.assertGreater(
            outcomes.posterior_evaluation_11, outcomes.posterior_evaluation_01
        )
        self.assertGreater(
            outcomes.posterior_evaluation_01, outcomes.posterior_evaluation_10
        )
        self.assertGreater(
            outcomes.posterior_evaluation_10, outcomes.posterior_evaluation_00
        )
        self.assertAlmostEqual(outcomes.posterior_evaluation_11, 1.0)
        self.assertAlmostEqual(outcomes.posterior_evaluation_01, 0.75)
        self.assertAlmostEqual(outcomes.posterior_evaluation_10, 0.25)
        self.assertAlmostEqual(outcomes.posterior_evaluation_00, 0.0)
        self.assertAlmostEqual(outcomes.admission_probability_11, 1.0)
        self.assertAlmostEqual(
            outcomes.admission_probability_01, (0.33 - 0.108) / 0.252
        )
        self.assertAlmostEqual(outcomes.admission_probability_10, 0.0)
        self.assertAlmostEqual(outcomes.admission_probability_00, 0.0)

    def test_mixed_untutored_ranking_puts_high_academic_ahead_of_diversity(self) -> None:
        outcomes = evaluate_threshold_pair(
            illustration_scenario(0.40, 0.25),
            UniformCostDistribution(upper=1.0),
            0.0,
            0.0,
        )

        self.assertGreater(
            outcomes.posterior_evaluation_11, outcomes.posterior_evaluation_10
        )
        self.assertGreater(
            outcomes.posterior_evaluation_10, outcomes.posterior_evaluation_01
        )
        self.assertGreater(
            outcomes.posterior_evaluation_01, outcomes.posterior_evaluation_00
        )
        self.assertAlmostEqual(outcomes.posterior_evaluation_11, 1.0)
        self.assertAlmostEqual(outcomes.posterior_evaluation_10, 0.75)
        self.assertAlmostEqual(outcomes.posterior_evaluation_01, 0.25)
        self.assertAlmostEqual(outcomes.posterior_evaluation_00, 0.0)
        self.assertAlmostEqual(outcomes.admission_probability_11, 1.0)
        self.assertAlmostEqual(outcomes.admission_probability_10, 1.0)
        self.assertAlmostEqual(outcomes.admission_probability_01, 0.1 / 0.252)
        self.assertAlmostEqual(outcomes.admission_probability_00, 0.0)

    def test_mixed_curve_identity_holds_at_a_constructed_pair(self) -> None:
        diversity_weight = 0.25
        outcomes = evaluate_threshold_pair(
            illustration_scenario(0.33, diversity_weight),
            UniformCostDistribution(upper=1.0),
            0.0,
            3 / 14,
        )

        self.assertAlmostEqual(
            outcomes.credibility_0 - outcomes.credibility_1,
            diversity_weight / (1.0 - diversity_weight),
        )
        self.assertAlmostEqual(
            outcomes.posterior_evaluation_10, outcomes.posterior_evaluation_11
        )
        self.assertGreater(outcomes.admission_probability_10, 0.0)
        self.assertLess(outcomes.admission_probability_10, 1.0)
        self.assertGreater(outcomes.admission_probability_11, 0.0)
        self.assertLess(outcomes.admission_probability_11, 1.0)
        self.assertAlmostEqual(
            outcomes.admission_probability_10, outcomes.admission_probability_11
        )

    def test_mixed_tight_quota_uses_the_unequal_rationalizing_split(self) -> None:
        tutoring_threshold_0 = 0.2
        diversity_weight = 0.25
        credibility_0 = 0.192 / (0.192 + 0.448 * tutoring_threshold_0)
        credibility_1 = credibility_0 - diversity_weight / (1.0 - diversity_weight)
        tutoring_threshold_1 = (0.108 / credibility_1 - 0.108) / 0.252
        high_academic_0 = 0.192 + 0.448 * tutoring_threshold_0
        high_academic_1 = 0.108 + 0.252 * tutoring_threshold_1
        university_quota = (
            high_academic_0 * tutoring_threshold_0
            + high_academic_1 * tutoring_threshold_1
        )
        outcomes = evaluate_threshold_pair(
            illustration_scenario(university_quota, diversity_weight),
            UniformCostDistribution(upper=1.0),
            tutoring_threshold_0,
            tutoring_threshold_1,
        )

        self.assertNotAlmostEqual(tutoring_threshold_0, tutoring_threshold_1)
        self.assertAlmostEqual(
            outcomes.admission_probability_10, tutoring_threshold_0
        )
        self.assertAlmostEqual(
            outcomes.admission_probability_11, tutoring_threshold_1
        )
        self.assertAlmostEqual(outcomes.admission_probability_00, 0.0)
        self.assertAlmostEqual(outcomes.admission_probability_01, 0.0)
        self.assertAlmostEqual(
            outcomes.posterior_evaluation_10, outcomes.posterior_evaluation_11
        )

    def test_every_one_stage_branch_is_the_image_of_the_diagonal_pair(self) -> None:
        cost_distribution = UniformCostDistribution(upper=1.0)
        for university_quota in (0.33, 0.40, 0.85):
            with self.subTest(university_quota=university_quota):
                one_stage = analyze_scenario(
                    Scenario(
                        benefit=1.0,
                        high_ability_share=0.3,
                        university_quota=university_quota,
                    ),
                    cost_distribution,
                )
                scenario = illustration_scenario(university_quota, 0.0)
                for equilibrium in one_stage.equilibria:
                    outcomes = evaluate_threshold_pair(
                        scenario,
                        cost_distribution,
                        equilibrium.tutoring_cost_cutoff,
                        equilibrium.tutoring_cost_cutoff,
                    )
                    self.assertAlmostEqual(
                        outcomes.admission_probability_10,
                        equilibrium.high_score_admission_probability,
                    )
                    self.assertAlmostEqual(
                        outcomes.admission_probability_11,
                        equilibrium.high_score_admission_probability,
                    )
                    self.assertAlmostEqual(
                        outcomes.admission_probability_00,
                        equilibrium.low_score_admission_probability,
                    )
                    self.assertAlmostEqual(
                        outcomes.admission_probability_01,
                        equilibrium.low_score_admission_probability,
                    )

    def test_empty_high_academic_pool_is_fully_credible(self) -> None:
        outcomes = evaluate_threshold_pair(
            TwoCriterionScenario(
                benefit=1.0,
                university_quota=0.4,
                diversity_weight=0.25,
                underlying_share_00=0.4,
                underlying_share_01=0.3,
                underlying_share_10=0.0,
                underlying_share_11=0.3,
            ),
            UniformCostDistribution(upper=1.0),
            0.0,
            0.0,
        )

        self.assertAlmostEqual(outcomes.credibility_0, 1.0)
        self.assertAlmostEqual(outcomes.posterior_evaluation_10, 0.75)

    def test_reports_input_thresholds_even_when_they_are_clipped_for_masses(self) -> None:
        outcomes = evaluate_threshold_pair(
            illustration_scenario(0.40, 0.0),
            UniformCostDistribution(upper=1.0),
            -0.25,
            1.5,
        )

        self.assertEqual(outcomes.tutoring_threshold_0, -0.25)
        self.assertEqual(outcomes.tutoring_threshold_1, 1.5)
        self.assertEqual(outcomes.tutoring_participation_rate_0, 0.0)
        self.assertEqual(outcomes.tutoring_participation_rate_1, 1.0)

    def test_rejects_scenarios_outside_the_model_domain(self) -> None:
        cost_distribution = UniformCostDistribution(upper=1.0)
        invalid_scenarios = (
            replace(illustration_scenario(0.40, 0.0), benefit=0.0),
            replace(illustration_scenario(0.40, 0.0), university_quota=0.0),
            replace(illustration_scenario(0.40, 0.0), university_quota=1.0),
            replace(illustration_scenario(0.40, 0.0), diversity_weight=-0.1),
            replace(illustration_scenario(0.40, 0.0), diversity_weight=1.1),
            replace(illustration_scenario(0.40, 0.0), underlying_share_00=-0.01),
            TwoCriterionScenario(
                benefit=1.0,
                university_quota=0.40,
                diversity_weight=0.0,
                underlying_share_00=0.5,
                underlying_share_01=0.5,
                underlying_share_10=0.5,
                underlying_share_11=0.5,
            ),
        )
        for scenario in invalid_scenarios:
            with self.subTest(scenario=scenario), self.assertRaises(ValueError):
                evaluate_threshold_pair(scenario, cost_distribution, 0.4, 0.4)

    def test_rejects_malformed_cost_distributions(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_threshold_pair(
                illustration_scenario(0.40, 0.0),
                MalformedCostDistribution(),
                0.4,
                0.4,
            )


class OneStageImageTests(unittest.TestCase):
    def test_nested_illustration_maps_to_the_paper_scenario(self) -> None:
        image = one_stage_image(illustration_scenario(0.40, 0.0))

        self.assertEqual(
            image,
            Scenario(benefit=1.0, high_ability_share=0.3, university_quota=0.40),
        )

    def test_rejects_a_positive_diversity_weight(self) -> None:
        with self.assertRaises(ValueError):
            one_stage_image(illustration_scenario(0.40, 0.25))

    def test_rejects_unequal_high_ability_odds(self) -> None:
        scenario = replace(
            illustration_scenario(0.40, 0.0),
            underlying_share_00=0.5,
            underlying_share_01=0.2,
            underlying_share_10=0.2,
            underlying_share_11=0.1,
        )

        with self.assertRaises(ValueError):
            one_stage_image(scenario)


class AnalyzeTwoCriterionScenarioTests(unittest.TestCase):
    def test_nested_selected_root_recovers_figure_1_at_quota_0_40(self) -> None:
        scenario = illustration_scenario(0.40, 0.0)
        analysis = analyze_two_criterion_scenario(
            scenario,
            UniformCostDistribution(upper=1.0),
        )
        selected = analysis.selected_equilibrium
        assert selected is not None
        outcomes = selected.outcomes

        self.assertEqual(len(analysis.equilibria), 1)
        self.assertIs(selected, analysis.equilibria[0])
        self.assertEqual(selected.regime, Regime.TIGHT)
        self.assertEqual(selected.stability, Stability.STABLE)
        self.assertAlmostEqual(outcomes.tutoring_threshold_0, 4 / 7)
        self.assertAlmostEqual(outcomes.tutoring_threshold_1, 4 / 7)
        self.assertAlmostEqual(outcomes.tutoring_participation_rate_0, 4 / 7)
        self.assertAlmostEqual(outcomes.tutoring_participation_rate_1, 4 / 7)
        self.assertAlmostEqual(outcomes.population_tutoring_mass, 0.4)
        self.assertAlmostEqual(outcomes.admission_probability_10, 4 / 7)
        self.assertAlmostEqual(outcomes.admission_probability_11, 4 / 7)
        self.assertAlmostEqual(outcomes.admission_probability_00, 0.0)
        self.assertAlmostEqual(outcomes.admission_probability_01, 0.0)
        self.assertAlmostEqual(outcomes.credibility_0, 3 / 7)
        self.assertAlmostEqual(outcomes.credibility_1, 3 / 7)
        self.assertAlmostEqual(outcomes.admitted_high_ability_share, 3 / 7)
        self.assertAlmostEqual(outcomes.aggregate_tutoring_expenditure, 4 / 35)

    def _assert_nested_image(self, two, one) -> None:
        outcomes = two.outcomes
        self.assertEqual(two.regime, one.regime)
        self.assertEqual(two.stability, one.stability)
        self.assertAlmostEqual(
            outcomes.tutoring_threshold_0, one.tutoring_cost_cutoff, places=7
        )
        self.assertAlmostEqual(
            outcomes.tutoring_threshold_1, one.tutoring_cost_cutoff, places=7
        )
        self.assertAlmostEqual(
            outcomes.admission_probability_10,
            one.high_score_admission_probability,
        )
        self.assertAlmostEqual(
            outcomes.admission_probability_11,
            one.high_score_admission_probability,
        )
        self.assertAlmostEqual(
            outcomes.admission_probability_00,
            one.low_score_admission_probability,
        )
        self.assertAlmostEqual(
            outcomes.admission_probability_01,
            one.low_score_admission_probability,
        )
        self.assertAlmostEqual(
            outcomes.tutoring_participation_rate_0,
            one.outcomes.low_ability_tutoring_participation_rate,
        )
        self.assertAlmostEqual(
            outcomes.tutoring_participation_rate_1,
            one.outcomes.low_ability_tutoring_participation_rate,
        )
        self.assertAlmostEqual(
            outcomes.population_tutoring_mass,
            one.outcomes.population_tutoring_mass,
        )
        self.assertAlmostEqual(outcomes.credibility_0, one.outcomes.credibility)
        self.assertAlmostEqual(outcomes.credibility_1, one.outcomes.credibility)
        self.assertAlmostEqual(
            outcomes.admitted_high_ability_share,
            one.outcomes.admitted_high_ability_share,
        )
        self.assertAlmostEqual(
            outcomes.aggregate_tutoring_expenditure,
            one.outcomes.aggregate_tutoring_expenditure,
        )

    def test_nested_branches_match_analyze_scenario_as_a_tuple(self) -> None:
        cost_distribution = UniformCostDistribution(upper=1.0)
        for university_quota in (0.33, 0.40, 0.85):
            with self.subTest(university_quota=university_quota):
                scenario = illustration_scenario(university_quota, 0.0)
                two = analyze_two_criterion_scenario(scenario, cost_distribution)
                one = analyze_scenario(one_stage_image(scenario), cost_distribution)
                self.assertEqual(len(two.equilibria), len(one.equilibria))
                for two_equilibrium, one_equilibrium in zip(
                    two.equilibria, one.equilibria, strict=True
                ):
                    self._assert_nested_image(two_equilibrium, one_equilibrium)
                if one.selected_equilibrium is None:
                    self.assertIsNone(two.selected_equilibrium)
                else:
                    assert two.selected_equilibrium is not None
                    self._assert_nested_image(
                        two.selected_equilibrium, one.selected_equilibrium
                    )

    def test_nested_tight_product_holds_at_the_selected_root(self) -> None:
        scenario = illustration_scenario(0.40, 0.0)
        selected = analyze_two_criterion_scenario(
            scenario, UniformCostDistribution(upper=1.0)
        ).selected_equilibrium
        assert selected is not None
        cutoff = selected.outcomes.tutoring_threshold_0
        high_ability_share = 0.3
        self.assertAlmostEqual(
            cutoff
            * (high_ability_share + (1.0 - high_ability_share) * cutoff),
            scenario.university_quota,
        )

    def test_mixed_selected_root_satisfies_the_curve_identities(self) -> None:
        diversity_weight = 0.25
        analysis = analyze_two_criterion_scenario(
            illustration_scenario(0.33, diversity_weight),
            UniformCostDistribution(upper=1.0),
        )
        selected = analysis.selected_equilibrium
        assert selected is not None
        outcomes = selected.outcomes

        self.assertGreater(outcomes.admission_probability_10, 0.0)
        self.assertLess(outcomes.admission_probability_10, 1.0)
        self.assertGreater(outcomes.admission_probability_11, 0.0)
        self.assertLess(outcomes.admission_probability_11, 1.0)
        self.assertAlmostEqual(
            outcomes.posterior_evaluation_10, outcomes.posterior_evaluation_11
        )
        self.assertAlmostEqual(
            outcomes.credibility_0 - outcomes.credibility_1,
            diversity_weight / (1.0 - diversity_weight),
        )
        self.assertEqual(selected.regime, Regime.TIGHT)
        self.assertAlmostEqual(
            outcomes.tutoring_threshold_0,
            outcomes.admission_probability_10 - outcomes.admission_probability_00,
        )
        self.assertAlmostEqual(
            outcomes.tutoring_threshold_1,
            outcomes.admission_probability_11 - outcomes.admission_probability_01,
        )

    def test_selected_root_posteriors_match_a_recompute_at_its_own_thresholds(
        self,
    ) -> None:
        cost_distribution = UniformCostDistribution(upper=1.0)
        grid = (
            (0.40, 0.0),
            (0.33, 0.25),
            (0.40, 0.75),
        )
        for university_quota, diversity_weight in grid:
            with self.subTest(
                university_quota=university_quota, diversity_weight=diversity_weight
            ):
                scenario = illustration_scenario(university_quota, diversity_weight)
                selected = analyze_two_criterion_scenario(
                    scenario, cost_distribution
                ).selected_equilibrium
                assert selected is not None
                outcomes = selected.outcomes
                recomputed = evaluate_threshold_pair(
                    scenario,
                    cost_distribution,
                    outcomes.tutoring_threshold_0,
                    outcomes.tutoring_threshold_1,
                )
                self.assertAlmostEqual(
                    outcomes.posterior_evaluation_00,
                    recomputed.posterior_evaluation_00,
                )
                self.assertAlmostEqual(
                    outcomes.posterior_evaluation_01,
                    recomputed.posterior_evaluation_01,
                )
                self.assertAlmostEqual(
                    outcomes.posterior_evaluation_10,
                    recomputed.posterior_evaluation_10,
                )
                self.assertAlmostEqual(
                    outcomes.posterior_evaluation_11,
                    recomputed.posterior_evaluation_11,
                )

    def test_stratified_ranking_is_tutoring_independent_and_selects_who_competes(
        self,
    ) -> None:
        cost_distribution = UniformCostDistribution(upper=1.0)
        diversity_weight = 0.75
        for university_quota in (0.33, 0.40):
            with self.subTest(university_quota=university_quota):
                scenario = illustration_scenario(university_quota, diversity_weight)
                selected = analyze_two_criterion_scenario(
                    scenario, cost_distribution
                ).selected_equilibrium
                assert selected is not None
                outcomes = selected.outcomes
                self.assertGreater(
                    outcomes.posterior_evaluation_11, outcomes.posterior_evaluation_01
                )
                self.assertGreater(
                    outcomes.posterior_evaluation_01, outcomes.posterior_evaluation_10
                )
                self.assertGreater(
                    outcomes.posterior_evaluation_10, outcomes.posterior_evaluation_00
                )
                if university_quota < 0.36:
                    self.assertAlmostEqual(outcomes.tutoring_threshold_0, 0.0)
                    self.assertGreater(outcomes.tutoring_threshold_1, 0.0)
                else:
                    self.assertAlmostEqual(outcomes.tutoring_threshold_1, 0.0)
                    self.assertGreater(outcomes.tutoring_threshold_0, 0.0)

    def test_selected_thresholds_rise_with_quota_in_mixed_and_stratified(
        self,
    ) -> None:
        cost_distribution = UniformCostDistribution(upper=1.0)
        mixed_small = analyze_two_criterion_scenario(
            illustration_scenario(0.33, 0.25), cost_distribution
        ).selected_equilibrium
        mixed_large = analyze_two_criterion_scenario(
            illustration_scenario(0.40, 0.25), cost_distribution
        ).selected_equilibrium
        assert mixed_small is not None and mixed_large is not None
        self.assertGreater(
            mixed_large.outcomes.tutoring_threshold_0,
            mixed_small.outcomes.tutoring_threshold_0,
        )
        self.assertGreater(
            mixed_large.outcomes.tutoring_threshold_1,
            mixed_small.outcomes.tutoring_threshold_1,
        )

        stratified_small = analyze_two_criterion_scenario(
            illustration_scenario(0.33, 0.75), cost_distribution
        ).selected_equilibrium
        stratified_large = analyze_two_criterion_scenario(
            illustration_scenario(0.40, 0.75), cost_distribution
        ).selected_equilibrium
        assert stratified_small is not None and stratified_large is not None
        self.assertGreater(
            stratified_large.outcomes.tutoring_threshold_0,
            stratified_small.outcomes.tutoring_threshold_0,
        )

    def test_marginal_stratum_threshold_rises_with_quota_within_a_regime(
        self,
    ) -> None:
        cost_distribution = UniformCostDistribution(upper=1.0)
        diversity_weight = 0.75

        small = analyze_two_criterion_scenario(
            illustration_scenario(0.30, diversity_weight), cost_distribution
        ).selected_equilibrium
        large = analyze_two_criterion_scenario(
            illustration_scenario(0.34, diversity_weight), cost_distribution
        ).selected_equilibrium
        assert small is not None and large is not None

        self.assertAlmostEqual(small.outcomes.tutoring_threshold_0, 0.0)
        self.assertAlmostEqual(large.outcomes.tutoring_threshold_0, 0.0)
        self.assertGreater(small.outcomes.tutoring_threshold_1, 0.0)
        self.assertGreater(
            large.outcomes.tutoring_threshold_1, small.outcomes.tutoring_threshold_1
        )

    def test_unreachable_mixed_tight_quota_is_not_reported(self) -> None:
        scenario = TwoCriterionScenario(
            benefit=1.0,
            university_quota=0.40,
            diversity_weight=0.40,
            underlying_share_00=0.4,
            underlying_share_01=0.1,
            underlying_share_10=0.2,
            underlying_share_11=0.3,
        )
        self.assertGreater(
            scenario.diversity_weight * scenario.underlying_share_11,
            (1.0 - 2.0 * scenario.diversity_weight) * scenario.underlying_share_01,
        )
        analysis = analyze_two_criterion_scenario(
            scenario, UniformCostDistribution(upper=1.0)
        )
        for equilibrium in analysis.equilibria:
            outcomes = equilibrium.outcomes
            mixed_tight = (
                0.0 < outcomes.admission_probability_10 < 1.0
                and 0.0 < outcomes.admission_probability_11 < 1.0
            )
            self.assertFalse(mixed_tight)

    def test_nested_steep_response_stays_directionally_stable(self) -> None:
        cutoff = 0.8
        university_quota = cutoff * (0.3 + 0.7 * cutoff**10)
        analysis = analyze_two_criterion_scenario(
            illustration_scenario(university_quota, 0.0),
            TenthPowerCostDistribution(),
        )
        equilibrium = min(
            analysis.equilibria,
            key=lambda candidate: abs(
                candidate.outcomes.tutoring_threshold_0 - cutoff
            ),
        )
        self.assertAlmostEqual(equilibrium.outcomes.tutoring_threshold_0, cutoff)
        self.assertAlmostEqual(equilibrium.outcomes.tutoring_threshold_1, cutoff)
        self.assertEqual(equilibrium.stability, Stability.STABLE)

    def test_rejects_scenarios_outside_the_model_domain(self) -> None:
        cost_distribution = UniformCostDistribution(upper=1.0)
        invalid_scenarios = (
            replace(illustration_scenario(0.40, 0.0), benefit=0.0),
            replace(illustration_scenario(0.40, 0.0), university_quota=0.0),
            replace(illustration_scenario(0.40, 0.0), university_quota=1.0),
            replace(illustration_scenario(0.40, 0.0), diversity_weight=-0.1),
            replace(illustration_scenario(0.40, 0.0), diversity_weight=1.1),
        )
        for scenario in invalid_scenarios:
            with self.subTest(scenario=scenario), self.assertRaises(ValueError):
                analyze_two_criterion_scenario(scenario, cost_distribution)


class TwoCriterionMonteCarloValidationTests(unittest.TestCase):
    def test_rejects_non_integer_illustration_counts(self) -> None:
        with self.assertRaises(ValueError):
            run_two_criterion_monte_carlo_validation(
                illustration_scenario(0.40, 0.0),
                UniformCostDistribution(upper=1.0),
                population_sizes=(100,),
                trials=1,
                seed=1729,
            )

    def test_nested_validation_recovers_figure_1_continuum_outcomes(self) -> None:
        validation = run_two_criterion_monte_carlo_validation(
            illustration_scenario(0.40, 0.0),
            UniformCostDistribution(upper=1.0),
            population_sizes=(500,),
            trials=200,
            seed=1729,
        )

        summary = validation.population_summaries[0]
        self.assertEqual(summary.population_size, 500)
        self.assertEqual(summary.trials, 200)
        self.assertEqual(summary.regime_agreement_rate, 1.0)
        expected_means = (
            (summary.tutoring_threshold_0.mean, 4 / 7),
            (summary.tutoring_threshold_1.mean, 4 / 7),
            (summary.tutoring_participation_rate_0.mean, 4 / 7),
            (summary.tutoring_participation_rate_1.mean, 4 / 7),
            (summary.population_tutoring_mass.mean, 0.4),
            (summary.admission_probability_10.mean, 4 / 7),
            (summary.admission_probability_11.mean, 4 / 7),
            (summary.admission_probability_00.mean, 0.0),
            (summary.admission_probability_01.mean, 0.0),
            (summary.credibility_0.mean, 3 / 7),
            (summary.credibility_1.mean, 3 / 7),
            (summary.admitted_high_ability_share.mean, 3 / 7),
            (summary.admitted_diversity_share.mean, 0.36),
            (summary.aggregate_tutoring_expenditure.mean, 4 / 35),
        )
        for actual, expected in expected_means:
            with self.subTest(expected=expected):
                self.assertAlmostEqual(actual, expected, delta=0.015)

    def test_repeats_a_seeded_validation_exactly(self) -> None:
        scenario = illustration_scenario(0.40, 0.0)
        cost_distribution = UniformCostDistribution(upper=1.0)
        first = run_two_criterion_monte_carlo_validation(
            scenario,
            cost_distribution,
            population_sizes=(500,),
            trials=5,
            seed=1729,
        )
        second = run_two_criterion_monte_carlo_validation(
            scenario,
            cost_distribution,
            population_sizes=(500,),
            trials=5,
            seed=1729,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.population_summaries[0].population_size, 500)
        self.assertEqual(first.population_summaries[0].trials, 5)

    def test_mixed_validation_recovers_the_selected_continuum_pair(self) -> None:
        diversity_weight = 0.25
        scenario = illustration_scenario(0.33, diversity_weight)
        validation = run_two_criterion_monte_carlo_validation(
            scenario,
            UniformCostDistribution(upper=1.0),
            population_sizes=(500,),
            trials=200,
            seed=1729,
        )
        selected = validation.continuum_equilibrium
        outcomes = selected.outcomes
        summary = validation.population_summaries[0]

        self.assertGreater(outcomes.admission_probability_10, 0.0)
        self.assertLess(outcomes.admission_probability_10, 1.0)
        self.assertGreater(outcomes.admission_probability_11, 0.0)
        self.assertLess(outcomes.admission_probability_11, 1.0)
        self.assertNotAlmostEqual(
            outcomes.admission_probability_10, outcomes.admission_probability_11
        )
        self.assertAlmostEqual(
            outcomes.posterior_evaluation_10, outcomes.posterior_evaluation_11
        )
        self.assertAlmostEqual(
            outcomes.credibility_0 - outcomes.credibility_1,
            diversity_weight / (1.0 - diversity_weight),
        )
        self.assertGreater(summary.regime_agreement_rate, 0.9)
        expected_means = (
            (summary.tutoring_threshold_0.mean, outcomes.tutoring_threshold_0),
            (summary.tutoring_threshold_1.mean, outcomes.tutoring_threshold_1),
            (summary.admission_probability_10.mean, outcomes.admission_probability_10),
            (summary.admission_probability_11.mean, outcomes.admission_probability_11),
            (summary.credibility_0.mean, outcomes.credibility_0),
            (summary.credibility_1.mean, outcomes.credibility_1),
            (
                summary.admitted_high_ability_share.mean,
                outcomes.admitted_high_ability_share,
            ),
            (
                summary.admitted_diversity_share.mean,
                outcomes.admitted_diversity_share,
            ),
        )
        for actual, expected in expected_means:
            with self.subTest(expected=expected):
                self.assertAlmostEqual(actual, expected, delta=0.03)

    def test_nested_validation_uses_one_stage_metric_images(self) -> None:
        scenario = illustration_scenario(0.40, 0.0)
        cost_distribution = UniformCostDistribution(upper=1.0)
        validation = run_two_criterion_monte_carlo_validation(
            scenario,
            cost_distribution,
            population_sizes=(500,),
            trials=50,
            seed=1729,
        )
        one_stage = analyze_scenario(
            one_stage_image(scenario), cost_distribution
        ).selected_equilibrium
        assert one_stage is not None
        outcomes = validation.continuum_equilibrium.outcomes
        summary = validation.population_summaries[0]

        self.assertAlmostEqual(outcomes.tutoring_threshold_0, one_stage.tutoring_cost_cutoff)
        self.assertAlmostEqual(outcomes.tutoring_threshold_1, one_stage.tutoring_cost_cutoff)
        self.assertAlmostEqual(
            outcomes.admission_probability_10, one_stage.high_score_admission_probability
        )
        self.assertAlmostEqual(
            outcomes.admission_probability_11, one_stage.high_score_admission_probability
        )
        self.assertAlmostEqual(
            outcomes.admission_probability_00, one_stage.low_score_admission_probability
        )
        self.assertAlmostEqual(
            outcomes.admission_probability_01, one_stage.low_score_admission_probability
        )
        self.assertAlmostEqual(
            outcomes.credibility_0, one_stage.outcomes.credibility
        )
        self.assertAlmostEqual(
            summary.tutoring_threshold_0.mean,
            summary.tutoring_threshold_1.mean,
            delta=0.02,
        )
        self.assertAlmostEqual(
            summary.admission_probability_10.mean,
            summary.admission_probability_11.mean,
            delta=0.02,
        )

    def test_stratified_validation_recovers_who_competes(self) -> None:
        scenario = illustration_scenario(0.33, 0.75)
        validation = run_two_criterion_monte_carlo_validation(
            scenario,
            UniformCostDistribution(upper=1.0),
            population_sizes=(500,),
            trials=100,
            seed=1729,
        )
        outcomes = validation.continuum_equilibrium.outcomes
        summary = validation.population_summaries[0]

        self.assertAlmostEqual(outcomes.tutoring_threshold_0, 0.0)
        self.assertGreater(outcomes.tutoring_threshold_1, 0.0)
        self.assertAlmostEqual(summary.tutoring_threshold_0.mean, 0.0, delta=0.03)
        self.assertAlmostEqual(
            summary.tutoring_threshold_1.mean,
            outcomes.tutoring_threshold_1,
            delta=0.03,
        )

    def test_accepts_the_illustration_default_population_sizes(self) -> None:
        validation = run_two_criterion_monte_carlo_validation(
            illustration_scenario(0.40, 0.0),
            UniformCostDistribution(upper=1.0),
            population_sizes=(500, 1000),
            trials=2,
            seed=1729,
        )

        self.assertEqual(
            tuple(
                summary.population_size for summary in validation.population_summaries
            ),
            (500, 1000),
        )

    def test_requires_at_least_one_finite_population_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one population size"):
            run_two_criterion_monte_carlo_validation(
                illustration_scenario(0.40, 0.0),
                UniformCostDistribution(upper=1.0),
                population_sizes=(),
                trials=5,
                seed=1729,
            )


if __name__ == "__main__":
    unittest.main()
