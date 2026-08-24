import json
from pathlib import Path
import unittest

import matplotlib


matplotlib.use("Agg")


class TwoCriterionNotebookTests(unittest.TestCase):
    def test_two_criterion_notebook_is_complete_and_executable(self) -> None:
        notebook_path = (
            Path(__file__).parents[1] / "two_criterion_admissions_simulation.ipynb"
        )
        notebook = json.loads(notebook_path.read_text())

        markdown = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        for required_section in (
            "Authoritative parameters",
            "Metric definitions",
            "Nested $w=0$ recovery",
            "Every equilibrium",
            "Stratified rat race",
            "Mixed tight-quota",
            "Finite-population validation",
            "Reproduction checklist",
        ):
            with self.subTest(required_section=required_section):
                self.assertIn(required_section, markdown)

        self.assertLess(
            markdown.find("Nested $w=0$ recovery"),
            markdown.find("Stratified rat race"),
        )
        self.assertLess(
            markdown.find("Nested $w=0$ recovery"),
            markdown.find("Mixed tight-quota"),
        )

        namespace = {"__name__": "__notebook_test__"}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec(
                    compile("".join(cell["source"]), str(notebook_path), "exec"),
                    namespace,
                )

        self.assertEqual(namespace["QUOTAS"], (0.33, 0.40))
        self.assertEqual(namespace["WEIGHTS"], (0.0, 0.25, 0.75))
        self.assertEqual(namespace["MC_POPULATION_SIZES"], (500, 1000))
        self.assertEqual(
            set(namespace["analyses"]),
            {(0.0, 0.33), (0.0, 0.40), (0.25, 0.33), (0.25, 0.40), (0.75, 0.33), (0.75, 0.40)},
        )
        nested = namespace["analyses"][0.0, 0.40].selected_equilibrium
        self.assertIsNotNone(nested)
        self.assertAlmostEqual(nested.outcomes.tutoring_threshold_0, 4 / 7)
        self.assertAlmostEqual(nested.outcomes.tutoring_threshold_1, 4 / 7)
        self.assertTrue(namespace["validation_results"])


if __name__ == "__main__":
    unittest.main()
