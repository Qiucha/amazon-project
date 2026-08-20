import json
from pathlib import Path
import unittest

import matplotlib


matplotlib.use("Agg")


class PrimaryNotebookTests(unittest.TestCase):
    def test_primary_notebook_is_complete_and_executable(self) -> None:
        notebook_path = (
            Path(__file__).parents[1]
            / "notebooks"
            / "one_stage_admissions_simulation.ipynb"
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
            "Paper Figure 1 reproduction",
            "Notes scenario (secondary)",
            "Finite-population validation",
        ):
            with self.subTest(required_section=required_section):
                self.assertIn(required_section, markdown)

        namespace = {"__name__": "__notebook_test__"}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), str(notebook_path), "exec"), namespace)

        self.assertEqual(set(namespace["paper_analyses"]), {0.33, 0.40})
        self.assertEqual(set(namespace["notes_analyses"]), {0.40, 0.50})
        self.assertTrue(namespace["validation_results"])


if __name__ == "__main__":
    unittest.main()
