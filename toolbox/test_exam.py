import unittest

from services import exam


class SelectPassagesTests(unittest.TestCase):
    def test_stays_within_token_budget(self):
        passages = ["word " * 200 for _ in range(20)]
        selected = exam._select_passages("word", passages)
        total = sum(len(exam._ENCODING.encode(p)) for p in selected)
        self.assertLessEqual(total, exam.TOKEN_BUDGET)

    def test_prefers_relevant_passages(self):
        passages = [
            "The sensor grid was last aligned on 14 March.",
            "The cafeteria menu rotates every two weeks.",
        ]
        selected = exam._select_passages("When was the sensor grid aligned?", passages)
        self.assertEqual(selected[0], passages[0])

    def test_skips_oversized_passage_to_fit_smaller_ones(self):
        huge = "sensor grid alignment fact " * 400
        small = "sensor grid alignment fact recorded here"
        selected = exam._select_passages("sensor grid alignment fact", [huge, small])
        self.assertIn(small, selected)
        self.assertNotIn(huge, selected)


class RecallStudyMaterialTests(unittest.TestCase):
    def setUp(self):
        self._original = exam._passages
        exam._passages = [
            "The sensor grid was last aligned on 14 March.",
            "Singapore has an efficient public transit system.",
        ]

    def tearDown(self):
        exam._passages = self._original

    def test_returns_list_of_strings(self):
        result = exam.recall_study_material({"question": "sensor grid alignment"})
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(p, str) for p in result))

    def test_requires_question(self):
        with self.assertRaises(ValueError):
            exam.recall_study_material({})


if __name__ == "__main__":
    unittest.main()
