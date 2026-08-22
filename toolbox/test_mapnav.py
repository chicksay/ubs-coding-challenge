import unittest
from unittest import mock

from services import mapnav


TOLL_ADJ = {"A": {"B": 4.0, "C": 2.0}, "B": {"D": 3.0}, "C": {"D": 2.0}}
TOLL_TOLLS = {"A": 5.0, "B": 1.0, "C": 9.0, "D": 2.0}

BOUNDED_ADJ = {
    "S": {"X": 1.0, "Z": 10.0},
    "X": {"Y": 1.0},
    "Y": {"D": 1.0},
    "Z": {"D": 1.0},
}
BOUNDED_TOLLS = {"S": 0.0, "X": 0.0, "Y": 0.0, "Z": 0.0, "D": 0.0}

REVISIT_ADJ = {"A": {"B": 1.0}, "B": {"A": 1.0, "D": 1.0}}
REVISIT_TOLLS = {"A": 0.0, "B": 0.0, "D": 0.0}


def _patched(adjacency, tolls):
    mapnav._graphs.clear()
    mapnav._journeys.clear()
    return mock.patch.object(mapnav, "_fetch_graph", return_value=(adjacency, tolls))


class DijkstraTests(unittest.TestCase):
    def setUp(self):
        patcher = _patched(TOLL_ADJ, TOLL_TOLLS)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_picks_cheaper_toll_inclusive_path(self):
        result = mapnav.next_hop({"map_id": "m1", "start": "A", "current": "A", "destination": "D"})
        self.assertEqual(result, "B")

    def test_full_journey_reaches_destination(self):
        node = mapnav.next_hop({"map_id": "m1", "start": "A", "current": "A", "destination": "D"})
        self.assertEqual(node, "B")
        node = mapnav.next_hop({"map_id": "m1", "start": "A", "current": node, "destination": "D"})
        self.assertEqual(node, "D")

    def test_never_returns_non_adjacent_node(self):
        node = mapnav.next_hop({"map_id": "m1", "start": "A", "current": "A", "destination": "D"})
        self.assertIn(node, TOLL_ADJ["A"])


class HopBudgetTests(unittest.TestCase):
    def setUp(self):
        patcher = _patched(BOUNDED_ADJ, BOUNDED_TOLLS)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_unconstrained_choice_needs_three_hops(self):
        node = mapnav.next_hop({"map_id": "m2", "start": "S", "current": "S", "destination": "D"})
        self.assertEqual(node, "X")

    def test_bounded_choice_switches_to_the_only_two_hop_route(self):
        node = mapnav.next_hop(
            {"map_id": "m3", "start": "S", "current": "S", "destination": "D", "hops_remaining": 2}
        )
        self.assertEqual(node, "Z")


class RevisitAvoidanceTests(unittest.TestCase):
    def setUp(self):
        patcher = _patched(REVISIT_ADJ, REVISIT_TOLLS)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_does_not_revisit_the_starting_node(self):
        first = mapnav.next_hop({"map_id": "m4", "start": "A", "current": "A", "destination": "D"})
        self.assertEqual(first, "B")
        second = mapnav.next_hop({"map_id": "m4", "start": "A", "current": first, "destination": "D"})
        self.assertEqual(second, "D")


if __name__ == "__main__":
    unittest.main()
