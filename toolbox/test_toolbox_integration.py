import unittest
from unittest import mock

from routes import app
import toolbox  # noqa: F401 - registers /mcp
from services import exam, mapnav


class ToolboxIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_tools_list_includes_all_seven_tools(self):
        response = self.client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )
        names = {tool["name"] for tool in response.get_json()["result"]["tools"]}
        self.assertEqual(
            names,
            {
                "get_name",
                "calculate",
                "identify_shape",
                "order_of_operations",
                "count_sides",
                "recall_study_material",
                "next_hop",
            },
        )

    def test_recall_study_material_tool_call_returns_list_answer(self):
        original = exam._passages
        exam._passages = ["The sensor grid was last aligned on 14 March."]
        try:
            response = self.client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "recall_study_material",
                        "arguments": {"question": "sensor grid alignment"},
                    },
                },
            )
            result = response.get_json()["result"]
            self.assertIsInstance(result["structuredContent"]["answer"], list)
            self.assertIn("sensor grid", result["content"][0]["text"])
        finally:
            exam._passages = original

    def test_next_hop_tool_call_returns_adjacent_node(self):
        adjacency = {"A": {"B": 1.0}, "B": {"D": 1.0}}
        tolls = {"A": 0.0, "B": 0.0, "D": 0.0}
        mapnav._graphs.clear()
        mapnav._journeys.clear()
        with mock.patch.object(mapnav, "_fetch_graph", return_value=(adjacency, tolls)):
            response = self.client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "next_hop",
                        "arguments": {
                            "map_id": "m1",
                            "start": "A",
                            "current": "A",
                            "destination": "D",
                        },
                    },
                },
            )
            result = response.get_json()["result"]
            self.assertEqual(result["structuredContent"]["answer"], "B")


if __name__ == "__main__":
    unittest.main()
