import unittest

from routes import app
import toolbox  # noqa: F401 - registers /mcp


class ServerSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_tools_list_returns_phase1_tools(self):
        response = self.client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )
        names = {tool["name"] for tool in response.get_json()["result"]["tools"]}
        self.assertEqual(
            names,
            {"get_name", "calculate", "identify_shape", "order_of_operations", "count_sides", "recall_study_material", "next_hop"},
        )

    def test_calculate_tool_call_round_trip(self):
        response = self.client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "calculate", "arguments": {"expression": "2 + 3 * 4"}},
            },
        )
        result = response.get_json()["result"]
        self.assertEqual(result["structuredContent"]["answer"], 14)


if __name__ == "__main__":
    unittest.main()
