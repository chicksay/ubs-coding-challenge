"""End-to-end smoke test for the Ghost Chains HTTP endpoints as actually
served by app.py -- not GhostChainsService imported directly.

Every test in ghost_chains/test_ghost_chains.py exercises GhostChainsService
in-process; none of them go through app.py's HTTP handler at all. That gap
matters: a wiring bug (wrong route, a response-shape mismatch, an exception
silently swallowed) would pass every one of those 32 tests while still
breaking the actual served endpoint. This file starts the real
ThreadingHTTPServer from app.py on an ephemeral port and hits it with real
HTTP requests, stdlib only (app.py's own convention -- no flask/requests
dependency for this side of the project).
"""
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import app as app_module


class GhostChainsServerSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), app_module.Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        status, body = self._post("/ghost-chains/reset", {"clearTransactions": True})
        assert (status, body) == (200, {"clearTransactions": True}), (status, body)

    def _url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def _get(self, path):
        with urllib.request.urlopen(self._url(path)) as resp:
            return resp.status, json.loads(resp.read())

    def _post(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url(path), data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_health_check(self):
        status, body = self._get("/ghost-chains/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok"})

    def test_reset_returns_exact_spec_shape(self):
        status, body = self._post("/ghost-chains/reset", {"clearTransactions": True})
        self.assertEqual(status, 200)
        self.assertEqual(body, {"clearTransactions": True})

    def test_spec_getting_started_smoke_example(self):
        # The exact request from ghost_chains.txt's own "Getting Started" curl example.
        status, body = self._post("/ghost-chains/transactions", {
            "transactions": [{
                "txId": "tx_meridian_001",
                "fromUserId": "meridian_holdings",
                "toUserId": "apex_logistics",
                "amount": 370.0,
                "createdAt": "2026-06-08T12:00:00Z",
            }],
        })
        self.assertEqual(status, 200)
        self.assertEqual(
            body, {"transactions": [{"txId": "tx_meridian_001", "riskScore": 0.0}]},
        )

    def test_batch_preserves_order_and_reaches_the_real_scorer(self):
        # Phase 1 Example 4 (Return), run entirely through the HTTP layer --
        # confirms the served endpoint reaches the actual tuned scorer, not
        # just that GhostChainsService behaves when imported directly in a
        # unit test. 0.564647158 must match ghost_chains/test_ghost_chains.py's
        # pinned value for this same example -- if it doesn't, the HTTP layer
        # and the unit-tested module have drifted apart.
        status, body = self._post("/ghost-chains/transactions", {"transactions": [
            {"txId": "e4_1", "fromUserId": "Meridian", "toUserId": "Apex", "amount": 10.0, "createdAt": "2026-06-08T12:00:00Z"},
            {"txId": "e4_2", "fromUserId": "Apex", "toUserId": "Cascade", "amount": 10.0, "createdAt": "2026-06-08T12:01:00Z"},
            {"txId": "e4_3", "fromUserId": "Cascade", "toUserId": "Oakridge", "amount": 10.0, "createdAt": "2026-06-08T12:02:00Z"},
            {"txId": "e4_4", "fromUserId": "Oakridge", "toUserId": "Apex", "amount": 10.0, "createdAt": "2026-06-08T12:03:00Z"},
        ]})
        self.assertEqual(status, 200)
        self.assertEqual(
            [t["txId"] for t in body["transactions"]], ["e4_1", "e4_2", "e4_3", "e4_4"],
        )
        scores = [t["riskScore"] for t in body["transactions"]]
        self.assertEqual(scores[-1], 0.564647158)

    def test_duplicate_txid_returns_original_score_via_http(self):
        tx = {"txId": "dup1", "fromUserId": "A", "toUserId": "B", "amount": 10.0, "createdAt": "2026-06-08T12:00:00Z"}
        _, first = self._post("/ghost-chains/transactions", {"transactions": [tx]})
        _, second = self._post("/ghost-chains/transactions", {"transactions": [tx]})
        self.assertEqual(first, second)

    def test_reset_between_requests_actually_clears_state_via_http(self):
        self._post("/ghost-chains/transactions", {"transactions": [
            {"txId": "a", "fromUserId": "X", "toUserId": "Y", "amount": 10.0, "createdAt": "2026-06-08T12:00:00Z"},
        ]})
        self._post("/ghost-chains/reset", {"clearTransactions": True})
        _, body = self._post("/ghost-chains/transactions", {"transactions": [
            {"txId": "b", "fromUserId": "Y", "toUserId": "X", "amount": 10.0, "createdAt": "2026-06-08T12:00:00Z"},
        ]})
        # X->Y no longer exists post-reset, so Y->X must be a fresh isolated edge.
        self.assertEqual(body["transactions"][0]["riskScore"], 0.0)

    def test_malformed_json_body_returns_400_not_a_crash(self):
        req = urllib.request.Request(
            self._url("/ghost-chains/transactions"), data=b"not json",
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            urllib.request.urlopen(req)
            self.fail("expected an HTTPError for malformed JSON")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)

    def test_missing_optional_identity_fields_do_not_fail_via_http(self):
        status, body = self._post("/ghost-chains/transactions", {"transactions": [
            {"txId": "no_opts", "fromUserId": "A", "toUserId": "B", "amount": 10.0, "createdAt": "2026-06-08T12:00:00Z"},
        ]})
        self.assertEqual(status, 200)
        self.assertEqual(body["transactions"][0]["riskScore"], 0.0)


if __name__ == "__main__":
    unittest.main()
