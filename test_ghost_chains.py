import unittest

from ghost_chains import GhostChainsService


def _tx(tx_id, from_user, to_user, i, **kwargs):
    tx = {
        "txId": tx_id,
        "fromUserId": from_user,
        "toUserId": to_user,
        "amount": 10.0,
        "createdAt": f"2026-06-08T12:{i:02d}:00Z",
    }
    tx.update(kwargs)
    return tx


def _run_chain(edges):
    """Process each edge as its own request, in order; return the list of risk scores."""
    svc = GhostChainsService()
    scores = []
    for i, edge in enumerate(edges):
        from_user, to_user = edge[0], edge[1]
        extra = edge[2] if len(edge) > 2 else {}
        result = svc.process_transactions({"transactions": [_tx(f"tx{i}", from_user, to_user, i, **extra)]})
        scores.append(result["transactions"][0]["riskScore"])
    return scores


class Phase1DocumentedExamplesTests(unittest.TestCase):
    def test_example_1_isolated_scores_zero(self):
        scores = _run_chain([("Meridian", "Apex")])
        self.assertEqual(scores[-1], 0.0)

    def test_example_2_extension(self):
        scores = _run_chain([("Meridian", "Apex"), ("Apex", "Cascade")])
        self.assertEqual(scores[-1], 0.049751093)

    def test_example_3_convergence(self):
        scores = _run_chain([
            ("Meridian", "Apex"),
            ("Meridian", "Horizon"),
            ("Apex", "Sterling"),
            ("Horizon", "Sterling"),
        ])
        self.assertEqual(scores[-1], 0.152311329)

    def test_example_4_return(self):
        scores = _run_chain([
            ("Meridian", "Apex"),
            ("Apex", "Cascade"),
            ("Cascade", "Oakridge"),
            ("Oakridge", "Apex"),
        ])
        self.assertEqual(scores[-1], 0.564647158)

    def test_example_5_multi_loop(self):
        scores = _run_chain([
            ("Meridian", "Apex"),
            ("Apex", "Cascade"),
            ("Cascade", "Meridian"),
            ("Apex", "Nimbus"),
            ("Nimbus", "Meridian"),
        ])
        self.assertEqual(scores[-1], 0.690655806)

    def test_examples_are_strictly_increasing(self):
        ex1 = _run_chain([("Meridian", "Apex")])[-1]
        ex2 = _run_chain([("Meridian", "Apex"), ("Apex", "Cascade")])[-1]
        ex3 = _run_chain([
            ("Meridian", "Apex"), ("Meridian", "Horizon"),
            ("Apex", "Sterling"), ("Horizon", "Sterling"),
        ])[-1]
        ex4 = _run_chain([
            ("Meridian", "Apex"), ("Apex", "Cascade"),
            ("Cascade", "Oakridge"), ("Oakridge", "Apex"),
        ])[-1]
        ex5 = _run_chain([
            ("Meridian", "Apex"), ("Apex", "Cascade"), ("Cascade", "Meridian"),
            ("Apex", "Nimbus"), ("Nimbus", "Meridian"),
        ])[-1]
        self.assertTrue(ex1 < ex2 < ex3 < ex4 < ex5)


class SelfTransferAndIsolationTests(unittest.TestCase):
    def test_self_transfer_always_scores_zero(self):
        scores = _run_chain([("M", "A"), ("A", "C"), ("C", "M"), ("M", "M")])
        self.assertEqual(scores, [0.0, 0.049751093, 0.548485363, 0.0])

    def test_structurally_isolated_transaction_scores_zero_even_with_unrelated_prior_activity(self):
        scores = _run_chain([
            ("Meridian", "Apex"),
            ("Cascade", "Horizon"),
            ("Oakridge", "Sterling"),
        ])
        self.assertEqual(scores, [0.0, 0.0, 0.0])


class IdempotencyAndValidationTests(unittest.TestCase):
    def test_duplicate_identical_payload_returns_original_score_no_mutation(self):
        svc = GhostChainsService()
        tx = _tx("dup", "A", "B", 0)
        first = svc.process_transactions({"transactions": [tx]})
        second = svc.process_transactions({"transactions": [tx]})
        self.assertEqual(first, second)

    def test_missing_optional_identity_fields_do_not_fail_processing(self):
        svc = GhostChainsService()
        result = svc.process_transactions({"transactions": [_tx("a", "A", "B", 0)]})
        self.assertEqual(result["transactions"][0]["riskScore"], 0.0)


class ResetTests(unittest.TestCase):
    def test_clear_restores_startup_equivalent_state(self):
        svc = GhostChainsService()
        svc.process_transactions({"transactions": [_tx("a", "A", "B", 0)]})
        svc.clear({"clearTransactions": True})
        result = svc.process_transactions({"transactions": [_tx("b", "B", "C", 0)]})
        self.assertEqual(result["transactions"][0]["riskScore"], 0.0)


class Phase2IdentityScoringTests(unittest.TestCase):
    def test_example_1_consistent_identity_chain(self):
        scores = _run_chain([
            ("Meridian", "Apex", {"deviceId": "dev_ios_7f3a91"}),
            ("Apex", "Cascade", {"deviceId": "dev_ios_7f3a91"}),
            ("Cascade", "Horizon", {"deviceId": "dev_ios_7f3a91"}),
        ])
        self.assertEqual(scores, [0.0, 0.132739289, 0.152765003])

    def test_example_2_identity_divergence_under_branching(self):
        scores = _run_chain([
            ("Meridian", "Apex", {"deviceId": "dev_ios_7f3a91"}),
            ("Apex", "Cascade", {"deviceId": "dev_ios_7f3a91"}),
            ("Apex", "Sterling", {"deviceId": "dev_ios_7f3a91"}),
            ("Cascade", "Oakridge", {"deviceId": "dev_android_c2e4b8"}),
        ])
        self.assertEqual(scores, [0.0, 0.132739289, 0.15172507, 0.074655924])
        self.assertLess(scores[3], scores[2])  # diverging branch loses the consistency bonus

    def test_example_3_identity_shift_mid_flow(self):
        scores = _run_chain([
            ("Meridian", "Apex", {"deviceId": "dev_ios_7f3a91"}),
            ("Apex", "Cascade", {"deviceId": "dev_ios_7f3a91"}),
            ("Cascade", "Horizon", {"deviceId": "dev_android_c2e4b8"}),
            ("Horizon", "Nimbus", {"deviceId": "dev_android_c2e4b8"}),
        ])
        self.assertEqual(scores, [0.0, 0.132739289, 0.074655924, 0.163309685])

    def test_example_4_shared_identity_across_disconnected_components(self):
        scores = _run_chain([
            ("Meridian", "Apex", {"ipAddress": "10.0.0.1"}),
            ("Cascade", "Horizon", {"ipAddress": "10.0.0.1"}),
            ("Oakridge", "Sterling", {"ipAddress": "10.0.0.1"}),
        ])
        self.assertEqual(scores, [0.0, 0.223743725, 0.270047044])
        self.assertGreater(scores[2], scores[1])  # reused by more external components now

    def test_disconnected_transactions_without_shared_identity_still_score_zero(self):
        scores = _run_chain([
            ("Meridian", "Apex"),
            ("Cascade", "Horizon"),
            ("Oakridge", "Sterling"),
        ])
        self.assertEqual(scores, [0.0, 0.0, 0.0])

    def test_missing_identity_after_it_was_present_on_connected_flow_is_suspicious(self):
        scores = _run_chain([
            ("Meridian", "Apex", {"deviceId": "dev_ios_7f3a91"}),
            ("Apex", "Cascade", {"deviceId": "dev_ios_7f3a91"}),
            ("Cascade", "Horizon", {}),  # device dropped after 2 consistent legs
        ])
        self.assertEqual(scores, [0.0, 0.132739289, 0.354102862])

    def test_isolated_transaction_with_identity_but_no_prior_context_scores_zero(self):
        scores = _run_chain([("M", "A", {"deviceId": "dev_x"})])
        self.assertEqual(scores[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
