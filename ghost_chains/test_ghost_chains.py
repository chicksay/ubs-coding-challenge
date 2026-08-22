import unittest

from ghost_chains import GhostChainsService


def _tx(tx_id, from_user, to_user, i, **kwargs):
    # amount defaults to a flat 10.0 on every call, so consecutive edges that
    # don't override it share ratio 1.0 -- zero deviation, hence zero value
    # signal either way (see _value_mass). Pre-Phase-3 pinned scores below
    # are therefore unaffected by amount unless a test overrides it.
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
        self.assertEqual(scores[-1], 0.082828427)

    def test_example_3_convergence(self):
        scores = _run_chain([
            ("Meridian", "Apex"),
            ("Meridian", "Horizon"),
            ("Apex", "Sterling"),
            ("Horizon", "Sterling"),
        ])
        self.assertEqual(scores[-1], 0.170996945)

    def test_example_4_return(self):
        scores = _run_chain([
            ("Meridian", "Apex"),
            ("Apex", "Cascade"),
            ("Cascade", "Oakridge"),
            ("Oakridge", "Apex"),
        ])
        self.assertEqual(scores[-1], 0.6395935)

    def test_example_5_multi_loop(self):
        scores = _run_chain([
            ("Meridian", "Apex"),
            ("Apex", "Cascade"),
            ("Cascade", "Meridian"),
            ("Apex", "Nimbus"),
            ("Nimbus", "Meridian"),
        ])
        self.assertEqual(scores[-1], 0.761169318)

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


class DeepChainCoherenceTests(unittest.TestCase):
    """Structural Consistency requires coherent behavior across structurally
    related scenarios, not just the five short documented examples. A long
    linear extension chain is the simplest structurally-related-but-untested
    scenario the spec's own Example 2 implies -- and previously exposed a
    real bug: MAX_DEPTH=6 caused every hop past #6 to score bit-for-bit
    identically (the walk stopped discovering ancestors rather than letting
    DECAY's own falloff make them negligible), meaning an 8-hop chain and a
    20-hop chain were indistinguishable. MAX_DEPTH=25 fixes this by making
    the exponential decay itself the reason distant nodes stop mattering."""

    def test_long_extension_chain_never_plateaus_to_an_identical_score(self):
        chain = [(f"N{i}", f"N{i + 1}") for i in range(15)]
        scores = _run_chain(chain)
        consecutive_pairs = list(zip(scores, scores[1:]))
        self.assertTrue(
            all(a != b for a, b in consecutive_pairs[5:]),
            f"scores repeated identically somewhere past hop 5: {scores}",
        )

    def test_long_extension_chain_keeps_increasing(self):
        chain = [(f"N{i}", f"N{i + 1}") for i in range(15)]
        scores = _run_chain(chain)
        self.assertTrue(
            all(a < b for a, b in zip(scores, scores[1:])),
            f"score did not strictly increase somewhere: {scores}",
        )


class SelfTransferAndIsolationTests(unittest.TestCase):
    def test_self_transfer_always_scores_zero(self):
        scores = _run_chain([("M", "A"), ("A", "C"), ("C", "M"), ("M", "M")])
        self.assertEqual(scores, [0.0, 0.082828427, 0.623990058, 0.0])

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


class TemporalSpacingConsistencyTests(unittest.TestCase):
    def test_return_loop_score_decreases_as_the_loop_takes_longer_to_close_within_window(self):
        # Diagnosis finding (ghost_chains_diagnosis.txt, experiment A): a loop
        # that snaps shut in minutes is a stronger layering signal than an
        # otherwise-identical loop that takes most of the 24h window to
        # close, so spacing is now expected to matter -- this replaces the
        # old "spacing is irrelevant" invariant this test used to assert.
        tight = _run_chain([
            ("Meridian", "Apex"), ("Apex", "Cascade"),
            ("Cascade", "Oakridge"), ("Oakridge", "Apex"),
        ])[-1]
        spread = _run_chain([
            ("Meridian", "Apex"), ("Apex", "Cascade"),
            ("Cascade", "Oakridge", {"createdAt": "2026-06-08T23:40:00Z"}),
            ("Oakridge", "Apex", {"createdAt": "2026-06-09T11:45:00Z"}),
        ])[-1]
        self.assertEqual(tight, 0.6395935)
        self.assertEqual(spread, 0.57494527)
        self.assertLess(spread, tight)


class WindowBoundaryTests(unittest.TestCase):
    def test_cycle_ignored_once_oldest_edge_reaches_the_24h_cutoff(self):
        # Age == 24h exactly is already expired ("active means age < 24h"),
        # so the closing edge sees no surviving predecessor -- just an
        # isolated new edge, scoring 0.0 like Example 1.
        scores = _run_chain([
            ("A", "B", {"createdAt": "2026-06-08T12:00:00Z"}),
            ("B", "A", {"createdAt": "2026-06-09T12:00:00Z"}),
        ])
        self.assertEqual(scores[-1], 0.0)

    def test_cycle_still_detected_one_second_before_the_24h_cutoff(self):
        scores = _run_chain([
            ("A", "B", {"createdAt": "2026-06-08T12:00:00Z"}),
            ("B", "A", {"createdAt": "2026-06-09T11:59:59Z"}),
        ])
        self.assertEqual(scores[-1], 0.576189309)


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
        self.assertEqual(scores, [0.0, 0.210775591, 0.239879901])

    def test_example_2_identity_divergence_under_branching(self):
        scores = _run_chain([
            ("Meridian", "Apex", {"deviceId": "dev_ios_7f3a91"}),
            ("Apex", "Cascade", {"deviceId": "dev_ios_7f3a91"}),
            ("Apex", "Sterling", {"deviceId": "dev_ios_7f3a91"}),
            ("Cascade", "Oakridge", {"deviceId": "dev_android_c2e4b8"}),
        ])
        self.assertEqual(scores, [0.0, 0.210775591, 0.226805605, 0.122515722])
        self.assertLess(scores[3], scores[2])  # diverging branch loses the consistency bonus

    def test_example_3_identity_shift_mid_flow(self):
        scores = _run_chain([
            ("Meridian", "Apex", {"deviceId": "dev_ios_7f3a91"}),
            ("Apex", "Cascade", {"deviceId": "dev_ios_7f3a91"}),
            ("Cascade", "Horizon", {"deviceId": "dev_android_c2e4b8"}),
            ("Horizon", "Nimbus", {"deviceId": "dev_android_c2e4b8"}),
        ])
        self.assertEqual(scores, [0.0, 0.210775591, 0.122515722, 0.254943322])

    def test_example_4_shared_identity_across_disconnected_components(self):
        scores = _run_chain([
            ("Meridian", "Apex", {"ipAddress": "10.0.0.1"}),
            ("Cascade", "Horizon", {"ipAddress": "10.0.0.1"}),
            ("Oakridge", "Sterling", {"ipAddress": "10.0.0.1"}),
        ])
        self.assertEqual(scores, [0.0, 0.337936699, 0.397873092])
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
        self.assertEqual(scores, [0.0, 0.210775591, 0.499201239])

    def test_isolated_transaction_with_identity_but_no_prior_context_scores_zero(self):
        scores = _run_chain([("M", "A", {"deviceId": "dev_x"})])
        self.assertEqual(scores[-1], 0.0)


class Phase3ValueScoringTests(unittest.TestCase):
    def _value_examples(self):
        """The four Phase 3 documented examples' final-tx scores, by name."""
        ex1 = _run_chain([
            ("Meridian", "Apex", {"amount": 10000.0}),
            ("Apex", "Cascade", {"amount": 9910.0}),
            ("Cascade", "Horizon", {"amount": 9820.81}),
            ("Horizon", "Nimbus", {"amount": 9732.42}),
        ])[-1]
        ex2 = _run_chain([
            ("Meridian", "Apex", {"amount": 10000.0}),
            ("Apex", "Cascade", {"amount": 9800.0}),
            ("Apex", "Sterling", {"amount": 5000.0}),
            ("Cascade", "Horizon", {"amount": 9700.0}),
            ("Sterling", "Oakridge", {"amount": 4900.0}),
        ])[-1]
        ex3 = _run_chain([
            ("Meridian", "Apex", {"amount": 10000.0}),
            ("Apex", "Cascade", {"amount": 9950.0}),
            ("Cascade", "Horizon", {"amount": 9800.0}),
            ("Horizon", "Nimbus", {"amount": 9950.0}),  # reverses the prior decay
        ])[-1]
        ex4 = _run_chain([
            ("Meridian", "Apex", {"amount": 10000.0}),
            ("Apex", "Cascade", {"amount": 9800.0}),
            ("Apex", "Sterling", {"amount": 5000.0}),
            ("Cascade", "Horizon", {"amount": 9700.0}),
            ("Sterling", "Horizon", {"amount": 4950.0}),
        ])[-1]
        return ex1, ex2, ex3, ex4

    def test_example_3_reversal_outranks_example_1_consistent_decay(self):
        # Same linear 4-hop shape in both chains; only the amount trajectory differs.
        ex1, _, ex3, _ = self._value_examples()
        self.assertGreater(ex3, ex1)

    def test_example_1_is_lowest_of_the_four_value_examples(self):
        # Consistent decay along a single path is the characteristic
        # layering pattern, not a deviation from it -- lowest of the four.
        ex1, ex2, ex3, ex4 = self._value_examples()
        self.assertLess(ex1, ex2)
        self.assertLess(ex1, ex3)
        self.assertLess(ex1, ex4)

    def test_example_3_is_highest_of_the_four_value_examples(self):
        # A trajectory reversal against structural continuity is a direct
        # contradiction -- highest of the four.
        ex1, ex2, ex3, ex4 = self._value_examples()
        self.assertGreater(ex3, ex1)
        self.assertGreater(ex3, ex2)
        self.assertGreater(ex3, ex4)

    def test_divergent_branch_is_unaffected_by_a_sibling_branchs_amount(self):
        # Example 2: Cascade -> Horizon is scored purely against its own
        # predecessor (Apex -> Cascade). Whether or not the sibling
        # Apex -> Sterling branch exists alongside it must not change that
        # score -- no blind aggregation across unrelated branches.
        with_sibling_branch = _run_chain([
            ("Meridian", "Apex", {"amount": 10000.0}),
            ("Apex", "Cascade", {"amount": 9800.0}),
            ("Apex", "Sterling", {"amount": 5000.0}),
            ("Cascade", "Horizon", {"amount": 9700.0}),
        ])[-1]
        without_sibling_branch = _run_chain([
            ("Meridian", "Apex", {"amount": 10000.0}),
            ("Apex", "Cascade", {"amount": 9800.0}),
            ("Cascade", "Horizon", {"amount": 9700.0}),
        ])[-1]
        self.assertEqual(with_sibling_branch, without_sibling_branch)

    def test_convergent_predecessors_each_contribute_independently(self):
        # Example 4-style convergence: once Horizon Capital has received from
        # two independent branches, a later outgoing edge from Horizon that
        # reverses against both is stronger evidence than reversing against
        # just one -- corroborating predecessors accumulate, they don't dilute.
        one_predecessor = _run_chain([
            ("Cascade", "Horizon", {"amount": 9700.0}),
            ("Horizon", "Nimbus", {"amount": 9850.0}),
        ])[-1]
        two_predecessors = _run_chain([
            ("Cascade", "Horizon", {"amount": 9700.0}),
            ("Sterling", "Horizon", {"amount": 4950.0}),
            ("Horizon", "Nimbus", {"amount": 9850.0}),
        ])[-1]
        self.assertGreater(two_predecessors, one_predecessor)

    def test_isolated_transaction_with_amount_but_no_prior_context_has_no_value_signal(self):
        scores = _run_chain([("M", "A", {"amount": 12345.0})])
        self.assertEqual(scores[-1], 0.0)


class CrossSignalMonotonicityTests(unittest.TestCase):
    """Each signal type should add to the score, not get lost or cancelled
    out when combined with another. No ground-truth score exists for these
    (the spec's own Cross-Signal Examples explicitly give no expected
    ordering), so each test holds topology fixed and toggles exactly one
    signal on/off, checking the score moves the direction that signal alone
    is already known to move it."""

    def test_structural_and_identity_signals_combine(self):
        # Example 4 shape (a closed return loop). Matching device identity on
        # every hop -- especially the closing edge against its immediate
        # predecessor -- should add to the cycle's structural mass, not be
        # ignored once a cycle is already present.
        baseline = _run_chain([
            ("Meridian", "Apex"), ("Apex", "Cascade"),
            ("Cascade", "Oakridge"), ("Oakridge", "Apex"),
        ])[-1]
        with_identity = _run_chain([
            ("Meridian", "Apex", {"deviceId": "dev_x"}),
            ("Apex", "Cascade", {"deviceId": "dev_x"}),
            ("Cascade", "Oakridge", {"deviceId": "dev_x"}),
            ("Oakridge", "Apex", {"deviceId": "dev_x"}),
        ])[-1]
        self.assertGreater(with_identity, baseline)

    def test_structural_and_value_signals_combine(self):
        # Phase 3 spec's own "Phase 1 and Phase 3" cross example: a return
        # path Apex -> Cascade -> Horizon -> Apex, closing edge amount 9850
        # exceeds the preceding leg's 9700 -- a reversal on top of a cycle.
        consistent_decay = _run_chain([
            ("Meridian", "Apex", {"amount": 10000.0}),
            ("Apex", "Cascade", {"amount": 9800.0}),
            ("Cascade", "Horizon", {"amount": 9700.0}),
            ("Horizon", "Apex", {"amount": 9650.0}),
        ])[-1]
        reversal = _run_chain([
            ("Meridian", "Apex", {"amount": 10000.0}),
            ("Apex", "Cascade", {"amount": 9800.0}),
            ("Cascade", "Horizon", {"amount": 9700.0}),
            ("Horizon", "Apex", {"amount": 9850.0}),  # matches spec example exactly
        ])[-1]
        self.assertGreater(reversal, consistent_decay)

    def test_identity_and_value_signals_combine_at_convergence(self):
        # Phase 3 spec's own "Phase 2 and Phase 3" cross example: two chains
        # sharing ipAddress 10.0.0.1 converge at Nimbus Trading; the closing
        # edge both switches to a different ipAddress AND reverses the
        # amount (10100 > the converging predecessor's 10000).
        same_ip_consistent_decay = _run_chain([
            ("Meridian", "Apex", {"amount": 10000.0, "ipAddress": "10.0.0.1"}),
            ("Cascade", "Horizon", {"amount": 10000.0, "ipAddress": "10.0.0.1"}),
            ("Apex", "Nimbus", {"amount": 9800.0, "ipAddress": "10.0.0.1"}),
            ("Horizon", "Nimbus", {"amount": 9700.0, "ipAddress": "10.0.0.1"}),
        ])[-1]
        different_ip_reversal = _run_chain([
            ("Meridian", "Apex", {"amount": 10000.0, "ipAddress": "10.0.0.1"}),
            ("Cascade", "Horizon", {"amount": 10000.0, "ipAddress": "10.0.0.1"}),
            ("Apex", "Nimbus", {"amount": 9800.0, "ipAddress": "10.0.0.1"}),
            ("Horizon", "Nimbus", {"amount": 10100.0, "ipAddress": "10.0.0.2"}),  # matches spec example
        ])[-1]
        self.assertGreater(different_ip_reversal, same_ip_consistent_decay)


if __name__ == "__main__":
    unittest.main()
