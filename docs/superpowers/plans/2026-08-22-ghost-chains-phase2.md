# Ghost Chains Phase 2 (Identity Signal) Implementation Plan

> **Status: implemented.** This plan originally targeted the BFS-distance-counting scorer from earlier in the day. That scorer was fully replaced (commits `40ded72`..`fcf5b4a`) with a decay-weighted mass-propagation engine before Phase 2 work started, so the plan below was rewritten to match the engine that actually shipped, and documents what was built rather than a forward task queue.

> **For agentic workers:** if extending this further, REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking; this plan's own steps are checked off as already done.

**Goal:** Add Phase 2's identity signal (`ipAddress` / `deviceId`) to the Ghost Chains scoring service, combined with the existing structural signal, without regressing any Phase 1 behavior.

**Architecture:** `ghost_chains.py` scores each transaction as a single decay-weighted "mass" value, converted to `[0, 1)` via `1 - (1 + mass/SATURATION)^-TAIL`. Mass is built from additive terms — new-reachability pairs, extra routes, fan-in/fan-out, and cycle mass — each computed from depth-limited BFS walks (`_walk`) over the live `_adj`/`_rev` adjacency, decayed by `DECAY ** depth` so short structures dominate long ones without ever enumerating individual node pairs. Identity signal is added as one more additive term in that same sum (`_identity_mass`), reusing the walks `_score` already computes and the already-24h-pruned `self._active` dict as its only data source — no new state, no new pruning logic, no second scoring system bolted on.

**Tech Stack:** Python stdlib only (no new dependencies — `requirements.txt` is empty; the engine uses `bisect`, `heapq`, `threading`, `collections.defaultdict`; tests use `unittest`, not `pytest`, since `pytest` isn't installed).

**Spec:** `ghost_chains.txt` (Phase 1, still fully in force) and `ghost_chains_phase2.txt` (Phase 2, this plan's target).

## Global Constraints

- Score range: `0.0 ≤ riskScore < 1.0` in practice (the `1 - (...)^-TAIL` map asymptotically approaches but never reaches 1.0 — still within the spec's `[0.0, 1.0]`).
- Lookback window `W = 24 hours`: an edge is active while its age is strictly less than 24h; `_expire` deletes expired edges from `_adj`/`_rev`/`_edge_times` (and now `_active`, which identity scoring depends on) so nothing stale can influence a later score.
- Idempotency: duplicate `txId` returns the original score unconditionally; a differing payload on the same `txId` logs a warning (`_process_one`, `ghost_chains.py:369-373`) but does not mutate state or re-score — the spec leaves the exact handling open, and this is the chosen interpretation.
- Optional fields `ipAddress` / `deviceId` may be absent on any transaction; absence must never cause a processing failure (nor did it before Phase 2 touched anything — `_process_one` never even read them).
- Unknown/unrecognised fields are ignored by construction (`_process_one` only reads the fields it needs from `raw`).
- `clear()` fully resets to a fresh `GhostChainsService()`-equivalent state.
- Ordering: within one request, transactions are processed sequentially in array order (`process_batch`); across requests, arrival order defines state evolution via the monotonically-advancing `self._clock`, not `createdAt` order.
- Memory usage is bounded by the active window: `_active`, `_adj`, `_rev`, `_edge_times` are all pruned together by `_expire`.
- A Phase 2 evaluation re-tests all Phase 1 requirements in the same run — nothing here may regress the five Phase 1 documented examples or the self-transfer / isolated-transaction edge cases the engine already handles correctly.
- Per the spec: *"Systems built on principled graph models are expected to outperform implementations tuned to specific patterns"* — identity scoring stays a general rule (component membership via the existing walks, direct-predecessor/successor consistency), not a hardcoded per-example special case.

## File Structure

- **Modified: `ghost_chains.py`** — single file, no split needed. Three additive changes: (1) three new module constants (`W_IDENTITY_REUSE`, `W_IDENTITY_CONSISTENCY`, `W_IDENTITY_EVASION`), (2) a new `_identity_mass` method, (3) `_score`/`_process_one`/`_active` threaded through with `ip`/`device` alongside the existing `src`/`dst`/`created`.
- **Created: `test_ghost_chains.py`** — the project had zero automated tests before this work (the original file written earlier today targeted the now-replaced scorer and was rewritten from scratch against the actual shipped engine). 18 tests: five Phase 1 documented examples plus strict ordering, self-transfer, the disconnected-isolation edge case, idempotency, reset, and all four Phase 2 examples plus the evasion case.

---

### Task 1: Add identity signal to the mass-based scorer — DONE

**Files:**
- Modified: `ghost_chains.py:51-56` (constants), `ghost_chains.py:221` (`_score` signature), `ghost_chains.py:282-290` (wiring into `mass`), `ghost_chains.py:302-346` (`_identity_mass`), `ghost_chains.py:376-417` (`_process_one` reading/threading `ip`/`device`)
- Created: `test_ghost_chains.py`

**Design (from the Phase 2 Core Principle):**

For each of `ip_address`/`ipAddress` and `device_id`/`deviceId` independently (*"When both are present, treat them as independent dimensions"*, handled by the `for attr, value in (("ip", ip), ("device", device))` loop in `_identity_mass`):

1. **Disconnected reuse** (Example 4): `local_component` is the union of all four BFS walks `_score` already runs (`upstream`, `downstream`, `src_reaches`, `reaches_dst`) plus `{src, dst}` — an approximation of "everything locally visible from this edge," bounded the same way the rest of the engine is bounded (`MAX_DEPTH`/`MAX_VISIT`). Scanning `self._active.values()` for the same identity value, any user *outside* that component is genuinely disconnected reuse — weighted `W_IDENTITY_REUSE * (1 - DECAY ** external_count)`, so it saturates like every other decay-based term in this engine rather than growing unboundedly.
2. **Path-local consistency** (Example 1, the non-diverging branches in 2/3): a direct predecessor edge into `src` or successor edge from `dst` carrying the same value adds `W_IDENTITY_CONSISTENCY` per direction that matches.
3. **Missing after present** (the evasion case flagged in both Phase 1's and Phase 2's docs): the attribute is `None` on this transaction but a direct predecessor into `src` carried it — a dropped trail — adds `W_IDENTITY_EVASION`, weighted above plain consistency since both spec docs call this out as a *deliberate* signal, not passive absence.
4. **Divergence** (the diverging branch in Example 2, the shifted leg in Example 3): deliberately gets neither bonus nor penalty beyond what (2) already omits — the spec says these "must be weighed together," not that divergence itself is suspicious.

Both `local_component` and the `_active` scan are reused as-is from state `_score` already has or already prunes — no separate identity index exists to fall out of sync with the graph (the exact class of bug the earlier lookback-window fix was written to prevent).

- [x] **Step 1: Add the identity weight constants**

`ghost_chains.py:51-56`:

```python
# Phase 2: identity signal. A shared ipAddress/deviceId is scored relative to
# where the transaction sits in the graph, not as a standalone rule -- these
# weights combine additively with the structural mass above.
W_IDENTITY_REUSE = 2.5
W_IDENTITY_CONSISTENCY = 0.6
W_IDENTITY_EVASION = 3.0
```

- [x] **Step 2: Thread `ip`/`device` into `_score` and fold `_identity_mass` into the mass sum**

`ghost_chains.py:221` — `_score` gains two parameters:

```python
def _score(self, src, dst, created, ip, device):
```

`ghost_chains.py:281-290` — computed alongside the existing terms, added into the same sum that later gets the cycle temporal-factor scaling applied to it (consistent with every other term in `mass`, no special-cased exemption):

```python
        damping = REPEAT_EDGE_DAMPING if (src, dst) in self._edge_times else 1.0
        local_component = upstream.keys() | downstream.keys() | src_reaches.keys() | reaches_dst.keys()
        identity_mass = self._identity_mass(src, dst, ip, device, local_component)
        mass = (
            W_NEW_PAIR * fresh_pairs * damping
            + W_EXTRA_ROUTE * extra_routes * damping
            + W_FAN * fan
            + W_CYCLE * cycle_mass
            + identity_mass
        )
```

- [x] **Step 3: Implement `_identity_mass`**

`ghost_chains.py:302-346`:

```python
    def _identity_mass(self, src, dst, ip, device, local_component):
        """Identity signal: ipAddress/deviceId scored relative to where this
        transaction sits in the active graph, ip and device as independent
        dimensions. Reuses self._active (already 24h-pruned) as the source of
        truth, so there is no separate identity index to keep in sync.
        """
        local_component = local_component | {src, dst}

        total = 0.0
        for attr, value in (("ip", ip), ("device", device)):
            if value is None:
                # Absence only matters if a direct predecessor into src carried
                # this attribute -- a dropped trail, not absence in a vacuum.
                for edge_src, edge_dst, edge_ip, edge_device in self._active.values():
                    if edge_dst != src:
                        continue
                    if (edge_ip if attr == "ip" else edge_device) is not None:
                        total += W_IDENTITY_EVASION
                        break
                continue

            external_users = set()
            predecessor_match = False
            successor_match = False
            for edge_src, edge_dst, edge_ip, edge_device in self._active.values():
                edge_value = edge_ip if attr == "ip" else edge_device
                if edge_value != value:
                    continue
                if edge_src not in local_component:
                    external_users.add(edge_src)
                if edge_dst not in local_component:
                    external_users.add(edge_dst)
                if edge_dst == src:
                    predecessor_match = True
                if edge_src == dst:
                    successor_match = True

            if external_users:
                total += W_IDENTITY_REUSE * (1.0 - DECAY ** len(external_users))
            if predecessor_match:
                total += W_IDENTITY_CONSISTENCY
            if successor_match:
                total += W_IDENTITY_CONSISTENCY

        return total
```

- [x] **Step 4: Read `ipAddress`/`deviceId` in `_process_one` and carry them through to `_active`**

`ghost_chains.py:376-383` — read right after `src`/`dst` are resolved:

```python
        src = raw.get("fromUserId")
        dst = raw.get("toUserId")
        if src is None or dst is None:
            return {"txId": tx_id, "riskScore": 0.0}
        src, dst = str(src), str(dst)

        ip = raw.get("ipAddress")
        device = raw.get("deviceId")
```

`ghost_chains.py:402-405` — passed into `_score`:

```python
        if src == dst:
            score = 0.0
        else:
            score = self._score(src, dst, created, ip, device)
```

`ghost_chains.py:414-417` — `_active` now stores a 4-tuple instead of 2 (the existing `_expire` → `_drop_edge(edge[0], edge[1], created)` call still works unmodified, since it only ever indexed the first two elements):

```python
        if created > self._clock - LOOKBACK and src != dst:
            self._add_edge(src, dst, created)
            heapq.heappush(self._expiry, (created, tx_id))
            self._active[tx_id] = (src, dst, ip, device)
```

- [x] **Step 5: Write and run the test suite**

`test_ghost_chains.py` — 18 tests, all passing (`python -m unittest test_ghost_chains -v`):
- `Phase1DocumentedExamplesTests` (6): the five documented examples pinned to exact values plus strict ordering — `0.0, 0.049751093, 0.152311329, 0.564647158, 0.690655806`.
- `SelfTransferAndIsolationTests` (2): self-transfer always `0.0`; a structurally-isolated transaction stays `0.0` even with unrelated activity already in the window (this engine already handles both correctly via the `already_linked` DECAY subtraction and the explicit `src == dst` guard — no fix needed here, unlike the earlier BFS-based scorer).
- `IdempotencyAndValidationTests` (2): duplicate identical payload returns the original result; missing optional identity fields don't fail processing.
- `ResetTests` (1): `clear()` restores startup-equivalent state.
- `Phase2IdentityScoringTests` (7): all four Phase 2 documented examples (consistent chain, branching divergence, mid-flow shift, disconnected reuse) pinned to exact values, plus the evasion case and the no-prior-context isolation case.

- [x] **Step 6: End-to-end smoke test through the live HTTP server**

Confirmed `GET /ghost-chains/health`, `POST /ghost-chains/reset`, and `POST /ghost-chains/transactions` (with `deviceId` present) all respond correctly through `app.py`'s existing routing — no `app.py` changes were needed since it already calls `GhostChainsService.process_transactions`/`clear`/`health` by name, and Phase 2 didn't change any of those three signatures.

- [x] **Step 7: Stage the changes**

```bash
git add ghost_chains.py test_ghost_chains.py
```

(Not committed — this repository's convention is to stage only and leave committing to the developer.)

---

## Self-Review

**Spec coverage:**
- Health/reset/transactions endpoints: unchanged, no new requirements in Phase 2 — correctly untouched.
- Lookback window, ordering, idempotency, missing-optionals, unknown-fields, memory bounding: all pre-existing engine behavior, covered by the non-identity tests, untouched by this work.
- "Optional fields contribute an identity signal relative to where the transaction sits in the active graph": `_identity_mass`, gated entirely on `self._active` (the live windowed graph).
- "When both are present, treat them as independent dimensions": the `for attr, value in (("ip", ip), ("device", device))` loop scores each separately and sums them.
- "Shared identity across disconnected components is a distinct coordination hint": the external-users branch, tested by Example 4.
- "Missing identity on a connected path... the absence itself can be a signal": the evasion branch, tested explicitly.
- "Weigh absence against the surrounding structure rather than treating every missing field as suspicious": gated on a real predecessor match existing — an isolated transaction with no inbound history never triggers it (`test_isolated_transaction_with_identity_but_no_prior_context_scores_zero`).
- "Systems built on principled graph models are expected to outperform implementations tuned to specific patterns": no example's exact values are hardcoded into the scoring logic itself — `_identity_mass` is a general rule over `local_component`/`self._active`; the tests check the doc's four examples but don't drive the implementation via special-casing.

**Placeholder scan:** none — every step reflects code that is actually in `ghost_chains.py` right now.

**Type consistency:** `_score(self, src, dst, created, ip, device)` and `_identity_mass(self, src, dst, ip, device, local_component)` signatures match their single call sites exactly (`_process_one` → `_score` → `_identity_mass`). `self._active[tx_id]` is a 4-tuple everywhere it's written (`_process_one`) and everywhere it's read (`_identity_mass`, and `_expire`'s `edge[0], edge[1]` unpacking, which only ever needed the first two elements and still gets them).
