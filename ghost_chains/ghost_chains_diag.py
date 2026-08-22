"""Diagnostic copy of ghost_chains for replaying Phase 1 examples.

This file is a non-destructive scratchpad; it does not modify the repository copy.
It replays the example sequences and prints final transaction scores and graph state.
"""

import bisect
import heapq
import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LOOKBACK = timedelta(hours=24)

DECAY = 0.55
MAX_DEPTH = 6
MAX_VISIT = 2000

W_NEW_PAIR = 1.0
W_EXTRA_ROUTE = 3.0
W_FAN = 0.35
W_CYCLE = 12.0
CYCLE_REINFORCEMENT = 1.0
REPEAT_EDGE_DAMPING = 0.35
TEMPORAL_FLOOR = 0.75

W_IDENTITY_REUSE = 2.5
W_IDENTITY_CONSISTENCY = 0.6
W_IDENTITY_EVASION = 3.0

SATURATION = 4.0
TAIL = 0.7

_DECAY_POW = [DECAY ** i for i in range(MAX_DEPTH * 2 + 4)]


def _parse_ts(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _fingerprint(tx):
    return (
        str(tx.get("fromUserId")),
        str(tx.get("toUserId")),
        repr(tx.get("amount")),
        str(tx.get("createdAt")),
    )


class GhostChainsService:
    def __init__(self):
        self._lock = threading.RLock()
        self.reset()

    def reset(self):
        with self._lock:
            self._adj = defaultdict(set)
            self._rev = defaultdict(set)
            self._edge_times = {}
            self._expiry = []
            self._active = {}
            self._scores = {}
            self._fingerprints = {}
            self._clock = None

    def health(self):
        return {"status": "ok"}

    def process_batch(self, transactions):
        with self._lock:
            return [self._process_one(raw) for raw in transactions]

    def process_transactions(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        transactions = payload.get("transactions")
        if not isinstance(transactions, list):
            raise ValueError("transactions must be an array")
        return {"transactions": self.process_batch(transactions)}

    def clear(self, payload):
        if not isinstance(payload, dict) or payload.get("clearTransactions") is not True:
            raise ValueError('Request body must be {"clearTransactions": true}')
        with self._lock:
            self.reset()
        return {"clearTransactions": True}

    def _add_edge(self, src, dst, created):
        self._adj[src].add(dst)
        self._rev[dst].add(src)
        times = self._edge_times.get((src, dst))
        if times is None:
            self._edge_times[(src, dst)] = [created]
        else:
            bisect.insort(times, created)

    def _drop_edge(self, src, dst, created):
        times = self._edge_times.get((src, dst))
        if times is None:
            return
        index = bisect.bisect_left(times, created)
        if index < len(times) and times[index] == created:
            times.pop(index)
        if times:
            return
        del self._edge_times[(src, dst)]
        self._adj[src].discard(dst)
        if not self._adj[src]:
            del self._adj[src]
        self._rev[dst].discard(src)
        if not self._rev[dst]:
            del self._rev[dst]

    def _expire(self, now):
        cutoff = now - LOOKBACK
        while self._expiry and self._expiry[0][0] <= cutoff:
            created, tx_id = heapq.heappop(self._expiry)
            edge = self._active.pop(tx_id, None)
            if edge is not None:
                self._drop_edge(edge[0], edge[1], created)

    def _walk(self, graph, start, limit=MAX_DEPTH):
        dist = {start: 0}
        loops_back = False
        frontier = [start]
        depth = 0
        while frontier and depth < limit:
            depth += 1
            nxt = []
            for node in frontier:
                for peer in graph.get(node, ()): 
                    if peer == start:
                        loops_back = True
                    if peer in dist:
                        continue
                    dist[peer] = depth
                    nxt.append(peer)
                    if len(dist) >= MAX_VISIT:
                        return dist, loops_back
            frontier = nxt
        return dist, loops_back

    def _oldest_edge_within(self, nodes):
        oldest = None
        for src in sorted(nodes):
            for dst in self._adj.get(src, ()): 
                if dst not in nodes:
                    continue
                first = self._edge_times[(src, dst)][0]
                if oldest is None or first < oldest:
                    oldest = first
        return oldest

    def _score(self, src, dst, created, ip, device):
        upstream, src_looped = self._walk(self._rev, src)
        downstream, dst_looped = self._walk(self._adj, dst)
        src_reaches, _ = self._walk(self._adj, src)
        reaches_dst, _ = self._walk(self._rev, dst)

        already_linked = dst in src_reaches

        total_in = extra_in = 0.0
        for node, depth in upstream.items():
            weight = _DECAY_POW[depth]
            total_in += weight
            if node in reaches_dst:
                extra_in += weight

        total_out = extra_out = 0.0
        for node, depth in downstream.items():
            weight = _DECAY_POW[depth]
            total_out += weight
            if node in src_reaches:
                extra_out += weight

        every_pair = DECAY * total_in * total_out
        fresh_pairs = DECAY * (total_in - extra_in) * (total_out - extra_out)
        extra_routes = every_pair - fresh_pairs
        if already_linked:
            extra_routes -= DECAY
        else:
            fresh_pairs -= DECAY
        fresh_pairs = max(0.0, fresh_pairs)
        extra_routes = max(0.0, extra_routes)

        cycle_nodes = set()
        cycle_mass = 0.0
        for node, depth in upstream.items():
            back = downstream.get(node)
            if back is None:
                continue
            cycle_nodes.add(node)
            cycle_mass += _DECAY_POW[min(depth + 1 + back, len(_DECAY_POW) - 1)]

        if src_looped or dst_looped:
            cycle_mass *= 1.0 + CYCLE_REINFORCEMENT

        in_peers = len(self._rev.get(dst, ())) - (1 if src in self._rev.get(dst, ()) else 0)
        out_peers = len(self._adj.get(src, ())) - (1 if dst in self._adj.get(src, ()) else 0)
        fan = (1.0 - DECAY ** in_peers) + (1.0 - DECAY ** out_peers)

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

        if cycle_nodes:
            mass *= self._temporal_factor(cycle_nodes, created)

        if mass <= 0.0:
            return 0.0
        return round(1.0 - (1.0 + mass / SATURATION) ** -TAIL, 9)

    def _identity_mass(self, src, dst, ip, device, local_component):
        local_component = local_component | {src, dst}

        total = 0.0
        for attr, value in (("ip", ip), ("device", device)):
            if value is None:
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

    def _temporal_factor(self, nodes, created):
        oldest = self._oldest_edge_within(nodes)
        if oldest is None:
            return 1.0
        span = (created - oldest).total_seconds()
        if span <= 0:
            return 1.0
        share = min(1.0, span / LOOKBACK.total_seconds())
        return 1.0 - (1.0 - TEMPORAL_FLOOR) * share

    def _process_one(self, raw):
        if not isinstance(raw, dict):
            return {"txId": None, "riskScore": 0.0}

        tx_id = raw.get("txId")
        if tx_id is None:
            return {"txId": None, "riskScore": 0.0}

        previous = self._scores.get(tx_id)
        if previous is not None:
            if self._fingerprints.get(tx_id) != _fingerprint(raw):
                logger.warning("duplicate txId %s with a different payload", tx_id)
            return {"txId": tx_id, "riskScore": previous}

        src = raw.get("fromUserId")
        dst = raw.get("toUserId")
        if src is None or dst is None:
            return {"txId": tx_id, "riskScore": 0.0}
        src, dst = str(src), str(dst)

        ip = raw.get("ipAddress")
        device = raw.get("deviceId")

        created = _parse_ts(raw.get("createdAt")) or self._clock
        if created is None:
            created = datetime.now(timezone.utc)
        if self._clock is None or created > self._clock:
            self._clock = created

        self._expire(self._clock)

        if src == dst:
            score = 0.0
        else:
            score = self._score(src, dst, created, ip, device)

        if created > self._clock - LOOKBACK and src != dst:
            self._add_edge(src, dst, created)
            heapq.heappush(self._expiry, (created, tx_id))
            self._active[tx_id] = (src, dst, ip, device)

        self._scores[tx_id] = score
        self._fingerprints[tx_id] = _fingerprint(raw)
        return {"txId": tx_id, "riskScore": score}


if __name__ == "__main__":
    svc = GhostChainsService()

    base = datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc)

    def mk(txid, f, t, minutes=0):
        return {
            "txId": txid,
            "fromUserId": f,
            "toUserId": t,
            "amount": 100.0,
            "createdAt": (base + timedelta(minutes=minutes)).isoformat(),
        }

    examples = {
        "Example1-Isolated": [mk("e1_t1", "Meridian Holdings", "Apex Logistics", 0)],
        "Example2-Extension": [mk("e2_t1", "Meridian Holdings", "Apex Logistics", 0), mk("e2_t2", "Apex Logistics", "Cascade Payments", 1)],
        "Example3-Convergence": [mk("e3_t1", "Meridian Holdings", "Apex Logistics", 0), mk("e3_t2", "Meridian Holdings", "Horizon Capital", 1), mk("e3_t3", "Apex Logistics", "Sterling Bridge", 2), mk("e3_t4", "Horizon Capital", "Sterling Bridge", 3)],
        "Example4-Return": [mk("e4_t1", "Meridian Holdings", "Apex Logistics", 0), mk("e4_t2", "Apex Logistics", "Cascade Payments", 1), mk("e4_t3", "Cascade Payments", "Oakridge Imports", 2), mk("e4_t4", "Oakridge Imports", "Apex Logistics", 3)],
        "Example5-MultiLoop": [mk("e5_t1", "Meridian Holdings", "Apex Logistics", 0), mk("e5_t2", "Apex Logistics", "Cascade Payments", 1), mk("e5_t3", "Cascade Payments", "Meridian Holdings", 2), mk("e5_t4", "Apex Logistics", "Nimbus Trading", 3), mk("e5_t5", "Nimbus Trading", "Meridian Holdings", 4)],
    }

    for name, txs in examples.items():
        svc.reset()
        print('\n===', name, '===')
        for i, tx in enumerate(txs):
            res = svc.process_batch([tx])[0]
            print(f"Processed {tx['txId']}: score={res['riskScore']}")
            if i == len(txs) - 1:
                # final tx in sequence; print some internal stats
                print("Adjacency:", dict((k, sorted(v)) for k, v in svc._adj.items()))
                print("Edge times count:", len(svc._edge_times))
                print("Active txs:", len(svc._active))

    print('\nDiagnostics complete')
