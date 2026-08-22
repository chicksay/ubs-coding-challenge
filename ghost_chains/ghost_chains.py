"""Phase 1 Ghost Chains scorer.


The score of an edge u -> v is the increase in the graph's capacity to carry
recurring flow, measured over the active 24h window:


 * pairs (a, d) that become connected for the first time      -> weak signal
 * pairs (a, d) that gain an additional route                 -> medium signal
 * pairs (a, d) whose existing shortest route gets shorter    -> medium signal
 * nodes that can now reach themselves through the new edge   -> strong signal


Every contribution is discounted by the length of the path it creates, so
short structures dominate long ones. Ancestor and descendant mass are each
computed once via bounded BFS (sum_a y^dist(a,u), sum_d y^dist(v,d)); pairs
themselves are not enumerated, but classifying whether a given (a, d) pair
already existed *does* require one bounded reachability check per ancestor
against the descendant set, since two nodes can already be connected by a
path that never touches this edge's own endpoints (see _score).


Active history is exactly the most recent 24 hours: an edge is live while
its age is strictly less than 24h, and expired edges are deleted from the
graph so they cannot influence any later score.
"""


import bisect
import heapq
import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone


logger = logging.getLogger(__name__)


LOOKBACK = timedelta(hours=24)


DECAY = 0.55
# 6 was an arbitrary cutoff that hard-froze scores for any chain longer than
# ~7 hops: every extension past that point produced the exact same score,
# bit-for-bit, because the walk simply stopped discovering ancestors/
# descendants rather than letting DECAY's own exponential falloff make them
# negligible. 25 is chosen so the decay itself does that work instead --
# DECAY**25 ~= 2e-7, already far below floating-point relevance at the score
# precision this service rounds to -- so this is the depth at which going
# further genuinely stops mattering, not an arbitrary earlier stopping point.
# MAX_VISIT below still bounds total work per call regardless of depth, so
# this costs nothing on wide graphs and only helps deep/chain-like ones.
MAX_DEPTH = 25
MAX_VISIT = 2000


W_NEW_PAIR = 1.0
W_EXTRA_ROUTE = 3.0
# Core Principle (ghost_chains.txt): "the combined effect of new OR SHORTENED
# paths between entities." fresh_pairs/extra_routes above only ever compare
# an already-linked pair's mass against zero -- they never compare the new
# route's length against the length the pair already had. A shortcut that
# collapses an existing 3-hop relationship into 1 hop and a same-length
# parallel route (e.g. Phase 1 Example 3's convergence) were previously
# scored by the identical mechanism; W_SHORTEN gives the strictly-shorter
# case its own explicit, additive credit, matching the same weight as an
# extra route since both are "route" signals -- shortening is the sub-case
# the principle calls out by name, not a separate tier of severity.
W_SHORTEN = 3.0
W_FAN = 0.35
W_CYCLE = 12.0
CYCLE_REINFORCEMENT = 1.0
REPEAT_EDGE_DAMPING = 0.35

# Diagnosis finding: a return loop that closes quickly is a stronger
# laundering signal than one that takes most of the 24h window to close --
# fast round-tripping is the classic layering red flag, while a slow-closing
# loop looks more like ordinary recurring business. Cycle mass is scaled
# toward TEMPORAL_FLOOR as the loop's oldest edge nears the window boundary;
# TEMPORAL_FLOOR = 1.0 would be a no-op (validated against ghost_chains_diag.py
# before porting here -- see ghost_chains_diagnosis.txt experiment A).
TEMPORAL_FLOOR = 0.75

# Phase 2: identity signal. A shared ipAddress/deviceId is scored relative to
# where the transaction sits in the graph, not as a standalone rule -- these
# weights combine additively with the structural mass above.
W_IDENTITY_REUSE = 2.5
W_IDENTITY_CONSISTENCY = 0.6
W_IDENTITY_EVASION = 3.0

# Phase 3: value signal. amount is compared against each direct active
# predecessor edge's amount -- one hop back, recomputed live every call, no
# persistent path state. Retaining most of the prior amount is the expected
# layering signature (a small confirming nudge); an amount *increase* against
# its immediate predecessor is a trajectory reversal, a direct contradiction
# of that pattern, and is weighted well above plain consistency.
W_VALUE_CONSISTENCY = 0.6
W_VALUE_REVERSAL = 3.0
# Both directions use the same DECAY**(deviation/scale) shape -- symmetric in
# the size of the deviation from ratio 1.0, differing only in which side of
# 1.0 they read and how heavily they're weighted above. VALUE_DEVIATION_SCALE
# is set to the ~1% per-hop skim rate the spec's own consistent-decay example
# uses, so that example's deviations sit near one scale unit (a moderate,
# non-saturated signal) while its reversal example's deviation -- explicitly
# "already a direct contradiction" per spec -- saturates quickly rather than
# growing slowly from zero the way DECAY**excess alone would.
VALUE_DEVIATION_SCALE = 0.01


# Scores are relative ranks, so the mass -> [0,1) map must stay strictly
# increasing. A map that flattens to 1.0 would tie every busy transaction
# together and destroy the ranking on dense streams.
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
       """Drop every transaction that is no longer inside the 24h window.


       Active means age < 24h, so an edge that has just reached exactly 24h
       is already gone and cannot contribute to the score being computed.
       """
       cutoff = now - LOOKBACK
       while self._expiry and self._expiry[0][0] <= cutoff:
           created, tx_id = heapq.heappop(self._expiry)
           edge = self._active.pop(tx_id, None)
           if edge is not None:
               self._drop_edge(edge[0], edge[1], created)


   def _walk(self, graph, start, limit=MAX_DEPTH):
       """Depth-limited BFS. Also reports whether start is reachable from
       itself, and each discovered node's parent in the shortest-path tree
       rooted at `start` -- used to reconstruct one witness path for a
       specific cycle, rather than conflating every cycle the new edge
       might close into a single merged node set (see _chain_nodes)."""
       dist = {start: 0}
       parent = {start: None}
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
                   parent[peer] = node
                   nxt.append(peer)
                   if len(dist) >= MAX_VISIT:
                       return dist, loops_back, parent
           frontier = nxt
       return dist, loops_back, parent


   def _chain_nodes(self, parent, node):
       """Nodes along the shortest-path chain from `node` back to the walk's
       root, by following BFS parent pointers -- one witness path, used to
       scope a cycle's temporal factor to its own edges only."""
       nodes = set()
       cur = node
       while cur is not None:
           nodes.add(cur)
           cur = parent.get(cur)
       return nodes


   def _oldest_edge_within(self, nodes):
       """Earliest active edge with both endpoints in `nodes` -- how long this
       local structure has existed, used to judge how quickly a cycle closed
       relative to the 24h window."""
       oldest = None
       for src in sorted(nodes):
           for dst in self._adj.get(src, ()):
               if dst not in nodes:
                   continue
               first = self._edge_times[(src, dst)][0]
               if oldest is None or first < oldest:
                   oldest = first
       return oldest


   def _temporal_factor(self, nodes, created):
       """Scale cycle mass down as the loop's oldest edge nears the 24h
       boundary -- a loop closing within minutes is weighted at full
       strength; one that barely closes before its start would expire is
       weighted toward TEMPORAL_FLOOR."""
       oldest = self._oldest_edge_within(nodes)
       if oldest is None:
           return 1.0
       span = (created - oldest).total_seconds()
       if span <= 0:
           return 1.0
       share = min(1.0, span / LOOKBACK.total_seconds())
       return 1.0 - (1.0 - TEMPORAL_FLOOR) * share


   def _score(self, src, dst, created, ip, device, amount):
       upstream, src_looped, upstream_parent = self._walk(self._rev, src)
       downstream, dst_looped, downstream_parent = self._walk(self._adj, dst)
       src_reaches, _, _ = self._walk(self._adj, src)
       reaches_dst, _, _ = self._walk(self._rev, dst)


       already_linked = dst in src_reaches


       downstream_weight = {node: _DECAY_POW[depth] for node, depth in downstream.items()}
       total_out = sum(downstream_weight.values())

       # Exact pair classification: for each ancestor `a`, find exactly how
       # much of the descendant mass it can already reach -- bounded by the
       # same MAX_DEPTH/MAX_VISIT budget as every other walk here -- instead
       # of approximating "already connected" via the coarse a-reaches-dst /
       # d-reached-by-src proxies. Those proxies are blind to an ancestor and
       # descendant that are already linked by some path which never touches
       # src or dst at all (e.g. a shared upstream node two branches over),
       # which would otherwise get double-counted as a brand-new connection.
       # a=src's reach is already computed as src_reaches; every other
       # ancestor gets its own bounded forward walk.
       fresh_pairs = extra_routes = 0.0
       for node, depth in upstream.items():
           weight = _DECAY_POW[depth]
           if node == src:
               reach = src_reaches
           else:
               reach, _, _ = self._walk(self._adj, node)
           reachable_out = sum(w for d, w in downstream_weight.items() if d in reach)
           fresh_pairs += weight * (total_out - reachable_out)
           extra_routes += weight * reachable_out

       fresh_pairs *= DECAY
       extra_routes *= DECAY
       # The edge's own (src, dst) pair is the edge itself, not a path it enables.
       if already_linked:
           extra_routes -= DECAY
       else:
           fresh_pairs -= DECAY
       fresh_pairs = max(0.0, fresh_pairs)
       extra_routes = max(0.0, extra_routes)


       # Shortened-path signal (Core Principle: "new or shortened paths").
       # For an already-linked pair (a, d), fresh_pairs/extra_routes above
       # only know a route exists -- not whether this edge made it shorter.
       # reaches_dst already gives every ancestor's existing distance to dst;
       # src_reaches gives dst's existing distance from every descendant of
       # src. If routing through this edge (upstream depth + 1, or 1 +
       # downstream depth) beats the pair's prior shortest distance, this
       # transaction collapsed that relationship -- weighted by how much
       # shorter the new route is, not just that an alternative exists.
       shorten_mass = 0.0
       existing_src_to_dst = reaches_dst.get(src)
       if existing_src_to_dst is not None and existing_src_to_dst > 1:
           saved = existing_src_to_dst - 1
           shorten_mass += _DECAY_POW[1] * (1.0 - DECAY ** saved)
       for node, depth in upstream.items():
           if depth == 0:
               continue  # src itself, handled above
           old_dist = reaches_dst.get(node)
           if old_dist is None:
               continue
           new_dist = depth + 1
           if new_dist < old_dist:
               saved = old_dist - new_dist
               shorten_mass += _DECAY_POW[new_dist] * (1.0 - DECAY ** saved)
       for node, depth in downstream.items():
           if depth == 0:
               continue  # dst itself, already covered via src_to_dst above
           old_dist = src_reaches.get(node)
           if old_dist is None:
               continue
           new_dist = depth + 1
           if new_dist < old_dist:
               saved = old_dist - new_dist
               shorten_mass += _DECAY_POW[new_dist] * (1.0 - DECAY ** saved)

       cycle_mass = 0.0
       for node, depth in upstream.items():
           back = downstream.get(node)
           if back is None:
               continue
           raw = _DECAY_POW[min(depth + 1 + back, len(_DECAY_POW) - 1)]
           # Scope the temporal factor to *this* cycle's own witnessed path
           # (src-side via upstream_parent, dst-side via downstream_parent),
           # not the union of every cycle the new edge happens to close --
           # an unrelated slow-forming loop that merely shares a node with
           # this one must not dilute this cycle's own closing speed.
           path_nodes = (
               self._chain_nodes(upstream_parent, node)
               | self._chain_nodes(downstream_parent, node)
           )
           cycle_mass += raw * self._temporal_factor(path_nodes, created)


       if src_looped or dst_looped:
           cycle_mass *= 1.0 + CYCLE_REINFORCEMENT


       # A destination several parties already pay into, or a source already
       # paying several parties, is a gathering point. Without this such a
       # transfer ties at exactly 0.0 with two strangers transacting once.
       in_peers = len(self._rev.get(dst, ())) - (1 if src in self._rev.get(dst, ()) else 0)
       out_peers = len(self._adj.get(src, ())) - (1 if dst in self._adj.get(src, ()) else 0)
       fan = (1.0 - DECAY ** in_peers) + (1.0 - DECAY ** out_peers)


       damping = REPEAT_EDGE_DAMPING if (src, dst) in self._edge_times else 1.0
       local_component = upstream.keys() | downstream.keys() | src_reaches.keys() | reaches_dst.keys()
       identity_mass = self._identity_mass(src, dst, ip, device, local_component)
       value_mass = self._value_mass(src, amount)
       mass = (
           W_NEW_PAIR * fresh_pairs * damping
           + W_EXTRA_ROUTE * extra_routes * damping
           + W_SHORTEN * shorten_mass * damping
           + W_FAN * fan
           + W_CYCLE * cycle_mass
           + identity_mass
           + value_mass
       )


       if mass <= 0.0:
           return 0.0
       return round(1.0 - (1.0 + mass / SATURATION) ** -TAIL, 9)


   def _identity_streak(self, node, attr):
       """How many consecutive active predecessor hops into `node` carry the
       identical identity value, walking backward -- the length of the
       established trail an evasion at `node` would be breaking. Phase 2
       Core Principle: "the suspicious case is a consistent flow that stops
       carrying its identity" -- a single prior hop that happened to carry
       some value is much weaker evidence of an established, consistent
       flow than several consecutive hops sharing the same one, so this must
       be graduated rather than a flat "predecessor had something" check."""
       streak = 0
       value = None
       current = node
       visited = {node}
       while streak < MAX_DEPTH:
           predecessor_value = None
           predecessor_node = None
           for edge_src, edge_dst, edge_ip, edge_device, _ in self._active.values():
               if edge_dst != current:
                   continue
               candidate = edge_ip if attr == "ip" else edge_device
               if candidate is None:
                   continue
               predecessor_value = candidate
               predecessor_node = edge_src
               break
           if predecessor_value is None or predecessor_node in visited:
               break
           if value is None:
               value = predecessor_value
           elif predecessor_value != value:
               break
           streak += 1
           visited.add(predecessor_node)
           current = predecessor_node
       return streak

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
               # this attribute -- a dropped trail, not absence in a vacuum --
               # and how suspicious depends on how established that trail was
               # (see _identity_streak).
               streak = self._identity_streak(src, attr)
               if streak > 0:
                   total += W_IDENTITY_EVASION * (1.0 - DECAY ** streak)
               continue

           external_users = set()
           predecessor_match = False
           successor_match = False
           for edge_src, edge_dst, edge_ip, edge_device, edge_amount in self._active.values():
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

   def _value_mass(self, src, amount):
       """Value signal: amount compared against each direct active predecessor
       edge's amount -- one hop back, recomputed live every call, no
       persistent path state. Retaining most of the prior amount confirms
       the expected layering pattern (a small nudge); an amount *increase*
       against its immediate predecessor is a trajectory reversal -- a
       direct contradiction -- weighted well above plain consistency.
       Only direct predecessor edges into src are considered, so branches
       and convergence never blend into one global ratio (Phase 3 Core
       Principle: no blind aggregation across unrelated branches).
       """
       if amount is None:
           return 0.0

       consistency_gap = 1.0
       reversal_gap = 1.0
       for edge_src, edge_dst, edge_ip, edge_device, edge_amount in self._active.values():
           if edge_dst != src or not edge_amount:
               continue
           deviation = amount / edge_amount - 1.0
           if deviation <= 0.0:
               consistency_gap *= DECAY ** ((-deviation) / VALUE_DEVIATION_SCALE)
           else:
               reversal_gap *= DECAY ** (deviation / VALUE_DEVIATION_SCALE)

       consistency_signal = 1.0 - consistency_gap
       reversal_signal = 1.0 - reversal_gap
       return W_VALUE_CONSISTENCY * consistency_signal + W_VALUE_REVERSAL * reversal_signal

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
       amount = raw.get("amount")


       created = _parse_ts(raw.get("createdAt")) or self._clock
       if created is None:
           created = datetime.now(timezone.utc)
       if self._clock is None or created > self._clock:
           self._clock = created


       # Prune against the newest time the stream has reached, so nothing older
       # than 24h can survive in the graph regardless of arrival order.
       self._expire(self._clock)


       # A transfer to oneself carries no laundering signal, and left
       # unguarded it trivially satisfies the "reaches itself" cycle check
       # below (the walk always contains its own start node), which would
       # score every self-transfer as a false-positive cycle.
       if src == dst:
           score = 0.0
       else:
           score = self._score(src, dst, created, ip, device, amount)


       # A transaction that is itself already outside the window is scored but
       # must not enter the graph, or it would outlive its own 24h lifetime.
       # Self-loops are also kept out of the graph entirely: a self-loop edge
       # would make future _walk() calls from this node see peer == start on
       # their very first hop, spuriously flagging src_looped/dst_looped and
       # inflating unrelated later scores through CYCLE_REINFORCEMENT.
       if created > self._clock - LOOKBACK and src != dst:
           self._add_edge(src, dst, created)
           heapq.heappush(self._expiry, (created, tx_id))
           self._active[tx_id] = (src, dst, ip, device, amount)


       self._scores[tx_id] = score
       self._fingerprints[tx_id] = _fingerprint(raw)
       return {"txId": tx_id, "riskScore": score}




ENGINE = GhostChainsService()
