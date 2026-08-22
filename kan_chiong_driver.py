import heapq
import json
from bisect import bisect_right
from datetime import datetime, timedelta

EPS = 1e-9


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s[:-1] + "+00:00" if s.endswith("Z") else s)


def _format_iso(dt: datetime) -> str:
    s = dt.isoformat()
    return s[:-6] + "Z" if s.endswith("+00:00") else s


def _round_duration(seconds: float):
    rounded = round(seconds)
    return rounded if abs(seconds - rounded) < 1e-6 else round(seconds, 6)


def _build_profile(windows):
    # flatten the obstruction windows for this edge into one timeline so we're
    # not re-scanning the raw list every time we traverse the edge again
    breakpoints = sorted({bp for start, end, _ in windows for bp in (start, end)})
    factors = []
    for i in range(len(breakpoints) - 1):
        at = breakpoints[i]
        active = [factor for start, end, factor in windows if start <= at < end]
        factors.append(min(active) if active else 1.0)
    return breakpoints, factors


def _factor_and_next(breakpoints, factors, t):
    idx = bisect_right(breakpoints, t) - 1
    if idx < 0:
        return 1.0, (breakpoints[0] if breakpoints else None)
    if idx >= len(factors):
        return 1.0, None
    return factors[idx], (breakpoints[idx + 1] if idx + 1 < len(breakpoints) else None)


def _traverse_edge(edge_id, frm, to, base_duration, departure, edge_profiles):
    # walks the speed-factor breakpoints one at a time since the factor can
    # change mid-traversal. if we hit a factor of 0 we're stuck (no waiting
    # it out), so just bail with None
    profile = edge_profiles.get((edge_id, frm, to))
    if profile is None:
        return departure + timedelta(seconds=base_duration)

    breakpoints, factors = profile
    remaining = float(base_duration)
    t = departure
    while remaining > EPS:
        factor, next_bp = _factor_and_next(breakpoints, factors, t)
        if factor <= 0.0:
            return None
        if next_bp is None:
            t += timedelta(seconds=remaining / factor)
            remaining = 0.0
            continue
        coverable = (next_bp - t).total_seconds() * factor
        if coverable >= remaining - EPS:
            t += timedelta(seconds=remaining / factor)
            remaining = 0.0
        else:
            remaining -= coverable
            t = next_bp
    return t


def _static_shortest_paths(adjacency, target):
    # plain old Dijkstra using base durations, no time-dependent stuff here.
    # once we're past the last obstruction window everything behaves like this,
    # so we can just jump straight to the target instead of keep exploring
    dist = {target: 0}
    prev_node = {}
    prev_edge = {}
    visited = set()
    heap = [(0, target)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, edge_id, weight in adjacency.get(u, []):
            nd = d + weight
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev_node[v] = u
                prev_edge[v] = edge_id
                heapq.heappush(heap, (nd, v))

    path_cache = {target: ()}

    def path_to_target(node):
        # caches as it goes so each node only gets resolved once, basically
        # union-find style path compression
        if node not in dist:
            return None
        chain = []
        cur = node
        while cur not in path_cache:
            chain.append(cur)
            cur = prev_node[cur]
        suffix = path_cache[cur]
        for n in reversed(chain):
            suffix = (prev_edge[n],) + suffix
            path_cache[n] = suffix
        return path_cache[node]

    return dist, path_to_target


def solve(data: str) -> str:
    # main entry point - finds the fastest route given the traffic
    # obstructions and spits back duration / arrival time / edge path
    payload = json.loads(data)

    start = tuple(payload["start_coordinate"])
    end = tuple(payload["end_coordinate"])
    departure_time = _parse_iso(payload["start_time"])

    if start == end:
        return json.dumps(
            {"total_duration_sec": 0, "arrival_time": _format_iso(departure_time), "path": []}
        )

    adjacency = {}
    for edge in payload.get("edges", []):
        edge_id = edge["edge_id"]
        node1, node2 = tuple(edge["node1"]), tuple(edge["node2"])
        duration = edge["base_duration_sec"]
        adjacency.setdefault(node1, []).append((node2, edge_id, duration))
        adjacency.setdefault(node2, []).append((node1, edge_id, duration))

    raw_windows = {}
    horizon = departure_time
    for obstruction in payload.get("obstructions", []):
        edge_id = obstruction["edge_id"]
        frm, to = tuple(obstruction["edge"]["from"]), tuple(obstruction["edge"]["to"])
        end_time = _parse_iso(obstruction["end_time"])
        window = (_parse_iso(obstruction["start_time"]), end_time, float(obstruction["speed_factor"]))
        raw_windows.setdefault((edge_id, frm, to), []).append(window)
        if end_time > horizon:
            horizon = end_time

    edge_profiles = {key: _build_profile(windows) for key, windows in raw_windows.items()}
    static_dist, static_path = _static_shortest_paths(adjacency, end)

    # storing states as parallel arrays + parent pointers instead of copying
    # the whole path onto the heap every push (that got slow fast when i tried
    # it the naive way). path only gets rebuilt once we actually pop the target
    node_of = [start]
    parent_of = [-1]
    edges_of = [()]

    def new_state(node, parent, edges):
        node_of.append(node)
        parent_of.append(parent)
        edges_of.append(edges)
        return len(node_of) - 1

    heap = [(departure_time, 0)]
    processed = set()

    while heap:
        time, state_id = heapq.heappop(heap)
        node = node_of[state_id]
        state_key = (node, time)
        if state_key in processed:
            continue
        processed.add(state_key)

        if node == end:
            segments = []
            cur = state_id
            while cur != -1:
                segments.append(edges_of[cur])
                cur = parent_of[cur]
            path = [edge_id for segment in reversed(segments) for edge_id in segment]
            total_seconds = (time - departure_time).total_seconds()
            return json.dumps(
                {
                    "total_duration_sec": _round_duration(total_seconds),
                    "arrival_time": _format_iso(time),
                    "path": path,
                }
            )

        if time >= horizon:
            remaining = static_dist.get(node)
            if remaining is not None:
                arrival = time + timedelta(seconds=remaining)
                new_id = new_state(end, state_id, static_path(node))
                heapq.heappush(heap, (arrival, new_id))
            continue

        for neighbor, edge_id, duration in adjacency.get(node, []):
            arrival = _traverse_edge(edge_id, node, neighbor, duration, time, edge_profiles)
            if arrival is not None:
                new_id = new_state(neighbor, state_id, (edge_id,))
                heapq.heappush(heap, (arrival, new_id))

    return json.dumps(
        {
            "total_duration_sec": None, 
            "arrival_time": None, 
            "path": []
        }
    )
