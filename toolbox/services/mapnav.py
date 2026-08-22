"""Phase 2 (School Days) Problem Set 2: tolled shortest-path map
navigation, with an optional bounded-hop budget."""

import heapq
import logging
import os
import threading
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

API_BASE = os.environ.get("TOOLBOX_API_BASE", "https://tool-box-2591eaa24fa3.herokuapp.com")
_FETCH_TIMEOUT = 5

_graph_lock = threading.Lock()
_graphs = {}

_journey_lock = threading.Lock()
_journeys = {}


def _fetch_graph(map_id):
    url = urljoin(API_BASE + "/", "graph")
    response = requests.get(url, params={"map_id": map_id}, timeout=_FETCH_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    return payload["adjacency"], payload["tolls"]


def _graph(map_id):
    with _graph_lock:
        if map_id not in _graphs:
            _graphs[map_id] = _fetch_graph(map_id)
        return _graphs[map_id]


def _journey_visited(map_id, start, destination):
    key = (map_id, start, destination)
    with _journey_lock:
        if key not in _journeys:
            _journeys[key] = {start}
        return key, _journeys[key]


def _mark_visited(key, node, destination):
    with _journey_lock:
        visited = _journeys.get(key)
        if visited is None:
            return
        if node == destination:
            _journeys.pop(key, None)
        else:
            visited.add(node)


def _entry_cost(adjacency, tolls, u, v):
    return adjacency[u][v] + tolls.get(v, 0.0)


def _shortest_next_hop(adjacency, tolls, current, destination, visited):
    def allowed(node):
        return node == destination or node not in visited

    distances = {current: 0.0}
    previous = {}
    frontier = [(0.0, current)]
    seen = set()
    while frontier:
        dist, node = heapq.heappop(frontier)
        if node in seen:
            continue
        seen.add(node)
        if node == destination:
            break
        for neighbor in adjacency.get(node, {}):
            if not allowed(neighbor):
                continue
            cost = dist + _entry_cost(adjacency, tolls, node, neighbor)
            if cost < distances.get(neighbor, float("inf")):
                distances[neighbor] = cost
                previous[neighbor] = node
                heapq.heappush(frontier, (cost, neighbor))

    if destination != current and destination not in previous:
        return None

    node = destination
    path = [node]
    while node != current:
        node = previous[node]
        path.append(node)
    path.reverse()
    return path[1] if len(path) > 1 else destination


def _bounded_next_hop(adjacency, tolls, current, destination, visited, hops_remaining):
    if hops_remaining <= 0:
        return None

    nodes = set(adjacency) | {destination}
    for edges in adjacency.values():
        nodes.update(edges)
    allowed_nodes = {n for n in nodes if n == destination or n not in visited}

    best = {0: {n: (0.0 if n == destination else float("inf")) for n in allowed_nodes}}
    for k in range(1, hops_remaining + 1):
        layer = {}
        for node in allowed_nodes:
            best_cost = best[k - 1].get(node, float("inf"))
            for neighbor, weight in adjacency.get(node, {}).items():
                if neighbor not in allowed_nodes:
                    continue
                cost = weight + tolls.get(neighbor, 0.0) + best[k - 1].get(neighbor, float("inf"))
                if cost < best_cost:
                    best_cost = cost
            layer[node] = best_cost
        best[k] = layer

    best_neighbor, best_cost = None, float("inf")
    for neighbor, weight in adjacency.get(current, {}).items():
        if neighbor not in allowed_nodes:
            continue
        cost = weight + tolls.get(neighbor, 0.0) + best[hops_remaining - 1].get(neighbor, float("inf"))
        if cost < best_cost:
            best_cost, best_neighbor = cost, neighbor

    return best_neighbor


def _fallback_next_hop(adjacency, current, visited):
    for neighbor in adjacency.get(current, {}):
        if neighbor not in visited:
            return neighbor
    return None


def next_hop(args):
    map_id = args.get("map_id")
    start = args.get("start")
    current = args.get("current")
    destination = args.get("destination")
    hops_remaining = args.get("hops_remaining")

    if not map_id or not start or not current or not destination:
        raise ValueError("map_id, start, current, and destination are required")

    adjacency, tolls = _graph(map_id)
    if current == destination:
        return destination

    key, visited = _journey_visited(map_id, start, destination)

    if hops_remaining is not None:
        result = _bounded_next_hop(adjacency, tolls, current, destination, visited, int(hops_remaining))
        if result is None:
            result = _shortest_next_hop(adjacency, tolls, current, destination, visited)
    else:
        result = _shortest_next_hop(adjacency, tolls, current, destination, visited)

    if result is None:
        result = _fallback_next_hop(adjacency, current, visited)
    if result is None:
        raise ValueError("no reachable next hop from {}".format(current))

    _mark_visited(key, result, destination)
    return result
