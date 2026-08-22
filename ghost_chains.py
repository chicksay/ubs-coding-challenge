from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


LOOKBACK_WINDOW_SECONDS = 24 * 60 * 60
EPS = 1e-9
REPEAT_EDGE_WEIGHT = 0.5


class GhostChainsValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Transaction:
    tx_id: str
    from_user_id: str
    to_user_id: str
    amount: float
    created_at: float
    ip_address: str | None
    device_id: str | None


@dataclass(frozen=True)
class StoredTransaction:
    transaction: Transaction
    risk_score: float


def parse_iso8601(timestamp: str) -> float:
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GhostChainsValidationError(f"Invalid ISO-8601 timestamp: {timestamp}") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)

    return parsed.timestamp()


def normalize_transaction(raw_transaction: Any) -> Transaction:
    if not isinstance(raw_transaction, dict):
        raise GhostChainsValidationError("Each transaction must be an object")

    required_fields = {"txId", "fromUserId", "toUserId", "amount", "createdAt"}
    missing_fields = required_fields - set(raw_transaction)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise GhostChainsValidationError(f"Missing required transaction field(s): {missing}")

    tx_id = raw_transaction["txId"]
    from_user_id = raw_transaction["fromUserId"]
    to_user_id = raw_transaction["toUserId"]
    amount = raw_transaction["amount"]
    created_at_raw = raw_transaction["createdAt"]
    ip_address = raw_transaction.get("ipAddress")
    device_id = raw_transaction.get("deviceId")

    if not isinstance(tx_id, str) or not tx_id:
        raise GhostChainsValidationError("transactions[].txId must be a non-empty string")
    if not isinstance(from_user_id, str) or not from_user_id:
        raise GhostChainsValidationError("transactions[].fromUserId must be a non-empty string")
    if not isinstance(to_user_id, str) or not to_user_id:
        raise GhostChainsValidationError("transactions[].toUserId must be a non-empty string")
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        raise GhostChainsValidationError("transactions[].amount must be numeric")
    if math.isnan(amount) or math.isinf(amount):
        raise GhostChainsValidationError("transactions[].amount must be finite")
    if not isinstance(created_at_raw, str):
        raise GhostChainsValidationError("transactions[].createdAt must be an ISO-8601 string")
    if ip_address is not None and not isinstance(ip_address, str):
        raise GhostChainsValidationError("transactions[].ipAddress must be a string when present")
    if device_id is not None and not isinstance(device_id, str):
        raise GhostChainsValidationError("transactions[].deviceId must be a string when present")

    return Transaction(
        tx_id=tx_id,
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        amount=float(amount),
        created_at=parse_iso8601(created_at_raw),
        ip_address=ip_address,
        device_id=device_id,
    )


class GhostChainsService:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._processed_order: deque[StoredTransaction] = deque()
        self._results_by_tx_id: dict[str, float] = {}
        self._payload_signature_by_tx_id: dict[str, str] = {}
        self._max_created_at_seen = -math.inf

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def clear(self, payload: Any) -> dict[str, bool]:
        if not isinstance(payload, dict) or payload.get("clearTransactions") is not True:
            raise GhostChainsValidationError("Request body must be {\"clearTransactions\": true}")
        self.reset()
        return {"clearTransactions": True}

    def process_transactions(self, payload: Any) -> dict[str, list[dict[str, Any]]]:
        if not isinstance(payload, dict):
            raise GhostChainsValidationError("Request body must be a JSON object")

        raw_transactions = payload.get("transactions")
        if not isinstance(raw_transactions, list):
            raise GhostChainsValidationError("transactions must be an array")

        results: list[dict[str, Any]] = []
        for raw_transaction in raw_transactions:
            transaction = normalize_transaction(raw_transaction)
            signature = self._signature(transaction)

            existing_score = self._results_by_tx_id.get(transaction.tx_id)
            if existing_score is not None:
                results.append({"txId": transaction.tx_id, "riskScore": existing_score})
                continue

            risk_score = self._score_transaction(transaction)
            stored = StoredTransaction(transaction=transaction, risk_score=risk_score)
            self._processed_order.append(stored)
            self._results_by_tx_id[transaction.tx_id] = risk_score
            self._payload_signature_by_tx_id[transaction.tx_id] = signature
            self._max_created_at_seen = max(self._max_created_at_seen, transaction.created_at)
            self._prune_storage()

            results.append({"txId": transaction.tx_id, "riskScore": risk_score})

        return {"transactions": results}

    def _score_transaction(self, transaction: Transaction) -> float:
        if transaction.from_user_id == transaction.to_user_id:
            return 0.0

        prior_transactions = self._active_prior_transactions(transaction.created_at)
        if not prior_transactions:
            return 0.0

        adjacency = _build_adjacency(prior_transactions)
        reverse_adjacency = _build_reverse_adjacency(prior_transactions)
        existing_neighbors = adjacency.get(transaction.from_user_id, set())
        if transaction.to_user_id in existing_neighbors:
            return self._score_repeat_edge(transaction, prior_transactions, adjacency, reverse_adjacency)

        reverse_distances_to_u = _bfs_distances(transaction.from_user_id, reverse_adjacency)
        forward_distances_from_v = _bfs_distances(transaction.to_user_id, adjacency)
        forward_distance_cache: dict[str, dict[str, int]] = {}

        new_reachability_pairs = 0
        new_reachability_compactness = 0.0
        shortened_path_gain = 0.0
        alternative_path_pairs = 0
        alternative_shortest_pairs = 0

        for ancestor, distance_to_u in reverse_distances_to_u.items():
            forward_distances = forward_distance_cache.setdefault(
                ancestor,
                _bfs_distances(ancestor, adjacency),
            )
            for descendant, distance_from_v in forward_distances_from_v.items():
                candidate_distance = distance_to_u + 1 + distance_from_v
                old_distance = forward_distances.get(descendant)

                if old_distance is None:
                    new_reachability_pairs += 1
                    new_reachability_compactness += 1.0 / candidate_distance
                    continue

                if candidate_distance < old_distance:
                    shortened_path_gain += old_distance - candidate_distance
                    continue

                alternative_path_pairs += 1
                if candidate_distance == old_distance:
                    alternative_shortest_pairs += 1

        closes_return_path = transaction.from_user_id in forward_distances_from_v
        cycle_component_size = 0
        cycle_compactness = 0.0
        scc_growth = 0
        if closes_return_path:
            cycle_nodes = set(reverse_distances_to_u) & set(forward_distances_from_v)
            cycle_component_size = len(cycle_nodes)
            cycle_length = forward_distances_from_v[transaction.from_user_id] + 1
            cycle_compactness = cycle_component_size / cycle_length
            scc_growth = max(
                0,
                cycle_component_size
                - max(
                    _scc_size(transaction.from_user_id, adjacency, reverse_adjacency),
                    _scc_size(transaction.to_user_id, adjacency, reverse_adjacency),
                ),
            )

        if (
            new_reachability_pairs == 0
            and shortened_path_gain <= EPS
            and alternative_path_pairs == 0
            and scc_growth == 0
        ):
            return 0.0

        raw_score = (
            0.16 * math.log1p(new_reachability_pairs)
            + 0.18 * math.log1p(new_reachability_compactness)
            + 0.34 * math.log1p(shortened_path_gain)
            + 0.18 * math.log1p(alternative_path_pairs)
            + 0.45 * math.log1p(alternative_shortest_pairs)
            + 0.45 * math.log1p(scc_growth)
            + 0.40 * math.log1p(max(0, cycle_component_size - 1))
            + 0.28 * cycle_compactness
        )
        score = 1.0 - math.exp(-raw_score / 1.9)
        return round(max(0.0, min(1.0, score)), 6)

    def _score_repeat_edge(
        self,
        transaction: Transaction,
        prior_transactions: list[Transaction],
        adjacency: dict[str, set[str]],
        reverse_adjacency: dict[str, set[str]],
    ) -> float:
        repeat_count = sum(
            1
            for prior in prior_transactions
            if prior.from_user_id == transaction.from_user_id
            and prior.to_user_id == transaction.to_user_id
        )

        forward_distances_from_v = _bfs_distances(transaction.to_user_id, adjacency)
        closes_return_path = transaction.from_user_id in forward_distances_from_v

        cycle_component_size = 0
        cycle_compactness = 0.0
        if closes_return_path:
            reverse_distances_to_u = _bfs_distances(transaction.from_user_id, reverse_adjacency)
            cycle_nodes = set(reverse_distances_to_u) & set(forward_distances_from_v)
            cycle_component_size = len(cycle_nodes)
            cycle_length = forward_distances_from_v[transaction.from_user_id] + 1
            cycle_compactness = cycle_component_size / cycle_length

        raw_score = (
            REPEAT_EDGE_WEIGHT * math.log1p(repeat_count)
            + 0.40 * math.log1p(max(0, cycle_component_size - 1))
            + 0.28 * cycle_compactness
        )
        score = 1.0 - math.exp(-raw_score / 1.9)
        return round(max(0.0, min(1.0, score)), 6)

    def _active_prior_transactions(self, created_at: float) -> list[Transaction]:
        earliest = created_at - LOOKBACK_WINDOW_SECONDS
        return [
            stored.transaction
            for stored in self._processed_order
            if stored.transaction.created_at >= earliest - EPS
        ]

    def _prune_storage(self) -> None:
        if self._max_created_at_seen == -math.inf:
            return

        threshold = self._max_created_at_seen - LOOKBACK_WINDOW_SECONDS
        retained: deque[StoredTransaction] = deque()
        for stored in self._processed_order:
            if stored.transaction.created_at >= threshold - EPS:
                retained.append(stored)
            else:
                tx_id = stored.transaction.tx_id
                self._results_by_tx_id.pop(tx_id, None)
                self._payload_signature_by_tx_id.pop(tx_id, None)
        self._processed_order = retained

    def _signature(self, transaction: Transaction) -> str:
        normalized = {
            "txId": transaction.tx_id,
            "fromUserId": transaction.from_user_id,
            "toUserId": transaction.to_user_id,
            "amount": transaction.amount,
            "createdAt": transaction.created_at,
            "ipAddress": transaction.ip_address,
            "deviceId": transaction.device_id,
        }
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _build_adjacency(transactions: list[Transaction]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for transaction in transactions:
        adjacency.setdefault(transaction.from_user_id, set()).add(transaction.to_user_id)
        adjacency.setdefault(transaction.to_user_id, set())
    return adjacency


def _build_reverse_adjacency(transactions: list[Transaction]) -> dict[str, set[str]]:
    reverse_adjacency: dict[str, set[str]] = {}
    for transaction in transactions:
        reverse_adjacency.setdefault(transaction.to_user_id, set()).add(transaction.from_user_id)
        reverse_adjacency.setdefault(transaction.from_user_id, set())
    return reverse_adjacency


def _reachable_nodes(start: str, adjacency: dict[str, set[str]]) -> set[str]:
    seen: set[str] = {start}
    stack = [start]

    while stack:
        current = stack.pop()
        for neighbor in adjacency.get(current, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)

    return seen


def _reverse_reachable_nodes(start: str, reverse_adjacency: dict[str, set[str]]) -> set[str]:
    seen: set[str] = {start}
    stack = [start]

    while stack:
        current = stack.pop()
        for neighbor in reverse_adjacency.get(current, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)

    return seen


def _bfs_distances(start: str, adjacency: dict[str, set[str]]) -> dict[str, int]:
    distances: dict[str, int] = {start: 0}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for neighbor in adjacency.get(current, set()):
            if neighbor not in distances:
                distances[neighbor] = next_distance
                queue.append(neighbor)

    return distances


def _scc_size(node: str, adjacency: dict[str, set[str]], reverse_adjacency: dict[str, set[str]]) -> int:
    return len(_reachable_nodes(node, adjacency) & _reverse_reachable_nodes(node, reverse_adjacency))
