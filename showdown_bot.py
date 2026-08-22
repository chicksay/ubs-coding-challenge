def _action_amount(turn):
    min_raise_to = turn.get("min_raise_to")
    max_raise_to = turn.get("max_raise_to")

    if isinstance(min_raise_to, bool) or not isinstance(min_raise_to, (int, float)):
        raise ValueError("min_raise_to is missing or invalid")
    if isinstance(max_raise_to, bool) or not isinstance(max_raise_to, (int, float)):
        raise ValueError("max_raise_to is missing or invalid")
    if min_raise_to > max_raise_to:
        raise ValueError("min_raise_to cannot exceed max_raise_to")

    return min_raise_to


def choose_action(turn: dict) -> dict:
    if not isinstance(turn, dict):
        raise ValueError("turn must be a JSON object")

    legal_actions = turn.get("legal_actions")
    if not isinstance(legal_actions, list) or not legal_actions:
        raise ValueError("legal_actions is missing or empty")

    if "check" in legal_actions:
        return {"action": "check"}

    if "call" in legal_actions:
        return {"action": "call"}

    if "raise" in legal_actions:
        try:
            return {"action": "raise", "amount": _action_amount(turn)}
        except ValueError:
            pass

    if "bet" in legal_actions:
        try:
            return {"action": "bet", "amount": _action_amount(turn)}
        except ValueError:
            pass

    if "fold" in legal_actions:
        return {"action": "fold"}

    raise ValueError("no supported action found in legal_actions")
