import base64
import json

PRIORITY_MAP = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def transform(request: dict) -> dict:
    # request is {"payload": "<base64-encoded JSON>"}; the encoded JSON
    # is the V1 shape ("adaptInput"), we bridge it to the V2 shape ("adaptOutput")
    decoded = base64.b64decode(request["payload"]).decode("utf-8")
    adapt_input = json.loads(decoded)["adaptInput"]

    user = adapt_input["user"]
    priority = adapt_input["metadata"]["priority"]

    return {
        "adaptOutput": {
            "id": user["id"],
            "name": user["fullName"],
            "action": adapt_input["action"].lower(),
            "priority": PRIORITY_MAP[priority],
        }
    }
