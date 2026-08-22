"""Phase 2 (School Days) tool definitions, merged into toolbox.py's
TOOLS/HANDLERS."""

from services import exam, mapnav

TOOLS = [
    {
        "name": "recall_study_material",
        "description": (
            "Fetch passages from the school's study material relevant to a "
            "question about revision content. Returns a list of short "
            "passages (never a full document) within a strict token budget "
            "-- read them and write your own answer from what they say. "
            "Call once per distinct question."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The full question being asked about the study material.",
                },
            },
            "required": ["question"],
            "additionalProperties": True,
        },
    },
    {
        "name": "next_hop",
        "description": (
            "Get the next node to travel to on the way from the current "
            "position to a destination on a weighted map. Call this "
            "repeatedly: pass back whatever node it returns as the new "
            "'current' on your next call, until it returns the destination "
            "itself. Always pass the same 'start' (the very first position "
            "named in the question) on every call for the same journey. If "
            "the question tells you how many hops/moves remain, pass that "
            "as hops_remaining (counting the hop about to be taken); omit "
            "it if no allowance was mentioned."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "map_id": {"type": "string", "description": "Opaque map handle from the question."},
                "start": {"type": "string", "description": "The very first node of this journey."},
                "current": {"type": "string", "description": "The node you are standing at right now."},
                "destination": {"type": "string", "description": "The node you are trying to reach."},
                "hops_remaining": {
                    "type": "integer",
                    "description": "Remaining hop allowance, if the question stated one.",
                },
            },
            "required": ["map_id", "start", "current", "destination"],
            "additionalProperties": True,
        },
    },
]

HANDLERS = {
    "recall_study_material": exam.recall_study_material,
    "next_hop": mapnav.next_hop,
}
