import json
import logging
import uuid

from flask import Response, jsonify, request

from routes import app
from services import nursery

from phase2 import HANDLERS as PHASE2_HANDLERS, TOOLS as PHASE2_TOOLS

logger = logging.getLogger(__name__)

SUPPORTED_VERSIONS = (
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    "2025-11-25",
    "2026-07-28",
)
SERVER_INFO = {"name": "nursery-toolbox", "version": "1.0.0"}
INSTRUCTIONS = (
    "You are a nursery agent. Never guess. Call a tool for every question. "
    "Use get_name for your name. "
    "Use calculate with the FULL expression (e.g. 2 + 2 + 5 or 2 + 3 * 4). "
    "* and / happen before + and -. "
    "Use identify_shape once per distinct base64 PNG; it returns rectangle, triangle, or circle. "
    "When the question lists what each shape is worth, use those stated values, "
    "multiply each by how many of that shape there are, and add the results with calculate. "
    "Use recall_study_material for any question about the study/exam material, then answer "
    "from the passages it returns. "
    "Use next_hop to travel a map: call it, move to the node it returns, then call it again "
    "with that node as the new current, repeating until it returns the destination."
)

TOOLS = [
    {
        "name": "get_name",
        "description": (
            "Return this agent's name as a string. Call when asked "
            "'What is your name?' or any name question."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate the full arithmetic expression, including more than two numbers. "
            "Operators + - * /. * and / bind tighter than + and -. "
            "Operands are integers from -100 to 100. "
            "Examples: 2 + 2 + 5 = 9; 2 + 3 * 4 = 14. "
            "Pass the whole question in expression. Returns a number."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Full expression such as '2 + 2 + 5' or 'What is 2 plus 3 times 4?'",
                },
                "a": {"type": "number", "description": "Left integer operand"},
                "op": {
                    "type": "string",
                    "description": "One of + - * /",
                    "enum": ["+", "-", "*", "/"],
                },
                "b": {"type": "number", "description": "Right integer operand"},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "identify_shape",
        "description": (
            "Classify a base64-encoded PNG toy as exactly one of: "
            "rectangle, triangle, circle. Call when asked 'What shape is this?'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Base64 PNG, with or without a data: URL prefix",
                },
                "image_base64": {"type": "string"},
                "png": {"type": "string"},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "order_of_operations",
        "description": (
            "In an expression, * and / bind more tightly than + and -. "
            "Work those out first. Call before answering a mixed + * expression."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        },
    },
    {
        "name": "count_sides",
        "description": (
            "How many straight sides a toy has: triangle=3, rectangle=4, circle=0. "
            "Geometry only. These are NOT point values -- if the question says what "
            "each shape is worth, use the values from the question instead. "
            "Pass a base64 PNG or a shape name."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "Base64 PNG"},
                "shape": {"type": "string", "description": "rectangle, triangle, or circle"},
            },
            "additionalProperties": True,
        },
    },
]
TOOLS = TOOLS + PHASE2_TOOLS

HANDLERS = {
    "get_name": nursery.get_name,
    "calculate": nursery.calculate,
    "identify_shape": nursery.identify_shape,
    "order_of_operations": nursery.order_of_operations,
    "count_sides": nursery.count_sides,
}
HANDLERS.update(PHASE2_HANDLERS)


@app.after_request
def _mcp_cors(response):
    if request.path.rstrip("/") == "/mcp":
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, GET, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Accept, MCP-Protocol-Version, Mcp-Session-Id, Last-Event-ID"
        )
        response.headers["Access-Control-Expose-Headers"] = (
            "Mcp-Session-Id, MCP-Protocol-Version"
        )
    return response


@app.route("/mcp", methods=["POST", "GET", "DELETE", "OPTIONS"])
@app.route("/mcp/", methods=["POST", "GET", "DELETE", "OPTIONS"])
def mcp_endpoint():
    if request.method == "OPTIONS":
        return ("", 204)
    if request.method == "DELETE":
        return ("", 200)
    if request.method == "GET":
        return _mcp_get()

    data = request.get_json(force=True, silent=True)
    logging.info("data sent for evaluation {}".format(data))
    if data is None:
        return _rpc_error(None, -32700, "Parse error"), 400

    if isinstance(data, list):
        results = [_dispatch(item) for item in data]
        body = [item for item in results if item is not None]
        logging.info("My result : {}".format(body))
        return _json_rpc(body)

    result = _dispatch(data)
    if result is None:
        return ("", 202)
    logging.info("My result : {}".format(result))
    return _json_rpc(result)


def _mcp_get():
    accept = request.headers.get("Accept", "")
    if "text/event-stream" in accept:
        session = request.headers.get("Mcp-Session-Id") or str(uuid.uuid4())
        payload = (
            "event: endpoint\n"
            "data: /mcp\n\n"
            "event: message\n"
            "data: {}\n\n".format(
                json.dumps({"jsonrpc": "2.0", "method": "ping"})
            )
        )
        response = Response(payload, mimetype="text/event-stream")
        response.headers["Mcp-Session-Id"] = session
        response.headers["Cache-Control"] = "no-cache"
        return response
    return jsonify(
        {"error": "POST JSON-RPC to /mcp", "hint": "initialize, tools/list, tools/call"}
    ), 405


def _dispatch(message):
    if not isinstance(message, dict) or message.get("jsonrpc") not in (None, "2.0"):
        return {
            "jsonrpc": "2.0",
            "id": message.get("id") if isinstance(message, dict) else None,
            "error": {"code": -32600, "message": "Invalid Request"},
        }

    method = message.get("method")
    msg_id = message.get("id", "missing")
    params = message.get("params") or {}
    is_notification = "id" not in message

    try:
        result = _handle_method(method, params)
    except LookupError as exc:
        if is_notification:
            return None
        return {
            "jsonrpc": "2.0",
            "id": None if msg_id == "missing" else msg_id,
            "error": {"code": -32601, "message": str(exc)},
        }
    except ValueError as exc:
        if is_notification:
            return None
        return {
            "jsonrpc": "2.0",
            "id": None if msg_id == "missing" else msg_id,
            "error": {"code": -32602, "message": str(exc)},
        }
    except Exception as exc:
        logger.exception("mcp method failed")
        if is_notification:
            return None
        return {
            "jsonrpc": "2.0",
            "id": None if msg_id == "missing" else msg_id,
            "error": {"code": -32603, "message": str(exc)},
        }

    if is_notification:
        return None
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _handle_method(method, params):
    if method in ("initialize", "server/discover"):
        requested = ""
        if isinstance(params, dict):
            requested = params.get("protocolVersion") or ""
            meta = params.get("_meta") or {}
            requested = requested or meta.get("io.modelcontextprotocol/protocolVersion") or ""
        version = requested if requested in SUPPORTED_VERSIONS else "2025-03-26"
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": INSTRUCTIONS,
            "supportedProtocolVersions": list(SUPPORTED_VERSIONS),
        }
    if method in ("notifications/initialized", "initialized"):
        return {}
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        return _call_tool(params if isinstance(params, dict) else {})
    if method in ("resources/list", "resources/templates/list"):
        return {"resources": []}
    if method == "prompts/list":
        return {"prompts": []}
    raise LookupError("Method not found: {}".format(method))


def _call_tool(params):
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {"expression": arguments, "image": arguments}
    handler = HANDLERS.get(name)
    if handler is None:
        return {
            "content": [{"type": "text", "text": "unknown tool: {}".format(name)}],
            "isError": True,
        }
    try:
        value = handler(arguments)
    except (TypeError, ValueError) as exc:
        return {
            "content": [{"type": "text", "text": str(exc)}],
            "isError": True,
        }
    text = _format_value(value)
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": {"answer": value},
        "isError": False,
    }


def _format_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".10g")
    if isinstance(value, list):
        return "\n\n".join(str(item) for item in value)
    return str(value)


def _json_rpc(payload):
    response = app.response_class(
        response=json.dumps(payload),
        status=200,
        mimetype="application/json",
    )
    session = request.headers.get("Mcp-Session-Id") or str(uuid.uuid4())
    response.headers["Mcp-Session-Id"] = session
    requested = request.headers.get("MCP-Protocol-Version") or "2025-03-26"
    response.headers["MCP-Protocol-Version"] = requested
    return response


def _rpc_error(msg_id, code, message):
    return jsonify(
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message},
        }
    )

