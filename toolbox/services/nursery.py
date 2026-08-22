"""Phase 1 (The Nursery) tool implementations."""

import ast
import base64
import io

import numpy as np
from PIL import Image

import re

AGENT_NAME = "Sprout"

_WORD_OPERATORS = [
    (re.compile(r"\bplus\b"), "+"),
    (re.compile(r"\bminus\b"), "-"),
    (re.compile(r"\bless\b"), "-"),
    (re.compile(r"\bmultiplied by\b"), "*"),
    (re.compile(r"\btimes\b"), "*"),
    (re.compile(r"\bdivided by\b"), "/"),
    (re.compile(r"\bover\b"), "/"),
]

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
    ast.Constant,
)


def get_name(args):
    return AGENT_NAME


def _normalize_expression(text):
    text = text.lower()
    for pattern, symbol in _WORD_OPERATORS:
        text = pattern.sub(symbol, text)
    text = re.sub(r"[^0-9+\-*/(). ]", " ", text)
    return text.strip()


def _safe_eval(node):
    if not isinstance(node, _ALLOWED_NODES):
        raise ValueError("Unsupported expression")
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("Unsupported constant")
    if isinstance(node, ast.UnaryOp):
        value = _safe_eval(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError("Unsupported expression")


def calculate(args):
    expression = args.get("expression")
    if not expression:
        a, op, b = args.get("a"), args.get("op"), args.get("b")
        if a is None or op is None or b is None:
            raise ValueError("expression or a/op/b is required")
        expression = "{} {} {}".format(a, op, b)

    normalized = _normalize_expression(str(expression))
    if not normalized:
        raise ValueError("no arithmetic expression found in: {}".format(expression))
    try:
        tree = ast.parse(normalized, mode="eval")
        return _safe_eval(tree)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("could not evaluate expression: {}".format(expression)) from exc


def _decode_image(args):
    raw = args.get("image") or args.get("image_base64") or args.get("png")
    if not raw:
        raise ValueError("image is required")
    if raw.strip().startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    data = base64.b64decode(raw)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _classify(image):
    arr = np.asarray(image)
    background = arr[0, 0]
    mask = np.any(arr != background, axis=-1)
    if not mask.any():
        raise ValueError("could not find a shape in the image")

    ys, xs = np.nonzero(mask)
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
    ratio = float(mask.sum()) / bbox_area

    if ratio > 0.9:
        return "rectangle"
    if ratio > 0.65:
        return "circle"
    return "triangle"


def identify_shape(args):
    return _classify(_decode_image(args))


_SIDES = {"triangle": 3, "rectangle": 4, "circle": 0}


def count_sides(args):
    shape = args.get("shape")
    shape = str(shape).strip().lower() if shape else identify_shape(args)
    if shape not in _SIDES:
        raise ValueError("unknown shape: {}".format(shape))
    return _SIDES[shape]


def order_of_operations(args):
    return (
        "Multiplication and division are evaluated before addition and "
        "subtraction. Within the same precedence level, operators are "
        "evaluated left to right."
    )
