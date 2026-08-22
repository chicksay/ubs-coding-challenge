# Toolbox Phase 1 Completion + Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `toolbox/toolbox.py` actually runnable (it currently imports two modules that don't exist) and add the two Stage-2 ("School Days") MCP tools: exam-material recall under a 900-token budget, and tolled shortest-path map navigation with an optional hop budget.

**Architecture:** One Flask app (`toolbox/routes.py`) serving one `/mcp` JSON-RPC endpoint (already implemented in `toolbox/toolbox.py`, unmodified except for two small additive edits). Phase 1 tool logic lives in `toolbox/services/nursery.py`; Phase 2 tool logic lives in `toolbox/services/exam.py` and `toolbox/services/mapnav.py`, registered via `toolbox/phase2.py` and merged into `toolbox.py`'s existing `TOOLS`/`HANDLERS`.

**Tech Stack:** Python 3, Flask, `requests`, `tiktoken` (`o200k_base` encoding), Pillow + numpy for image classification. `unittest` for tests (matches this repo's existing convention — see `test_ghost_chains.py`).

**Spec:** `docs/superpowers/specs/2026-08-22-toolbox-phase2-design.md`, `toolbox/challenge statements/toolbox.txt`, `toolbox/challenge statements/toolbox_phase2.txt`, `toolbox/challenge statements/run-summary.txt`.

## Global Constraints

- All imports across `toolbox/` are flat (`from routes import app`, `from services import nursery`, `import toolbox`) — every command in this plan must run **from inside the `toolbox/` directory**, not the repo root.
- Deployment (and local testing) must stay **single-process** — `services/exam.py` and `services/mapnav.py` hold in-memory caches/journey-state that would desync across multiple worker processes. Never add gunicorn multi-worker config.
- No network calls at import time — only lazily inside a handler, on first need.
- `toolbox.py`'s existing JSON-RPC dispatch machinery (`_dispatch`, `_handle_method`, `_call_tool`, error shapes) must not change — only `TOOLS`, `HANDLERS`, `INSTRUCTIONS`, and `_format_value` get additive edits.
- Per this repo's convention (see CLAUDE.md), **never run `git commit`** — every task ends with `git add` only, staged for the developer to commit.
- New dependency versions aren't pinned beyond what's already installed locally (`tiktoken` 0.12.0, `requests` 2.32.5, `pillow` 11.3.0, `numpy` 2.3.2); `flask` has no version constraint from this environment since it isn't installed yet — plain `flask` in `requirements.txt` is fine.

---

### Task 1: Flask scaffold + Phase 1 nursery tools

**Files:**
- Create: `toolbox/routes.py`
- Create: `toolbox/requirements.txt`
- Create: `toolbox/server.py`
- Create: `toolbox/services/__init__.py`
- Create: `toolbox/services/nursery.py`
- Test: `toolbox/test_nursery.py`
- Test: `toolbox/test_server_smoke.py`

**Interfaces:**
- Produces: `services.nursery.get_name(args: dict) -> str`, `services.nursery.calculate(args: dict) -> int|float`, `services.nursery.identify_shape(args: dict) -> str`, `services.nursery.count_sides(args: dict) -> int`, `services.nursery.order_of_operations(args: dict) -> str`. `routes.app` — the shared Flask app singleton every later task imports.

- [ ] **Step 1: Create `toolbox/routes.py`**

```python
from flask import Flask

app = Flask(__name__)
```

- [ ] **Step 2: Create `toolbox/requirements.txt`**

```
flask
requests
tiktoken
pillow
numpy
```

- [ ] **Step 3: Create `toolbox/services/__init__.py`** (empty file)

- [ ] **Step 4: Write the failing tests for `calculate` and `get_name`**

`toolbox/test_nursery.py`:

```python
import base64
import io
import unittest

from PIL import Image, ImageDraw

from services import nursery


def _encode(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _canvas(size=100):
    return Image.new("RGB", (size, size), "white")


class NameTests(unittest.TestCase):
    def test_get_name_meets_criteria(self):
        name = nursery.get_name({})
        self.assertTrue(3 <= len(name) <= 30)
        self.assertRegex(name, r"^[A-Za-z0-9 _\-']+$")


class CalculateTests(unittest.TestCase):
    def test_symbolic_expression(self):
        self.assertEqual(nursery.calculate({"expression": "2 + 2 + 5"}), 9)

    def test_precedence(self):
        self.assertEqual(nursery.calculate({"expression": "2 + 3 * 4"}), 14)

    def test_natural_language(self):
        self.assertEqual(
            nursery.calculate({"expression": "What is 2 plus 3 times 4?"}), 14
        )

    def test_negative_operand(self):
        self.assertEqual(nursery.calculate({"expression": "-5 + 10"}), 5)

    def test_division(self):
        self.assertEqual(nursery.calculate({"expression": "10 / 2"}), 5)

    def test_a_op_b_fallback(self):
        self.assertEqual(nursery.calculate({"a": 3, "op": "*", "b": 4}), 12)

    def test_rejects_non_arithmetic(self):
        with self.assertRaises(ValueError):
            nursery.calculate({"expression": "__import__('os')"})


class ShapeTests(unittest.TestCase):
    def test_rectangle(self):
        image = _canvas()
        ImageDraw.Draw(image).rectangle([20, 20, 80, 80], fill="black")
        self.assertEqual(nursery.identify_shape({"image": _encode(image)}), "rectangle")

    def test_circle(self):
        image = _canvas()
        ImageDraw.Draw(image).ellipse([20, 20, 80, 80], fill="black")
        self.assertEqual(nursery.identify_shape({"image": _encode(image)}), "circle")

    def test_triangle(self):
        image = _canvas()
        ImageDraw.Draw(image).polygon([(50, 15), (15, 85), (85, 85)], fill="black")
        self.assertEqual(nursery.identify_shape({"image": _encode(image)}), "triangle")

    def test_count_sides_from_shape_name(self):
        self.assertEqual(nursery.count_sides({"shape": "Triangle"}), 3)
        self.assertEqual(nursery.count_sides({"shape": "rectangle"}), 4)
        self.assertEqual(nursery.count_sides({"shape": "circle"}), 0)

    def test_count_sides_from_image(self):
        image = _canvas()
        ImageDraw.Draw(image).rectangle([20, 20, 80, 80], fill="black")
        self.assertEqual(nursery.count_sides({"image": _encode(image)}), 4)


class OrderOfOperationsTests(unittest.TestCase):
    def test_returns_a_string(self):
        self.assertIsInstance(nursery.order_of_operations({}), str)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 5: Run the test to verify it fails**

Run (from inside `toolbox/`): `python -m unittest test_nursery -v`
Expected: FAIL/ERROR — `services.nursery` doesn't exist yet (`ModuleNotFoundError`).

- [ ] **Step 6: Implement `toolbox/services/nursery.py`**

```python
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
```

- [ ] **Step 7: Run the tests to verify they pass**

Run (from inside `toolbox/`): `python -m unittest test_nursery -v`
Expected: all tests PASS.

- [ ] **Step 8: Create `toolbox/server.py`**

```python
import os

from routes import app
import toolbox  # noqa: F401 - imported for its /mcp route registration

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)
```

- [ ] **Step 9: Write the failing smoke test**

`toolbox/test_server_smoke.py`:

```python
import unittest

from routes import app
import toolbox  # noqa: F401 - registers /mcp


class ServerSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_tools_list_returns_phase1_tools(self):
        response = self.client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )
        names = {tool["name"] for tool in response.get_json()["result"]["tools"]}
        self.assertEqual(
            names,
            {"get_name", "calculate", "identify_shape", "order_of_operations", "count_sides"},
        )

    def test_calculate_tool_call_round_trip(self):
        response = self.client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "calculate", "arguments": {"expression": "2 + 3 * 4"}},
            },
        )
        result = response.get_json()["result"]
        self.assertEqual(result["structuredContent"]["answer"], 14)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 10: Install Flask, then run the smoke test to verify it fails, then passes**

Run: `pip install flask`
Run (from inside `toolbox/`): `python -m unittest test_server_smoke -v`
Expected: PASS (Task 1 has no missing pieces left once `flask` is installed — this step is really "confirm the scaffold works end to end", not a strict red-then-green cycle, since `routes.py`/`server.py` have no separate failing state to observe).

- [ ] **Step 11: Stage the changes**

```bash
git add toolbox/routes.py toolbox/requirements.txt toolbox/server.py toolbox/services/__init__.py toolbox/services/nursery.py toolbox/test_nursery.py toolbox/test_server_smoke.py
```

---

### Task 2: Exam recall (`services/exam.py`)

**Files:**
- Create: `toolbox/services/exam.py`
- Test: `toolbox/test_exam.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `services.exam.recall_study_material(args: dict) -> list[str]`, `services.exam._select_passages(question: str, passages: list[str]) -> list[str]`, `services.exam._passages` (module-level cache, `None` until first fetched — tests set it directly to skip network), `services.exam.TOKEN_BUDGET = 900`, `services.exam._ENCODING` (the `tiktoken` `o200k_base` encoding). Task 4 imports `recall_study_material`.

- [ ] **Step 1: Write the failing tests**

`toolbox/test_exam.py`:

```python
import unittest

from services import exam


class SelectPassagesTests(unittest.TestCase):
    def test_stays_within_token_budget(self):
        passages = ["word " * 200 for _ in range(20)]
        selected = exam._select_passages("word", passages)
        total = sum(len(exam._ENCODING.encode(p)) for p in selected)
        self.assertLessEqual(total, exam.TOKEN_BUDGET)

    def test_prefers_relevant_passages(self):
        passages = [
            "The sensor grid was last aligned on 14 March.",
            "The cafeteria menu rotates every two weeks.",
        ]
        selected = exam._select_passages("When was the sensor grid aligned?", passages)
        self.assertEqual(selected[0], passages[0])

    def test_skips_oversized_passage_to_fit_smaller_ones(self):
        huge = "sensor grid alignment fact " * 400
        small = "sensor grid alignment fact recorded here"
        selected = exam._select_passages("sensor grid alignment fact", [huge, small])
        self.assertIn(small, selected)
        self.assertNotIn(huge, selected)


class RecallStudyMaterialTests(unittest.TestCase):
    def setUp(self):
        self._original = exam._passages
        exam._passages = [
            "The sensor grid was last aligned on 14 March.",
            "Singapore has an efficient public transit system.",
        ]

    def tearDown(self):
        exam._passages = self._original

    def test_returns_list_of_strings(self):
        result = exam.recall_study_material({"question": "sensor grid alignment"})
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(p, str) for p in result))

    def test_requires_question(self):
        with self.assertRaises(ValueError):
            exam.recall_study_material({})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run (from inside `toolbox/`): `python -m unittest test_exam -v`
Expected: FAIL — `services.exam` doesn't exist yet.

- [ ] **Step 3: Implement `toolbox/services/exam.py`**

```python
"""Phase 2 (School Days) Problem Set 1: exam-material recall under a
900-token (o200k_base) budget."""

import json
import logging
import os
import re
import threading
from urllib.parse import urljoin

import requests
import tiktoken

logger = logging.getLogger(__name__)

API_BASE = os.environ.get("TOOLBOX_API_BASE", "https://tool-box-2591eaa24fa3.herokuapp.com")
_ENCODING = tiktoken.get_encoding("o200k_base")
_FETCH_TIMEOUT = 5
TOKEN_BUDGET = 900
_MAX_PASSAGE_TOKENS = 220

_lock = threading.Lock()
_passages = None


def _fetch_json_or_text(url, **params):
    response = requests.get(url, params=params or None, timeout=_FETCH_TIMEOUT)
    response.raise_for_status()
    try:
        return response.json()
    except ValueError:
        return response.text


def _extract_addresses(listing):
    if isinstance(listing, dict):
        for key in ("materials", "documents", "items", "data"):
            if isinstance(listing.get(key), list):
                listing = listing[key]
                break
    if not isinstance(listing, list):
        raise ValueError("unexpected study-materials listing shape")

    addresses = []
    for entry in listing:
        if isinstance(entry, str):
            addresses.append(entry)
        elif isinstance(entry, dict):
            for key in ("address", "url", "path", "href"):
                if entry.get(key):
                    addresses.append(entry[key])
                    break
    return addresses


def _document_text(payload):
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("content", "text", "body", "material"):
            if isinstance(payload.get(key), str):
                return payload[key]
    return json.dumps(payload)


def _split_long_passage(text, max_tokens=_MAX_PASSAGE_TOKENS):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current, current_tokens = [], 0
    for sentence in sentences:
        tokens = len(_ENCODING.encode(sentence))
        if current and current_tokens + tokens > max_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def _passages_from_text(text):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    passages = []
    for paragraph in paragraphs:
        if len(_ENCODING.encode(paragraph)) <= _MAX_PASSAGE_TOKENS:
            passages.append(paragraph)
        else:
            passages.extend(_split_long_passage(paragraph))
    return passages


def _load_materials():
    listing = _fetch_json_or_text(urljoin(API_BASE + "/", "study-materials"))
    addresses = _extract_addresses(listing)
    passages = []
    for address in addresses:
        url = urljoin(API_BASE + "/", address)
        try:
            payload = _fetch_json_or_text(url)
        except requests.RequestException:
            logger.warning("failed to fetch study material %s", url)
            continue
        passages.extend(_passages_from_text(_document_text(payload)))
    return passages


def _materials():
    global _passages
    with _lock:
        if _passages is None:
            try:
                _passages = _load_materials()
            except requests.RequestException:
                logger.warning("failed to fetch study-materials listing")
                _passages = []
        return _passages


def _score(question, passage):
    q_words = set(re.findall(r"[a-z0-9]+", question.lower()))
    p_words = set(re.findall(r"[a-z0-9]+", passage.lower()))
    if not q_words or not p_words:
        return 0
    return len(q_words & p_words)


def _select_passages(question, passages):
    ranked = sorted(passages, key=lambda p: _score(question, p), reverse=True)
    selected = []
    total_tokens = 0
    for passage in ranked:
        tokens = len(_ENCODING.encode(passage))
        if total_tokens + tokens > TOKEN_BUDGET:
            continue
        selected.append(passage)
        total_tokens += tokens
    return selected


def recall_study_material(args):
    question = str(args.get("question") or "").strip()
    if not question:
        raise ValueError("question is required")
    passages = _materials()
    if not passages:
        return []
    return _select_passages(question, passages)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run (from inside `toolbox/`): `python -m unittest test_exam -v`
Expected: all tests PASS.

- [ ] **Step 5: Stage the changes**

```bash
git add toolbox/services/exam.py toolbox/test_exam.py
```

---

### Task 3: Map navigation (`services/mapnav.py`)

**Files:**
- Create: `toolbox/services/mapnav.py`
- Test: `toolbox/test_mapnav.py`

**Interfaces:**
- Consumes: nothing from Tasks 1–2.
- Produces: `services.mapnav.next_hop(args: dict) -> str`, `services.mapnav._fetch_graph(map_id) -> (adjacency: dict, tolls: dict)` (patched in tests to avoid network), `services.mapnav._graphs` / `services.mapnav._journeys` (module-level caches — tests clear these in `setUp`). Task 4 imports `next_hop`.

- [ ] **Step 1: Write the failing tests**

`toolbox/test_mapnav.py`:

```python
import unittest
from unittest import mock

from services import mapnav


TOLL_ADJ = {"A": {"B": 4.0, "C": 2.0}, "B": {"D": 3.0}, "C": {"D": 2.0}}
TOLL_TOLLS = {"A": 5.0, "B": 1.0, "C": 9.0, "D": 2.0}

BOUNDED_ADJ = {
    "S": {"X": 1.0, "Z": 10.0},
    "X": {"Y": 1.0},
    "Y": {"D": 1.0},
    "Z": {"D": 1.0},
}
BOUNDED_TOLLS = {"S": 0.0, "X": 0.0, "Y": 0.0, "Z": 0.0, "D": 0.0}

REVISIT_ADJ = {"A": {"B": 1.0}, "B": {"A": 1.0, "D": 1.0}}
REVISIT_TOLLS = {"A": 0.0, "B": 0.0, "D": 0.0}


def _patched(adjacency, tolls):
    mapnav._graphs.clear()
    mapnav._journeys.clear()
    return mock.patch.object(mapnav, "_fetch_graph", return_value=(adjacency, tolls))


class DijkstraTests(unittest.TestCase):
    def setUp(self):
        patcher = _patched(TOLL_ADJ, TOLL_TOLLS)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_picks_cheaper_toll_inclusive_path(self):
        result = mapnav.next_hop({"map_id": "m1", "start": "A", "current": "A", "destination": "D"})
        self.assertEqual(result, "B")

    def test_full_journey_reaches_destination(self):
        node = mapnav.next_hop({"map_id": "m1", "start": "A", "current": "A", "destination": "D"})
        self.assertEqual(node, "B")
        node = mapnav.next_hop({"map_id": "m1", "start": "A", "current": node, "destination": "D"})
        self.assertEqual(node, "D")

    def test_never_returns_non_adjacent_node(self):
        node = mapnav.next_hop({"map_id": "m1", "start": "A", "current": "A", "destination": "D"})
        self.assertIn(node, TOLL_ADJ["A"])


class HopBudgetTests(unittest.TestCase):
    def setUp(self):
        patcher = _patched(BOUNDED_ADJ, BOUNDED_TOLLS)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_unconstrained_choice_needs_three_hops(self):
        node = mapnav.next_hop({"map_id": "m2", "start": "S", "current": "S", "destination": "D"})
        self.assertEqual(node, "X")

    def test_bounded_choice_switches_to_the_only_two_hop_route(self):
        node = mapnav.next_hop(
            {"map_id": "m3", "start": "S", "current": "S", "destination": "D", "hops_remaining": 2}
        )
        self.assertEqual(node, "Z")


class RevisitAvoidanceTests(unittest.TestCase):
    def setUp(self):
        patcher = _patched(REVISIT_ADJ, REVISIT_TOLLS)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_does_not_revisit_the_starting_node(self):
        first = mapnav.next_hop({"map_id": "m4", "start": "A", "current": "A", "destination": "D"})
        self.assertEqual(first, "B")
        second = mapnav.next_hop({"map_id": "m4", "start": "A", "current": first, "destination": "D"})
        self.assertEqual(second, "D")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run (from inside `toolbox/`): `python -m unittest test_mapnav -v`
Expected: FAIL — `services.mapnav` doesn't exist yet.

- [ ] **Step 3: Implement `toolbox/services/mapnav.py`**

```python
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
    else:
        result = _shortest_next_hop(adjacency, tolls, current, destination, visited)

    if result is None:
        result = _fallback_next_hop(adjacency, current, visited)
    if result is None:
        raise ValueError("no reachable next hop from {}".format(current))

    _mark_visited(key, result, destination)
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run (from inside `toolbox/`): `python -m unittest test_mapnav -v`
Expected: all tests PASS.

- [ ] **Step 5: Stage the changes**

```bash
git add toolbox/services/mapnav.py toolbox/test_mapnav.py
```

---

### Task 4: Wire Phase 2 into `toolbox.py`

**Files:**
- Create: `toolbox/phase2.py`
- Modify: `toolbox/toolbox.py:8` (import), `toolbox/toolbox.py:20-28` (`INSTRUCTIONS`), `toolbox/toolbox.py:118` (after `TOOLS` list), `toolbox/toolbox.py:126` (after `HANDLERS` dict), `toolbox/toolbox.py:297-302` (`_format_value`)
- Test: `toolbox/test_toolbox_integration.py`

**Interfaces:**
- Consumes: `services.exam.recall_study_material`, `services.mapnav.next_hop` (Tasks 2–3); `routes.app`, existing `toolbox.py` dispatch (Task 1).
- Produces: `toolbox.TOOLS` and `toolbox.HANDLERS` extended with `recall_study_material` and `next_hop`; end-to-end `/mcp` behavior other agents/graders hit directly.

- [ ] **Step 1: Create `toolbox/phase2.py`**

```python
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
```

- [ ] **Step 2: Write the failing integration test**

`toolbox/test_toolbox_integration.py`:

```python
import unittest
from unittest import mock

from routes import app
import toolbox  # noqa: F401 - registers /mcp
from services import exam, mapnav


class ToolboxIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_tools_list_includes_all_seven_tools(self):
        response = self.client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )
        names = {tool["name"] for tool in response.get_json()["result"]["tools"]}
        self.assertEqual(
            names,
            {
                "get_name",
                "calculate",
                "identify_shape",
                "order_of_operations",
                "count_sides",
                "recall_study_material",
                "next_hop",
            },
        )

    def test_recall_study_material_tool_call_returns_list_answer(self):
        original = exam._passages
        exam._passages = ["The sensor grid was last aligned on 14 March."]
        try:
            response = self.client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "recall_study_material",
                        "arguments": {"question": "sensor grid alignment"},
                    },
                },
            )
            result = response.get_json()["result"]
            self.assertIsInstance(result["structuredContent"]["answer"], list)
            self.assertIn("sensor grid", result["content"][0]["text"])
        finally:
            exam._passages = original

    def test_next_hop_tool_call_returns_adjacent_node(self):
        adjacency = {"A": {"B": 1.0}, "B": {"D": 1.0}}
        tolls = {"A": 0.0, "B": 0.0, "D": 0.0}
        mapnav._graphs.clear()
        mapnav._journeys.clear()
        with mock.patch.object(mapnav, "_fetch_graph", return_value=(adjacency, tolls)):
            response = self.client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "next_hop",
                        "arguments": {
                            "map_id": "m1",
                            "start": "A",
                            "current": "A",
                            "destination": "D",
                        },
                    },
                },
            )
            result = response.get_json()["result"]
            self.assertEqual(result["structuredContent"]["answer"], "B")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the test to verify it fails**

Run (from inside `toolbox/`): `python -m unittest test_toolbox_integration -v`
Expected: FAIL — `tools/list` only returns the 5 Phase 1 tools; `recall_study_material`/`next_hop` are unknown tools.

- [ ] **Step 4: Add the phase2 import to `toolbox.py`**

`toolbox/toolbox.py:8`, before:
```python
from services import nursery
```
after:
```python
from services import nursery

from phase2 import HANDLERS as PHASE2_HANDLERS, TOOLS as PHASE2_TOOLS
```

- [ ] **Step 5: Extend `INSTRUCTIONS` in `toolbox.py`**

`toolbox/toolbox.py:20-28`, before:
```python
INSTRUCTIONS = (
    "You are a nursery agent. Never guess. Call a tool for every question. "
    "Use get_name for your name. "
    "Use calculate with the FULL expression (e.g. 2 + 2 + 5 or 2 + 3 * 4). "
    "* and / happen before + and -. "
    "Use identify_shape once per distinct base64 PNG; it returns rectangle, triangle, or circle. "
    "When the question lists what each shape is worth, use those stated values, "
    "multiply each by how many of that shape there are, and add the results with calculate."
)
```
after:
```python
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
```

- [ ] **Step 6: Merge `TOOLS` and `HANDLERS`**

`toolbox/toolbox.py:118`, immediately after the `TOOLS = [` list's closing `]`, add:
```python
TOOLS = TOOLS + PHASE2_TOOLS
```

`toolbox/toolbox.py:126`, immediately after the `HANDLERS = {` dict's closing `}`, add:
```python
HANDLERS.update(PHASE2_HANDLERS)
```

- [ ] **Step 7: Add list handling to `_format_value`**

`toolbox/toolbox.py:297-302`, before:
```python
def _format_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".10g")
    return str(value)
```
after:
```python
def _format_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".10g")
    if isinstance(value, list):
        return "\n\n".join(str(item) for item in value)
    return str(value)
```

- [ ] **Step 8: Run the full test suite to verify everything passes**

Run (from inside `toolbox/`): `python -m unittest discover -p "test_*.py" -v`
Expected: all tests across `test_nursery.py`, `test_server_smoke.py`, `test_exam.py`, `test_mapnav.py`, `test_toolbox_integration.py` PASS.

- [ ] **Step 9: Stage the changes**

```bash
git add toolbox/phase2.py toolbox/toolbox.py toolbox/test_toolbox_integration.py
```

---

## Self-Review

**Spec coverage:**
- Phase 1 `get_name`/`calculate`/`identify_shape`/`count_sides`/`order_of_operations` → Task 1.
- Phase 1 Combo Challenge (Problem Set 4) and Phase 2 School Trip (Problem Set 3) → no task; spec text says these are the agent orchestrating already-built tools, nothing to implement.
- Phase 2 Exam recall, 900-token `o200k_base` budget summed across returned passages → Task 2 (`_select_passages` never lets the running total exceed `TOKEN_BUDGET`).
- Phase 2 Map navigation, toll-inclusive cost, hop budget, no-revisit, no-non-adjacent → Task 3 (`_shortest_next_hop`/`_bounded_next_hop` both read only from `adjacency[current]`, both exclude `visited`).
- Merging into one `/mcp` endpoint, list-typed tool output rendering → Task 4.
- Single-process constraint → called out in Global Constraints; `server.py` (Task 1) never invokes gunicorn/multiple workers.

**Placeholder scan:** none — every step has complete, runnable code.

**Type consistency:** `nursery.get_name/calculate/identify_shape/count_sides/order_of_operations` all take a single `args: dict` and match `HANDLERS` call site `value = handler(arguments)` in `toolbox.py:283`. `exam.recall_study_material` and `mapnav.next_hop` both take a single `args: dict` too, matching the same call site via `phase2.HANDLERS`. `phase2.TOOLS`/`phase2.HANDLERS` names match exactly what `toolbox.py` imports (`PHASE2_TOOLS`/`PHASE2_HANDLERS`) and merges.

**Scope check:** four tasks, each independently testable (Task 1 stands alone as a running Phase-1-only server; Tasks 2–3 are pure-logic modules testable without Flask; Task 4 is the integration). No task depends on a later one.
