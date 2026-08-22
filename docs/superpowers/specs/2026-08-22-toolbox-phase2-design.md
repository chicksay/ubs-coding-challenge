# Toolbox Phase 1 completion + Phase 2 design

**Status:** approved, moving to implementation plan.

**Goal:** `toolbox/toolbox.py` (Stage 1 — The Nursery) references `routes.py` and
`services/nursery.py`, neither of which exists, so the MCP server cannot run at
all today. This spec (1) completes Phase 1 so the server actually starts and
serves the five documented nursery tools, and (2) adds Phase 2 (Stage 2 —
School Days): exam-material recall under a 900-token budget, and tolled
shortest-path map navigation with an optional hop budget.

**Spec sources:** `toolbox/challenge statements/toolbox.txt` (Phase 1),
`toolbox/challenge statements/toolbox_phase2.txt` (Phase 2), `toolbox/challenge
statements/run-summary.txt` (how runs are scored/read).

## Global constraints

- MCP contract, JSON-RPC dispatch (`_dispatch`/`_handle_method`/`_call_tool`),
  and error shapes in `toolbox.py` are already correct and must not change —
  only `TOOLS`, `HANDLERS`, `INSTRUCTIONS`, and `_format_value` are extended.
- One Flask `app`, one `/mcp` endpoint. Phase 2 tools are additive entries in
  the same `TOOLS` list / `HANDLERS` dict, not a second route.
- Deployment must stay **single-process** — `services/exam.py` and
  `services/mapnav.py` hold in-memory caches (fetched study materials, fetched
  graphs, per-journey visited-node state) that would desync across multiple
  worker processes. `server.py` runs `app.run(threaded=True)`; no gunicorn
  multi-worker config is introduced.
- No network calls happen at import time — only lazily, on first tool
  invocation that needs them, so the server starts even if the challenge host
  is briefly unreachable.
- New dependencies: `flask`, `requests`, `tiktoken`, `pillow` — all already
  present in this Python environment except `flask`. `requirements.txt` lives
  under `toolbox/` (the root `requirements.txt` belongs to the unrelated
  stdlib-http-server challenges and stays untouched).

## File structure

```
toolbox/
  toolbox.py           MODIFIED — merge phase2.py's TOOLS/HANDLERS, extend
                        INSTRUCTIONS, extend _format_value for list output
  routes.py             NEW — Flask app factory only
  server.py             NEW — entrypoint: import toolbox + phase2 (registers
                        routes/handlers), then app.run()
  phase2.py             NEW — TOOLS/HANDLERS for exam recall + map navigation
  requirements.txt      NEW
  services/
    __init__.py
    nursery.py           NEW — Phase 1 handlers
    exam.py               NEW — Phase 2 Problem Set 1
    mapnav.py              NEW — Phase 2 Problem Set 2
  test_nursery.py        NEW
  test_exam.py            NEW
  test_mapnav.py            NEW
```

## Phase 1 — `services/nursery.py`

- **`get_name(args)`** — returns a fixed constant name satisfying the
  criteria (3–30 chars; letters/digits/spaces/`_`/`-`/`'`). No input needed.
- **`calculate(args)`** — accepts `expression` (preferred) or `a`/`op`/`b`.
  `expression` may be natural language ("What is 2 plus 3 times 4?"): lower it,
  map word-operators (`plus`→`+`, `minus`/`less`→`-`, `times`/`multiplied by`→`*`,
  `divided by`→`/`) to symbols, strip everything that isn't a digit, operator,
  paren, or `.`/`-` sign, then parse with `ast.parse(..., mode="eval")` and
  evaluate a restricted AST (only `Add`/`Sub`/`Mult`/`Div`/`USub`/`Num`/`Constant`
  nodes allowed — anything else raises `ValueError`, never `eval()`/`exec()`).
  Operands are integers -100..100 per spec; the evaluator itself doesn't need
  to enforce that range (the challenge only sends compliant inputs) but it
  does need correct `*`/`/`-before-`+`/`-` precedence, which `ast` gives for
  free.
- **`identify_shape(args)`** — accepts `image`/`image_base64`/`png` (whichever
  is present), strips a `data:` prefix if present, decodes with
  `PIL.Image.open`, converts to a foreground/background mask (non-background
  pixels — background sampled from the corner pixel), computes
  `filled_pixels / bounding_box_area`. Thresholds: `>0.9` → rectangle,
  `0.65..0.9` → circle (π/4≈0.785 is the analytic fill ratio of a circle
  inscribed in its bounding box), `<0.65` → triangle (analytic ratio 0.5).
- **`count_sides(args)`** — accepts `shape` (name, case-insensitive) directly,
  or `image` (classifies via `identify_shape` first). Maps
  `{triangle: 3, rectangle: 4, circle: 0}`.
- **`order_of_operations(args)`** — returns a fixed explanatory string; no
  input needed.

## Phase 2, Problem Set 1 — Exam recall (`services/exam.py`)

**Tool:** `recall_study_material(question: str) -> List[str]`

- `_materials()` (module-level, lock-guarded, memoized): GET
  `{API_BASE}/study-materials`, parse the listing (list of `{name, address}`
  or similar — implementation reads whatever keys are actually present),
  GET every linked document, split each into paragraph passages (blank-line
  boundaries; a paragraph longer than ~220 `o200k_base` tokens is further
  split on sentence boundaries so no single passage can dominate the whole
  budget). Cached forever for the process lifetime — study material content
  doesn't change mid-run.
- On each call: score every cached passage against the question via lexical
  overlap (lowercased word-set Jaccard/overlap count — no embedding model
  available), sort descending, greedily accept passages while
  `sum(len(encoding.encode(p)) for p in accepted) <= 900` using
  `tiktoken.get_encoding("o200k_base")`, stop before exceeding.
  If nothing scores above a minimal relevance floor, fall back to the
  highest-scoring passages anyway (never return an empty list on a
  recognized question — an empty answer scores zero for certain, a wrong
  guess only probably scores zero).
- Network fetch has its own short timeout (a few seconds) so a slow challenge
  host can't blow the 10s tool response budget; on fetch failure, returns
  whatever is already cached (possibly empty on a cold, failed start).

## Phase 2, Problem Set 2 — Map navigation (`services/mapnav.py`)

**Tool:** `next_hop(map_id, start, current, destination, hops_remaining?) -> str`

- `_graph(map_id)` (lock-guarded, memoized per `map_id`): GET
  `{API_BASE}/graph?map_id=...` once, returns `{adjacency, tolls}`. `API_BASE`
  defaults to `https://tool-box-2591eaa24fa3.herokuapp.com`, overridable via
  the `TOOLBOX_API_BASE` env var (the phase-2 doc doesn't give `/graph` a full
  host, only infers it from context — flagged as an assumption in the design
  discussion; env var makes it a one-line fix if wrong).
- Journey state: module-level dict keyed by `(map_id, start, destination)` →
  `set` of visited nodes, lock-guarded. Seeded with `{start}` on first sight
  of a journey key. After choosing a next node, it's added to the visited
  set. The entry is dropped once the chosen node equals `destination`
  (journey complete, key won't grow unboundedly across many runs).
- Cost model: entering node `v` from `u` costs `adjacency[u][v] + tolls[v]`
  (tolls charged on arrival; the start node's own toll is never charged since
  no edge is entered to reach it).
- **No `hops_remaining`:** plain Dijkstra from `current` to `destination` over
  the full graph with visited nodes removed (except `destination` itself
  stays reachable even if — degenerate case — it was already visited, which
  shouldn't happen in a well-formed journey). Return the first node on the
  resulting shortest path.
- **`hops_remaining` given:** bounded-hop DP — `best[k][v]` = min cost to
  reach `destination` from `v` using at most `k` remaining edges, computed
  bottom-up for `k = 0..hops_remaining` over the visited-excluded graph; pick
  the neighbor `v` of `current` minimizing
  `adjacency[current][v] + tolls[v] + best[hops_remaining-1][v]`.
- If no path exists at all from `current` (excluding visited) — shouldn't
  happen given the spec's guarantees, but defensively — falls back to any
  adjacent unvisited node so the tool never errors out; the journey likely
  scores 0 regardless per the spec's own rules, which is outside our control
  at that point.
- Every returned node is read out of `adjacency[current]`, so it is
  structurally impossible for this tool to violate "not adjacent" or "already
  visited" on its own.

## `toolbox.py` changes

- `_format_value`: add `if isinstance(value, list): return "\n\n".join(str(v) for v in value)` before the existing fallback, so `recall_study_material`'s
  list return renders as readable text in the `content` channel while
  `structuredContent.answer` keeps the real `List[str]`.
- `TOOLS`/`HANDLERS`: extended with `phase2.TOOLS`/`phase2.HANDLERS` (simple
  list-concat / dict-merge at module load).
- `INSTRUCTIONS`: gets a short added paragraph covering the two new tools
  (call `recall_study_material` for anything about the study/exam material;
  call `next_hop` repeatedly, feeding back the node it returns as the next
  `current`, until it returns the destination).

## Combo challenges (Phase 1 Problem Set 4, Phase 2 Problem Set 3)

No dedicated code — the spec says these are the agent orchestrating the tools
already built. Nothing to implement beyond making sure the underlying tools
and their descriptions are correct and clear enough for the model to chain
correctly.

## Testing

`unittest` (matches this repo's existing convention — `test_ghost_chains.py`
uses `unittest`, not `pytest`).

- `test_nursery.py`: `calculate` (symbol + natural-language expressions,
  precedence, negative operands), `identify_shape` (synthetic Pillow-drawn
  rectangle/triangle/circle fixtures), `count_sides`, `get_name` criteria.
- `test_mapnav.py`: Dijkstra + toll cost on a small fixed adjacency, hop-bounded
  DP against a case where the unconstrained shortest path needs more hops
  than allowed, and a revisit-avoidance case (graph where the naive shortest
  path from an intermediate node would loop back through an already-visited
  node absent the visited-set exclusion). No network — `_graph` is monkeypatched.
- `test_exam.py`: token-budget enforcement against a fixed mocked passage
  list (some passages individually near the 900 cap), never exceeds 900
  when summed via the real `tiktoken` encoding. No network — the materials
  cache is monkeypatched directly.

## Self-review

- **Placeholders:** none — every section above names the actual file/function
  it targets.
- **Consistency:** `_format_value` change is additive (only new branch for
  `list`), doesn't touch existing scalar formatting used by the five Phase 1
  tools. Journey-state and materials caches are separate dicts with separate
  locks — no shared mutable state between the two Phase 2 tools.
- **Scope:** bounded to the two subprojects discussed; no unrelated refactor
  of the existing `_dispatch`/`_handle_method` JSON-RPC machinery.
- **Ambiguity resolved explicitly:** `/graph` host assumption is called out
  and made overridable rather than silently hardcoded.
