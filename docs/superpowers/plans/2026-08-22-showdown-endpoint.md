# Showdown Endpoint Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight HTTP endpoint that reads the showdown turn payload and returns a legal action for every game state.

**Architecture:** Keep game logic pure and separate from HTTP plumbing. Put state parsing and action selection in a small module so it can be reused independently, then wire that module into the existing `BaseHTTPRequestHandler` route table in `app.py`. The endpoint should rely on the authoritative `legal_actions` data from the challenge payload and never invent illegal moves.

**Tech Stack:** Python standard library (`json`, `http.server`), existing server scaffold in `app.py`.

---

## File structure

- Create: `showdown_bot.py` — pure showdown turn parser and action-selection logic.
- Modify: `app.py` — add the showdown route and call the new bot module.

## Chunk 1: Build the showdown bot core

### Task 1: Implement basic action selection

**Files:**
- Create: `showdown_bot.py`

- [ ] **Step 1: Write the implementation**

Implement `choose_action(turn: dict) -> dict` so it:
1. Reads `legal_actions`.
2. Returns `check` if available.
3. Returns `call` if `check` is unavailable but `call` is legal.
4. Returns `raise` if it is legal and a raise amount helper can supply a valid amount.
5. Returns `bet` if `raise` is unavailable but `bet` is legal.
6. Returns `fold` only when `fold` is present in `legal_actions` and no other supported action is available.
7. Raises `ValueError` if `legal_actions` is missing, empty, or contains no supported action.

- [ ] **Step 2: Add a small amount helper**

Implement a helper in the same module that returns a valid amount for `bet` and `raise`, using:
1. `min_raise_to` for normal aggressive actions.
2. The exact allowed amount when `min_raise_to == max_raise_to`.
3. A `ValueError` when the required amount fields are missing or invalid.

- [ ] **Step 3: Keep the module self-contained**

Do not add HTTP code, networking, or server concerns to this file. It should stay pure and easy to reuse from the handler.

- [ ] **Step 4: Verify the module imports cleanly**

Run: `python -c "from showdown_bot import choose_action; print(choose_action({'"'"'legal_actions'"'"': ['"'"'check'"'"']}))"`
Expected: prints `{"action": "check"}` or equivalent dict output.

- [ ] **Step 5: Save the work**

```bash
git add showdown_bot.py
```

### Task 2: Wire the endpoint into the HTTP server

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the new route**

Update the route table in `app.py` so `/showdown` is handled by a dedicated function.

- [ ] **Step 2: Parse and validate the request body**

Make the handler:
1. Parse JSON.
2. Require a JSON object with a list-valued `legal_actions`.
3. Return HTTP 400 for malformed JSON or invalid payloads.

- [ ] **Step 3: Call the bot module**

Pass the parsed body to `showdown_bot.choose_action` and return its result as JSON with HTTP 200.

- [ ] **Step 4: Smoke test the endpoint locally**

Run: `python app.py`

Then POST a sample payload:

```bash
curl -X POST http://127.0.0.1:8000/showdown -H "Content-Type: application/json" -d "{\"legal_actions\":[\"check\",\"bet\"]}"
```

Expected: JSON response with a legal action.

Also verify invalid JSON returns HTTP 400:

```bash
curl -X POST http://127.0.0.1:8000/showdown -H "Content-Type: application/json" -d "not-json"
```

- [ ] **Step 5: Save the work**

```bash
git add app.py showdown_bot.py
```
