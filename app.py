import json
import os
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from adaptive_gateway import transform as adaptive_gateway_transform
from ghost_chains import GhostChainsService
from kan_chiong_driver import solve as kan_chiong_solve
from showdown_bot import choose_action as showdown_choose_action
from stonks_man import solve_all as stonks_solve_all

NULL_RESULT = {"total_duration_sec": None, "arrival_time": None, "path": []}

EXECUTOR = None


def _solve_case(case_input):
    # isolated per-process worker fn: any bad case input degrades to the
    # spec's unreachable response instead of failing the whole batch
    try:
        return json.loads(kan_chiong_solve(json.dumps(case_input)))
    except Exception:
        return dict(NULL_RESULT)


def _handle_kan_chiong(body):
    if not isinstance(body, dict):
        raise ValueError("expected a JSON object of case_id -> input")

    results = {}
    if len(body) <= 1:
        for case_id, case_input in body.items():
            results[case_id] = _solve_case(case_input)
    else:
        futures = {
            case_id: EXECUTOR.submit(_solve_case, case_input)
            for case_id, case_input in body.items()
        }
        for case_id, future in futures.items():
            results[case_id] = future.result()
    return results


def _handle_adaptive_gateway(body):
    return adaptive_gateway_transform(body)


def _handle_showdown(body):
    return showdown_choose_action(body)


def _handle_stonks(body):
    return stonks_solve_all(body)


GHOST_CHAINS = GhostChainsService()


def _handle_ghost_chains_transactions(body):
    return GHOST_CHAINS.process_transactions(body)


def _handle_ghost_chains_reset(body):
    return GHOST_CHAINS.clear(body)


POST_ROUTES = {
    "/kan-cheong-delivery-driver": _handle_kan_chiong,
    "/solve": _handle_adaptive_gateway,
    "/showdown": _handle_showdown,
    "/ghost-chains/transactions": _handle_ghost_chains_transactions,
    "/ghost-chains/reset": _handle_ghost_chains_reset,
    "/stonks": _handle_stonks,
}

GET_ROUTES = {
    "/ghost-chains/health": lambda: GHOST_CHAINS.health(),
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        handler = GET_ROUTES.get(self.path)
        if handler is None:
            self._respond(404, {"error": "not found"})
            return

        try:
            result = handler()
        except Exception as exc:
            self._respond(400, {"error": str(exc)})
            return

        self._respond(200, result)

    def do_POST(self):
        handler = POST_ROUTES.get(self.path)
        if handler is None:
            self._respond(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid JSON body"})
            return

        try:
            result = handler(body)
        except Exception as exc:
            self._respond(400, {"error": str(exc)})
            return

        self._respond(200, result)

    def _respond(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def main():
    global EXECUTOR
    EXECUTOR = ProcessPoolExecutor(max_workers=os.cpu_count())
    port = int(os.environ.get("PORT", 8000))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Listening on http://0.0.0.0:{port}")
    for path in GET_ROUTES:
        print(f"  GET  {path}")
    for path in POST_ROUTES:
        print(f"  POST {path}")
    try:
        server.serve_forever()
    finally:
        EXECUTOR.shutdown(wait=False)


if __name__ == "__main__":
    main()
