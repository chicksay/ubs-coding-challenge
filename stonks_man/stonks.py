"""Time Travelling Stonks Man -- buy low, sell high, get home to 2037."""

import time

HOME = 2037

# _beam_trips' candidate-path generation is O(years^2 * stocks) per beam
# state, repeated across 14 rounds and up to 18 beam states -- with enough
# distinct years in the timeline and a generous energy budget that blows up
# to tens of seconds per case, which a request-level batch of test cases
# turns into an upstream timeout (observed as a 502 from the Render proxy)
# well before any HTTP-server-side timeout would trip. A wall-clock deadline
# bounds worst-case latency regardless of input shape: the search just
# returns whatever best-so-far it has (always replay-validated below), which
# is strictly better than not responding at all.
REQUEST_BUDGET_SECONDS = 5.0
EXACT_MAX_YEARS = 9
EXACT_MAX_ENERGY = 30
EXACT_MAX_LOTS = 40
EXACT_SOLVER_SECONDS = 0.45


def solve_all(payload):
    if isinstance(payload, dict):
        deadline = time.monotonic() + REQUEST_BUDGET_SECONDS
        return [solve_one(payload, deadline)]
    if not isinstance(payload, list):
        raise ValueError("JSON array required")
    request_deadline = time.monotonic() + REQUEST_BUDGET_SECONDS
    out = []
    total = len(payload)
    for idx, case in enumerate(payload):
        remaining = request_deadline - time.monotonic()
        cases_left = total - idx
        # Keep early hard cases from starving later ones in the same request.
        case_budget = max(0.05, remaining / cases_left) if cases_left else 0.05
        deadline = min(request_deadline, time.monotonic() + case_budget)
        try:
            out.append(solve_one(case, deadline))
        except (TypeError, ValueError, KeyError, ZeroDivisionError):
            out.append([])
    return out


def solve_one(case, deadline=None):
    if not isinstance(case, dict):
        raise ValueError("case must be an object")
    energy = _int(case.get("energy"), 0)
    capital = _int(case.get("capital"), 0)
    timeline = _timeline(case.get("timeline"))
    if energy < 2 or capital <= 0 or not timeline:
        return []
    if deadline is None:
        deadline = time.monotonic() + REQUEST_BUDGET_SECONDS

    seed = {"energy": energy, "capital": capital, "timeline": timeline}
    best_actions = []
    best_profit = -1
    for acts in (
        _exact_milp_trips(seed, deadline),
        _targeted_chain_trades(seed, deadline),
        _two_step_chain_trades(seed, deadline),
        _simple_round_trips(seed, deadline),
        _beam_trips(seed, deadline),
        _oscillate_pairs(seed, deadline),
        _pair_trades(seed, deadline),
        _multi_step_trades(seed, deadline),
    ):
        profit = _replay(seed, acts)
        if profit is not None and profit > best_profit:
            best_profit = profit
            best_actions = acts
    return best_actions


def _int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _timeline(raw):
    years = {}
    if not isinstance(raw, dict):
        return years
    for year_key, stocks in raw.items():
        year = _int(year_key, None)
        if year is None or year <= 0 or year > HOME:
            continue
        if not isinstance(stocks, dict):
            continue
        bucket = {}
        for name, info in stocks.items():
            if not isinstance(info, dict):
                continue
            price = _int(info.get("price"), 0)
            qty = _int(info.get("qty"), 0)
            if price <= 0:
                continue
            bucket[str(name)] = {"price": price, "qty": max(0, qty)}
        if bucket:
            years[year] = bucket
    return years


def _price(timeline, year, stock):
    info = timeline.get(year, {}).get(stock)
    return None if info is None else info["price"]


def _qty_map(timeline):
    return {year: {name: info["qty"] for name, info in stocks.items()} for year, stocks in timeline.items()}


def _copy_qty(qty):
    return {year: dict(left) for year, left in qty.items()}


def _exact_milp_trips(seed, deadline=None):
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0.1:
            return []
        time_limit = min(EXACT_SOLVER_SECONDS, remaining)
    else:
        time_limit = EXACT_SOLVER_SECONDS

    try:
        from collections import defaultdict
        import pulp
    except ImportError:
        return []

    energy = seed["energy"]
    capital = seed["capital"]
    timeline = seed["timeline"]
    years = sorted(set(timeline) | {HOME})
    lots = []
    quote_price = {}
    tickers = set()
    for year, stocks in timeline.items():
        for ticker, info in stocks.items():
            price = info["price"]
            qty = info["qty"]
            quote_price[(year, ticker)] = price
            tickers.add(ticker)
            if qty > 0:
                lots.append((year, ticker, price, qty))

    if (
        len(years) > EXACT_MAX_YEARS
        or energy > EXACT_MAX_ENERGY
        or len(lots) > EXACT_MAX_LOTS
        or not lots
    ):
        return []

    try:
        nodes = [(year, spent) for spent in range(energy + 1) for year in years]
        root = (HOME, 0)
        arcs = []
        arcs_from = defaultdict(list)
        arcs_to = defaultdict(list)
        for node in nodes:
            year, spent = node
            for dest_year in years:
                cost = abs(dest_year - year)
                if dest_year == year or spent + cost > energy:
                    continue
                dest = (dest_year, spent + cost)
                arc = (node, dest)
                arcs.append(arc)
                arcs_from[node].append(arc)
                arcs_to[dest].append(arc)

        terminal_nodes = [(HOME, spent) for spent in range(energy + 1)]
        model = pulp.LpProblem("stonks", pulp.LpMaximize)
        route = {
            arc: pulp.LpVariable(f"route_{idx}", cat="Binary")
            for idx, arc in enumerate(arcs)
        }
        finish = {
            node: pulp.LpVariable(f"finish_{node[1]}", cat="Binary")
            for node in terminal_nodes
        }

        for node in nodes:
            incoming = pulp.lpSum(route[arc] for arc in arcs_to[node])
            outgoing = pulp.lpSum(route[arc] for arc in arcs_from[node])
            ends_here = finish[node] if node in finish else 0
            if node == root:
                model += outgoing + ends_here == 1
            else:
                model += incoming == outgoing + ends_here
        model += pulp.lpSum(finish.values()) == 1

        visit = {}
        for node in nodes:
            visit[node] = pulp.lpSum(route[arc] for arc in arcs_from[node]) + (
                finish[node] if node in finish else 0
            )

        buy = {}
        for lot_idx, (lot_year, _ticker, _price, qty) in enumerate(lots):
            for spent in range(energy + 1):
                node = (lot_year, spent)
                var = pulp.LpVariable(f"buy_{lot_idx}_{spent}", lowBound=0, upBound=qty, cat="Integer")
                buy[(lot_idx, node)] = var
                model += var <= qty * visit[node]
            model += pulp.lpSum(var for (idx, _node), var in buy.items() if idx == lot_idx) <= qty

        stock_upper = defaultdict(int)
        max_price = defaultdict(int)
        for year, ticker, price, qty in lots:
            stock_upper[ticker] += qty
            max_price[ticker] = max(max_price[ticker], price)
        for (year, ticker), price in quote_price.items():
            max_price[ticker] = max(max_price[ticker], price)

        sell = {}
        for quote_idx, ((year, ticker), _price) in enumerate(quote_price.items()):
            for spent in range(energy + 1):
                node = (year, spent)
                var = pulp.LpVariable(
                    f"sell_{quote_idx}_{spent}",
                    lowBound=0,
                    upBound=stock_upper[ticker],
                    cat="Integer",
                )
                sell[(ticker, node)] = var
                model += var <= stock_upper[ticker] * visit[node]

        stock_flow = {}
        for ticker in tickers:
            for arc_idx, arc in enumerate(arcs):
                var = pulp.LpVariable(
                    f"stock_{len(stock_flow)}_{arc_idx}",
                    lowBound=0,
                    upBound=stock_upper[ticker],
                    cat="Integer",
                )
                stock_flow[(ticker, arc)] = var
                model += var <= stock_upper[ticker] * route[arc]

        for ticker in tickers:
            for node in nodes:
                inventory_in = pulp.lpSum(stock_flow[(ticker, arc)] for arc in arcs_to[node])
                inventory_out = pulp.lpSum(stock_flow[(ticker, arc)] for arc in arcs_from[node])
                purchases = pulp.lpSum(
                    var
                    for (lot_idx, buy_node), var in buy.items()
                    if buy_node == node and lots[lot_idx][1] == ticker
                )
                sales = sell.get((ticker, node), 0)
                model += sales <= inventory_in
                model += inventory_in + purchases == inventory_out + sales

        cash_upper = capital + sum(qty * max_price[ticker] for _year, ticker, _price, qty in lots)
        cash_flow = {}
        for arc_idx, arc in enumerate(arcs):
            var = pulp.LpVariable(f"cash_{arc_idx}", lowBound=0, upBound=cash_upper, cat="Integer")
            cash_flow[("travel", arc)] = var
            model += var <= cash_upper * route[arc]
        for node in terminal_nodes:
            var = pulp.LpVariable(f"cash_finish_{node[1]}", lowBound=0, upBound=cash_upper, cat="Integer")
            cash_flow[("finish", node)] = var
            model += var <= cash_upper * finish[node]

        for node in nodes:
            cash_in = pulp.lpSum(cash_flow[("travel", arc)] for arc in arcs_to[node])
            cash_out = pulp.lpSum(cash_flow[("travel", arc)] for arc in arcs_from[node])
            if node in finish:
                cash_out += cash_flow[("finish", node)]
            sale_revenue = pulp.lpSum(
                quote_price[(node[0], ticker)] * var
                for (ticker, sell_node), var in sell.items()
                if sell_node == node
            )
            purchase_cost = pulp.lpSum(
                lots[lot_idx][2] * var
                for (lot_idx, buy_node), var in buy.items()
                if buy_node == node
            )
            initial = capital if node == root else 0
            model += cash_in + initial + sale_revenue == cash_out + purchase_cost

        final_cash = pulp.lpSum(cash_flow[("finish", node)] for node in terminal_nodes)
        action_penalty = pulp.lpSum(route.values()) + pulp.lpSum(buy.values()) + pulp.lpSum(sell.values())
        max_penalty = energy + 2 * sum(qty for _year, _ticker, _price, qty in lots)
        model += final_cash * (max_penalty + 1) - action_penalty

        status = model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
        if pulp.LpStatus[status] != "Optimal":
            return []

        def ival(var):
            value = pulp.value(var)
            return int(round(0 if value is None else value))

        selected_next = {arc[0]: arc[1] for arc, var in route.items() if ival(var) == 1}
        selected_finish = {node for node, var in finish.items() if ival(var) == 1}
        actions = []
        node = root
        seen = set()
        while node not in selected_finish:
            if node in seen or node not in selected_next:
                return []
            seen.add(node)
            for ticker in sorted(tickers):
                qty = ival(sell[(ticker, node)]) if (ticker, node) in sell else 0
                if qty > 0:
                    actions.append(f"s-{ticker}-{qty}")
            buys_here = defaultdict(int)
            for (lot_idx, buy_node), var in buy.items():
                if buy_node == node:
                    qty = ival(var)
                    if qty > 0:
                        buys_here[lots[lot_idx][1]] += qty
            for ticker in sorted(buys_here):
                actions.append(f"b-{ticker}-{buys_here[ticker]}")
            dest = selected_next[node]
            actions.append(f"j-{node[0]}-{dest[0]}")
            node = dest

        for ticker in sorted(tickers):
            qty = ival(sell[(ticker, node)]) if (ticker, node) in sell else 0
            if qty > 0:
                actions.append(f"s-{ticker}-{qty}")
        return actions if _replay(seed, actions) is not None else []
    except Exception:
        return []


def _beam_trips(seed, deadline=None):
    timeline = seed["timeline"]
    years = sorted(y for y in timeline if y <= HOME)
    if HOME not in years:
        years.append(HOME)
        years.sort()

    beam = [(seed["energy"], seed["capital"], _qty_map(timeline), {}, [])]
    best_actions = []
    best_profit = -1

    for _ in range(14):
        nxt = []
        seen = set()
        out_of_time = False
        for energy, capital, qty, holdings, prefix in beam:
            if out_of_time:
                break
            if deadline is not None and time.monotonic() >= deadline:
                out_of_time = True
                break
            farthest = HOME - energy // 2
            for path_idx, path in enumerate(_candidate_paths(years, farthest, energy)):
                # A single state's candidate list can itself run into the
                # hundreds/thousands for a wide timeline, each costing an
                # O(len(path)^2 * stocks) _walk call, so poll for the
                # deadline here too (not just once per state) -- but not on
                # every iteration, since time.monotonic() isn't free either.
                if path_idx % 32 == 0 and deadline is not None and time.monotonic() >= deadline:
                    out_of_time = True
                    break
                acts, new_cap, new_qty, new_hold = _walk(
                    path, capital, qty, holdings, timeline
                )
                cost = _jump_cost(acts)
                if not acts or new_cap <= capital or cost > energy:
                    continue
                trial = prefix + acts
                # Score every fully-built candidate the instant it exists,
                # rather than waiting for it to be "promoted" to best_actions
                # when a later round happens to process it as a beam state --
                # a deadline (or the round/beam-width caps below) can stop
                # that promotion from ever happening, which would silently
                # discard an already-valid, already-profitable trial.
                trial_profit = _replay(seed, trial)
                if trial_profit is None:
                    continue
                if trial_profit > best_profit:
                    best_profit = trial_profit
                    best_actions = trial
                key = (energy - cost, new_cap, _qty_key(new_qty), tuple(sorted(new_hold.items())))
                if key in seen:
                    continue
                seen.add(key)
                nxt.append((energy - cost, new_cap, new_qty, new_hold, trial))
        if out_of_time or not nxt:
            break
        nxt.sort(key=lambda st: (st[1], st[0]), reverse=True)
        beam = nxt[:18]
    return _finish(best_actions, {}, timeline)


def _qty_key(qty):
    return tuple(sorted((year, name, left) for year, stocks in qty.items() for name, left in stocks.items() if left))


def _candidate_paths(years, farthest, energy):
    paths = []
    seen = set()
    for ymin in years:
        if ymin < farthest or ymin >= HOME:
            continue
        span = [y for y in years if ymin <= y <= HOME]
        simple = _out_and_back(years, ymin)
        for path in (simple,):
            key = tuple(path)
            if key not in seen and _path_travel(path) <= energy:
                seen.add(key)
                paths.append(path)
        for y_hi in span:
            if y_hi <= ymin:
                continue
            for extra in (1, 2):
                cost = 2 * (HOME - ymin) + 2 * extra * (y_hi - ymin)
                if cost > energy:
                    continue
                path = _out_zag_back(years, ymin, y_hi, extra)
                key = tuple(path)
                if key not in seen:
                    seen.add(key)
                    paths.append(path)
    
    # CRITICAL: Chained intermediate paths - visit multiple years sequentially
    # without returning to HOME between visits. Much cheaper than multi-arm patterns!
    # e.g., HOME -> y_earliest -> ... -> y_middle -> ... -> y_latest -> HOME
    sorted_years = sorted([y for y in years if y <= HOME])
    if len(sorted_years) >= 2:
        # Two-year chains: visit earliest and latest in sequence
        for i in range(len(sorted_years)):
            for j in range(i + 1, len(sorted_years)):
                y1, y2 = sorted_years[i], sorted_years[j]
                # Forward: HOME -> y1 -> y2 -> HOME
                cost = (HOME - y1) + (y2 - y1) + (HOME - y2)
                if cost <= energy:
                    path = [HOME, y1, y2, HOME]
                    key = tuple(path)
                    if key not in seen:
                        seen.add(key)
                        paths.append(path)
                
                # Backward: HOME -> y2 -> y1 -> HOME
                cost = (HOME - y2) + (y2 - y1) + (HOME - y1)
                if cost <= energy:
                    path = [HOME, y2, y1, HOME]
                    key = tuple(path)
                    if key not in seen:
                        seen.add(key)
                        paths.append(path)
        
        # Three-year chains (visit in sequence without returning home)
        if len(sorted_years) >= 3:
            for i in range(len(sorted_years)):
                for j in range(i + 1, len(sorted_years)):
                    for k in range(j + 1, len(sorted_years)):
                        y1, y2, y3 = sorted_years[i], sorted_years[j], sorted_years[k]
                        # Pattern: HOME -> y1 -> y2 -> y3 -> HOME
                        cost = (HOME - y1) + (y2 - y1) + (y3 - y2) + (HOME - y3)
                        if cost <= energy:
                            path = [HOME, y1, y2, y3, HOME]
                            key = tuple(path)
                            if key not in seen:
                                seen.add(key)
                                paths.append(path)
                        
                        # Reverse order: HOME -> y3 -> y2 -> y1 -> HOME
                        cost = (HOME - y3) + (y3 - y2) + (y2 - y1) + (HOME - y1)
                        if cost <= energy:
                            path = [HOME, y3, y2, y1, HOME]
                            key = tuple(path)
                            if key not in seen:
                                seen.add(key)
                                paths.append(path)
        
        # Four-year chains for sufficient energy
        if len(sorted_years) >= 4:
            for i in range(len(sorted_years) - 3):
                y1 = sorted_years[i]
                y2 = sorted_years[i + 1]
                y3 = sorted_years[i + 2]
                y4 = sorted_years[i + 3]
                # Forward chain
                cost = (HOME - y1) + (y2 - y1) + (y3 - y2) + (y4 - y3) + (HOME - y4)
                if cost <= energy:
                    path = [HOME, y1, y2, y3, y4, HOME]
                    key = tuple(path)
                    if key not in seen:
                        seen.add(key)
                        paths.append(path)
                
                # Reverse chain
                cost = (HOME - y4) + (y4 - y3) + (y3 - y2) + (y2 - y1) + (HOME - y1)
                if cost <= energy:
                    path = [HOME, y4, y3, y2, y1, HOME]
                    key = tuple(path)
                    if key not in seen:
                        seen.add(key)
                        paths.append(path)

    # General monotonic routes: HOME -> pivot -> HOME, where pivot is any
    # subset of years traversed in ascending/descending order. This captures
    # long chained arbitrage opportunities missed by fixed-size patterns.
    interior = [y for y in sorted_years if y != HOME]
    n = len(interior)
    if 1 <= n <= 10:
        for mask in range(1, 1 << n):
            chain = [interior[idx] for idx in range(n) if (mask >> idx) & 1]
            asc = [HOME] + chain + [HOME]
            key = tuple(asc)
            if key not in seen and _path_travel(asc) <= energy:
                seen.add(key)
                paths.append(asc)
            desc = [HOME] + list(reversed(chain)) + [HOME]
            key = tuple(desc)
            if key not in seen and _path_travel(desc) <= energy:
                seen.add(key)
                paths.append(desc)
    
    return paths


def _out_and_back(years, ymin):
    span = [y for y in years if ymin <= y <= HOME]
    if HOME not in span:
        span = sorted(span + [HOME])
    if ymin not in span:
        span = sorted(span + [ymin])
    outbound = list(reversed(span))
    return outbound + span[1:]


def _out_zag_back(years, ymin, y_hi, extra):
    span = [y for y in years if ymin <= y <= HOME]
    if HOME not in span:
        span = sorted(span + [HOME])
    to_hi = [y for y in span if ymin <= y <= y_hi]
    to_lo = list(reversed(to_hi))
    path = list(reversed(span))
    for _ in range(extra):
        path.extend(to_hi[1:])
        path.extend(to_lo[1:])
    path.extend(span[1:])
    return path


def _path_travel(path):
    return sum(abs(path[i] - path[i - 1]) for i in range(1, len(path)))


def _walk(path, capital, qty, holdings, timeline):
    qty = _copy_qty(qty)
    holdings = dict(holdings)
    capital = int(capital)
    ops = [[] for _ in path]

    for i, year in enumerate(path):
        here = timeline.get(year, {})
        for stock, have in list(holdings.items()):
            if have <= 0 or stock not in here:
                continue
            now = here[stock]["price"]
            future_max = _suffix_max(timeline, path, i + 1, stock)
            if now >= future_max:
                ops[i].append(("s", stock, have))
                capital += have * now
                holdings[stock] = 0

        candidates = []
        for stock, info in here.items():
            avail = qty.get(year, {}).get(stock, 0)
            buy_price = info["price"]
            if avail <= 0 or buy_price <= 0:
                continue
            best_price, peak = _best_future(timeline, path, i + 1, stock)
            if peak is None or best_price <= buy_price:
                continue
            soak = _cheaper_soak(path, i, peak, stock, buy_price, qty, timeline)
            spend = max(0, capital - soak)
            take = min(avail, spend // buy_price)
            if take <= 0:
                continue
            candidates.append((stock, buy_price, best_price - buy_price, take))
        for stock, take in _allocate_buys(candidates, capital).items():
            buy_price = here[stock]["price"]
            ops[i].append(("b", stock, take))
            capital -= take * buy_price
            holdings[stock] = holdings.get(stock, 0) + take
            qty[year][stock] -= take

    actions = []
    current = path[0]
    for i, year in enumerate(path):
        if not ops[i] and year != path[-1]:
            continue
        if year != current:
            actions.append(f"j-{current}-{year}")
            current = year
        for kind, stock, take in ops[i]:
            actions.append(f"{kind}-{stock}-{take}")
    if current != path[-1]:
        actions.append(f"j-{current}-{path[-1]}")
    return actions, capital, qty, holdings


def _walk_two_step_budget(path, capital, qty, holdings, timeline):
    qty = _copy_qty(qty)
    holdings = dict(holdings)
    capital = int(capital)
    ops = [[] for _ in path]

    for i, year in enumerate(path):
        here = timeline.get(year, {})
        for stock, have in list(holdings.items()):
            if have <= 0 or stock not in here:
                continue
            now = here[stock]["price"]
            future_max = _suffix_max(timeline, path, i + 1, stock)
            if now >= future_max:
                ops[i].append(("s", stock, have))
                capital += have * now
                holdings[stock] = 0

        current_candidates = []
        for stock, info in here.items():
            avail = qty.get(year, {}).get(stock, 0)
            buy_price = info["price"]
            if avail <= 0 or buy_price <= 0:
                continue
            best_price, peak = _best_future(timeline, path, i + 1, stock)
            if peak is None or best_price <= buy_price:
                continue
            soak = _cheaper_soak(path, i, peak, stock, buy_price, qty, timeline)
            spend = max(0, capital - soak)
            take = min(avail, spend // buy_price)
            if take <= 0:
                continue
            current_candidates.append((stock, buy_price, best_price - buy_price, take))

        next_candidates = []
        if i + 1 < len(path):
            next_year = path[i + 1]
            nxt = timeline.get(next_year, {})
            for stock, info in nxt.items():
                avail = qty.get(next_year, {}).get(stock, 0)
                buy_price = info["price"]
                if avail <= 0 or buy_price <= 0:
                    continue
                best_price, peak = _best_future(timeline, path, i + 2, stock)
                if peak is None or best_price <= buy_price:
                    continue
                take = min(avail, capital // buy_price)
                if take <= 0:
                    continue
                next_candidates.append((stock, buy_price, best_price - buy_price, take))

        alloc_now = _allocate_buys_with_next(current_candidates, next_candidates, capital)
        for stock, take in alloc_now.items():
            buy_price = here[stock]["price"]
            ops[i].append(("b", stock, take))
            capital -= take * buy_price
            holdings[stock] = holdings.get(stock, 0) + take
            qty[year][stock] -= take

    actions = []
    current = path[0]
    for i, year in enumerate(path):
        if not ops[i] and year != path[-1]:
            continue
        if year != current:
            actions.append(f"j-{current}-{year}")
            current = year
        for kind, stock, take in ops[i]:
            actions.append(f"{kind}-{stock}-{take}")
    if current != path[-1]:
        actions.append(f"j-{current}-{path[-1]}")
    return actions, capital, qty, holdings


def _suffix_max(timeline, path, start, stock):
    best = -1
    for j in range(start, len(path)):
        later = _price(timeline, path[j], stock)
        if later is not None and later > best:
            best = later
    return best


def _best_future(timeline, path, start, stock):
    best_price = -1
    peak = None
    for j in range(start, len(path)):
        later = _price(timeline, path[j], stock)
        if later is not None and later > best_price:
            best_price = later
            peak = j
    return best_price, peak


def _cheaper_soak(path, i, peak, stock, buy_price, qty, timeline):
    soak = 0
    for k in range(i + 1, peak):
        later = _price(timeline, path[k], stock)
        if later is None or later >= buy_price:
            continue
        avail = qty.get(path[k], {}).get(stock, 0)
        if avail > 0:
            soak += avail * later
    return soak


_KNAPSACK_CAP_LIMIT = 2000
_KNAPSACK_MAX_CANDIDATES = 8


def _allocate_buys(candidates, capital):
    """Choose how many units of each candidate stock to buy at one visit.

    candidates: (stock, buy_price, profit_per_unit, max_take) tuples, all
    already independently profitable. Multiple candidates compete for the
    same capital, so picking the single best-ROI-ratio stock first and
    filling it to its max (the previous approach) is the fractional-knapsack
    rule -- it is only exact when capital doesn't bind. Here each stock's
    units are a bounded 0/1 choice at a shared price, i.e. a genuine bounded
    knapsack, so ROI-ratio-first can strictly underperform (verified: it
    maxes out the higher-ratio stock and leaves the higher-absolute-profit
    one underfunded, when splitting capital across both beats maxing either
    one alone).

    Solved exactly via a DP over capital when the decision space is cheap
    enough to be worth it (this runs inside the beam search's hot _walk
    loop, so it must stay cheap on the overwhelmingly common case of 0 or 1
    candidates, where there's no allocation decision to make anyway).
    Falls back to the ROI-ratio greedy otherwise -- which remains exact in
    the capital-abundant limit, and is only ever a heuristic when capital
    is genuinely the binding constraint across several competing stocks.
    """
    if not candidates:
        return {}
    if len(candidates) == 1:
        stock, price, _profit, take = candidates[0]
        take = min(take, capital // price)
        return {stock: take} if take > 0 else {}

    total_spend = sum(price * take for _, price, _, take in candidates)
    cap = min(capital, total_spend)
    if cap <= 0:
        return {}
    if cap > _KNAPSACK_CAP_LIMIT or len(candidates) > _KNAPSACK_MAX_CANDIDATES:
        return _greedy_allocate_buys(candidates, capital)

    # Binary-split each stock's bounded 0..max_take choice into O(log
    # max_take) 0/1 lots (clamping max_take to what cap could ever afford,
    # so a huge qty on a cheap stock doesn't blow up the split count) so a
    # standard 0/1 knapsack DP can solve this exactly.
    items = []  # (cost, value, stock, units)
    for stock, price, profit, max_take in candidates:
        remaining = min(max_take, cap // price)
        chunk = 1
        while remaining > 0:
            take = min(chunk, remaining)
            items.append((price * take, profit * take, stock, take))
            remaining -= take
            chunk *= 2

    n = len(items)
    dp = [[0] * (cap + 1) for _ in range(n + 1)]
    for idx in range(1, n + 1):
        cost, value, _, _ = items[idx - 1]
        row, prev = dp[idx], dp[idx - 1]
        for c in range(cap + 1):
            row[c] = prev[c]
            if cost <= c and prev[c - cost] + value > row[c]:
                row[c] = prev[c - cost] + value

    allocation = {}
    c = cap
    for idx in range(n, 0, -1):
        if dp[idx][c] != dp[idx - 1][c]:
            cost, _, stock, take = items[idx - 1]
            allocation[stock] = allocation.get(stock, 0) + take
            c -= cost
    return allocation


def _allocate_buys_with_next(current_candidates, next_candidates, capital):
    if not current_candidates:
        return {}
    if not next_candidates:
        return _allocate_buys(current_candidates, capital)

    now_items = []
    for stock, price, profit, max_take in current_candidates:
        now_items.append((stock, price, profit, max_take, "now"))
    nxt_items = []
    for stock, price, profit, max_take in next_candidates:
        nxt_items.append((stock, price, profit, max_take, "next"))
    all_items = now_items + nxt_items

    total_spend = sum(price * take for _stock, price, _profit, take, _tag in all_items)
    cap = min(capital, total_spend)
    if cap <= 0:
        return {}
    if cap > _KNAPSACK_CAP_LIMIT or len(all_items) > (_KNAPSACK_MAX_CANDIDATES * 2):
        return _allocate_buys(current_candidates, capital)

    items = []  # (cost, value, stock, units, tag)
    for stock, price, profit, max_take, tag in all_items:
        remaining = min(max_take, cap // price)
        chunk = 1
        while remaining > 0:
            take = min(chunk, remaining)
            items.append((price * take, profit * take, stock, take, tag))
            remaining -= take
            chunk *= 2

    n = len(items)
    dp = [[0] * (cap + 1) for _ in range(n + 1)]
    for idx in range(1, n + 1):
        cost, value, _stock, _units, _tag = items[idx - 1]
        row, prev = dp[idx], dp[idx - 1]
        for c in range(cap + 1):
            row[c] = prev[c]
            if cost <= c and prev[c - cost] + value > row[c]:
                row[c] = prev[c - cost] + value

    allocation = {}
    c = cap
    for idx in range(n, 0, -1):
        if dp[idx][c] != dp[idx - 1][c]:
            cost, _value, stock, take, tag = items[idx - 1]
            if tag == "now":
                allocation[stock] = allocation.get(stock, 0) + take
            c -= cost
    return allocation


def _greedy_allocate_buys(candidates, capital):
    allocation = {}
    remaining = capital
    for stock, price, profit, max_take in sorted(
        candidates, key=lambda c: (c[1] + c[2]) / c[1], reverse=True
    ):
        take = min(max_take, remaining // price)
        if take <= 0:
            continue
        allocation[stock] = take
        remaining -= take * price
    return allocation


def _simple_round_trips(seed, deadline=None):
    """Simple exhaustive strategy: try all round-trip buy/sell pairs."""
    energy = seed["energy"]
    capital = seed["capital"]
    timeline = seed["timeline"]
    best_actions = []
    best_profit = -1
    years = sorted(timeline.keys())
    
    for buy_year in years:
        if deadline is not None and time.monotonic() >= deadline:
            break
        for sell_year in years:
            if buy_year == sell_year or deadline is not None and time.monotonic() >= deadline:
                continue
            
            cost_to_buy = abs(HOME - buy_year)
            cost_to_sell = abs(buy_year - sell_year)
            cost_home = abs(sell_year - HOME)
            total_cost = cost_to_buy + cost_to_sell + cost_home
            
            if total_cost > energy:
                continue
            
            qty = _qty_map(timeline)
            holdings = {}
            year = HOME
            actions = []
            current_capital = capital
            
            # Jump to buy year
            if year != buy_year:
                actions.append(f"j-{year}-{buy_year}")
                year = buy_year
            
            # Buy stocks with best ROI
            candidates = []
            for name, info in timeline[buy_year].items():
                sell_price = timeline.get(sell_year, {}).get(name, {}).get("price")
                if sell_price is None or sell_price <= info["price"] or qty[buy_year][name] <= 0:
                    continue
                roi = sell_price / info["price"]
                profit_per_unit = sell_price - info["price"]
                candidates.append((roi, profit_per_unit, name, info["price"]))
            
            if not candidates:
                continue
            
            candidates.sort(reverse=True)
            bought_any = False
            for roi, profit_per_unit, name, price in candidates:
                take = min(qty[buy_year][name], current_capital // price)
                if take <= 0:
                    continue
                actions.append(f"b-{name}-{take}")
                current_capital -= take * price
                holdings[name] = take
                qty[buy_year][name] -= take
                bought_any = True
            
            if not bought_any:
                continue
            
            # Jump to sell year
            if year != sell_year:
                actions.append(f"j-{year}-{sell_year}")
                year = sell_year
            
            # Sell all holdings
            for name, have in holdings.items():
                price = timeline.get(sell_year, {}).get(name, {}).get("price")
                if price is not None and have > 0:
                    actions.append(f"s-{name}-{have}")
                    current_capital += have * price
            
            # Return home
            if year != HOME:
                actions.append(f"j-{year}-{HOME}")
            
            # Validate and score
            profit = _replay(seed, actions)
            if profit is not None and profit > best_profit:
                best_profit = profit
                best_actions = actions
    
    return best_actions


def _multi_step_trades(seed, deadline=None):
    """Execute sequential trades along chained paths: buy-sell-buy-sell pattern."""
    energy = seed["energy"]
    capital = seed["capital"]
    timeline = seed["timeline"]
    best_actions = []
    best_profit = -1
    years = sorted(timeline.keys())
    
    if len(years) < 2:
        return best_actions
    
    # Try all pairs of (buy1, sell1, buy2, sell2) combinations
    for i, buy1 in enumerate(years):
        if deadline is not None and time.monotonic() >= deadline:
            break
        for sell1 in years:
            if sell1 == buy1:
                continue
            for buy2 in years:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if buy2 == sell1:
                    continue
                for sell2 in years:
                    if sell2 == buy2 or sell2 == buy1:
                        continue
                    
                    # Check energy for path: HOME -> buy1 -> sell1 -> buy2 -> sell2 -> HOME
                    path = [HOME, buy1, sell1, buy2, sell2]
                    path_cost = sum(abs(path[i] - path[i-1]) for i in range(1, len(path)))
                    cost_home = abs(path[-1] - HOME)
                    total_cost = path_cost + cost_home
                    
                    if total_cost > energy:
                        continue
                    
                    qty = _qty_map(timeline)
                    holdings = {}
                    current_capital = capital
                    actions = []
                    year = HOME
                    
                    # Execute the trades along the path
                    for next_year in [buy1, sell1, buy2, sell2, HOME]:
                        if year != next_year:
                            actions.append(f"j-{year}-{next_year}")
                            year = next_year
                        
                        # Try to buy at buy1 and buy2
                        if year == buy1 or year == buy2:
                            candidates = []
                            for name, info in timeline.get(year, {}).items():
                                # Look for next profitable sell opportunity
                                if year == buy1:
                                    future_price = timeline.get(sell1, {}).get(name, {}).get("price")
                                else:  # year == buy2
                                    future_price = timeline.get(sell2, {}).get(name, {}).get("price")
                                
                                if future_price is None or future_price <= info["price"] or qty[year][name] <= 0:
                                    continue
                                roi = future_price / info["price"]
                                profit_per = future_price - info["price"]
                                candidates.append((roi, profit_per, name, info["price"]))
                            
                            if candidates:
                                candidates.sort(reverse=True)
                                for roi, profit_per, name, price in candidates:
                                    take = min(qty[year][name], current_capital // price)
                                    if take <= 0:
                                        continue
                                    actions.append(f"b-{name}-{take}")
                                    current_capital -= take * price
                                    holdings[name] = holdings.get(name, 0) + take
                                    qty[year][name] -= take
                        
                        # Try to sell at sell1 and sell2
                        elif year == sell1 or year == sell2:
                            for name in list(holdings):
                                if holdings[name] > 0 and name in timeline.get(year, {}):
                                    price = timeline[year][name]["price"]
                                    have = holdings[name]
                                    actions.append(f"s-{name}-{have}")
                                    current_capital += have * price
                                    holdings[name] = 0
                    
                    # Validate
                    profit = _replay(seed, actions)
                    if profit is not None and profit > best_profit:
                        best_profit = profit
                        best_actions = actions
    
    # Also try a pure monotonic sweep strategy that can realize profits from
    # many small edges across multiple years.
    for ordered in (years, list(reversed(years))):
        if deadline is not None and time.monotonic() >= deadline:
            break
        path = [HOME] + [y for y in ordered if y != HOME] + [HOME]
        if _path_travel(path) > energy:
            continue
        acts, *_rest = _walk(path, capital, _qty_map(timeline), {}, timeline)
        profit = _replay(seed, acts)
        if profit is not None and profit > best_profit:
            best_profit = profit
            best_actions = acts

    return best_actions


_TARGETED_YEAR_LIMIT = 12
_TARGETED_PATH_EVAL_LIMIT = 1200


def _targeted_chain_trades(seed, deadline=None):
    energy = seed["energy"]
    capital = seed["capital"]
    timeline = seed["timeline"]
    years = sorted(y for y in timeline if y <= HOME and y != HOME)
    if len(years) <= 10:
        return []

    scores = {year: 0 for year in years}
    for buy_idx, (buy_year, stocks) in enumerate(timeline.items()):
        if buy_idx % 8 == 0 and deadline is not None and time.monotonic() >= deadline:
            return []
        if buy_year > HOME:
            continue
        for stock, info in stocks.items():
            buy_price = info["price"]
            avail = info["qty"]
            if avail <= 0 or buy_price <= 0:
                continue
            affordable = min(avail, max(1, capital // buy_price))
            for sell_year, sell_stocks in timeline.items():
                sell_price = sell_stocks.get(stock, {}).get("price")
                if sell_price is None or sell_price <= buy_price:
                    continue
                if abs(HOME - buy_year) + abs(buy_year - sell_year) + abs(sell_year - HOME) > energy:
                    continue
                edge = (sell_price - buy_price) * affordable
                if buy_year in scores:
                    scores[buy_year] += edge
                if sell_year in scores:
                    scores[sell_year] += edge // 2

    ranked_years = [year for year, score in sorted(scores.items(), key=lambda item: item[1], reverse=True) if score > 0]
    if not ranked_years:
        return []
    chosen = sorted(ranked_years[:_TARGETED_YEAR_LIMIT])

    candidate_paths = []
    seen = set()

    def add_path(path):
        key = tuple(path)
        if key in seen or _path_travel(path) > energy:
            return
        seen.add(key)
        score = sum(scores.get(year, 0) for year in set(path))
        candidate_paths.append((score, len(path), path))

    add_path([HOME] + years + [HOME])
    add_path([HOME] + list(reversed(years)) + [HOME])

    n = len(chosen)
    for mask in range(1, 1 << n):
        if mask % 128 == 0 and deadline is not None and time.monotonic() >= deadline:
            break
        chain = [chosen[idx] for idx in range(n) if (mask >> idx) & 1]
        add_path([HOME] + chain + [HOME])
        add_path([HOME] + list(reversed(chain)) + [HOME])

    # Keep local windows too: a medium-scoring bridge year can unlock a sell and
    # buy sequence that pure top-year subsets miss.
    for width in range(3, min(8, len(years)) + 1):
        for start in range(0, len(years) - width + 1):
            window = years[start:start + width]
            add_path([HOME] + window + [HOME])
            add_path([HOME] + list(reversed(window)) + [HOME])

    candidate_paths.sort(reverse=True)
    best_actions = []
    best_profit = -1
    for idx, (_score, _length, path) in enumerate(candidate_paths):
        if idx >= _TARGETED_PATH_EVAL_LIMIT:
            break
        if idx % 16 == 0 and deadline is not None and time.monotonic() >= deadline:
            break
        for walker in (_walk, _walk_two_step_budget):
            acts, _new_cap, _new_qty, _new_hold = walker(path, capital, _qty_map(timeline), {}, timeline)
            profit = _replay(seed, acts)
            if profit is not None and profit > best_profit:
                best_profit = profit
                best_actions = acts
    return best_actions


def _two_step_chain_trades(seed, deadline=None):
    energy = seed["energy"]
    capital = seed["capital"]
    timeline = seed["timeline"]
    best_actions = []
    best_profit = -1

    years = sorted(y for y in timeline if y <= HOME and y != HOME)
    if not years:
        return best_actions

    candidate_paths = []
    asc = [HOME] + years + [HOME]
    desc = [HOME] + list(reversed(years)) + [HOME]
    candidate_paths.extend([asc, desc])
    for i in range(len(years)):
        for j in range(i + 1, len(years)):
            y1, y2 = years[i], years[j]
            candidate_paths.append([HOME, y1, y2, HOME])
            candidate_paths.append([HOME, y2, y1, HOME])

    seen = set()
    for path in candidate_paths:
        if deadline is not None and time.monotonic() >= deadline:
            break
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        if _path_travel(path) > energy:
            continue
        acts, _new_cap, _new_qty, _new_hold = _walk_two_step_budget(
            path, capital, _qty_map(timeline), {}, timeline
        )
        profit = _replay(seed, acts)
        if profit is not None and profit > best_profit:
            best_profit = profit
            best_actions = acts
    return best_actions


def _oscillate_pairs(seed, deadline=None):
    timeline = seed["timeline"]
    best_actions = []
    best_profit = -1
    years = list(timeline)
    for buy_year in years:
        if deadline is not None and time.monotonic() >= deadline:
            break
        for sell_year in years:
            if buy_year == sell_year:
                continue
            acts = _run_oscillation(seed, buy_year, sell_year)
            profit = _replay(seed, acts)
            if profit is not None and profit > best_profit:
                best_profit = profit
                best_actions = acts
    return best_actions


def _run_oscillation(seed, buy_year, sell_year):
    energy = seed["energy"]
    capital = seed["capital"]
    timeline = seed["timeline"]
    qty = _qty_map(timeline)
    holdings = {}
    year = HOME
    actions = []

    def jump(dest):
        nonlocal energy, year
        if dest == year:
            return
        energy -= abs(dest - year)
        actions.append(f"j-{year}-{dest}")
        year = dest

    def buy_lot(name):
        nonlocal capital
        price = _price(timeline, year, name)
        take = min(qty[year][name], capital // price)
        if take <= 0:
            return 0
        qty[year][name] -= take
        capital -= take * price
        holdings[name] = holdings.get(name, 0) + take
        actions.append(f"b-{name}-{take}")
        return take

    def sell_lot(name):
        nonlocal capital
        price = _price(timeline, year, name)
        have = holdings.get(name, 0)
        if price is None or have <= 0:
            return 0
        holdings[name] = 0
        capital += have * price
        actions.append(f"s-{name}-{have}")
        return have

    def profitable_names():
        ranked = []
        for name, info in timeline.get(buy_year, {}).items():
            sell_price = _price(timeline, sell_year, name)
            if sell_price is None or sell_price <= info["price"]:
                continue
            if qty[buy_year][name] <= 0:
                continue
            ranked.append((sell_price / info["price"], sell_price - info["price"], name))
        ranked.sort(reverse=True)
        return [name for _roi, _edge, name in ranked]

    while True:
        names = profitable_names()
        if not names:
            break
        cheapest = min(timeline[buy_year][name]["price"] for name in names)
        if capital < cheapest:
            break
        to_buy = abs(year - buy_year)
        to_sell = abs(buy_year - sell_year)
        to_home = abs(sell_year - HOME)
        if to_buy + to_sell + to_home > energy:
            break
        jump(buy_year)
        bought = 0
        for name in names:
            bought += buy_lot(name)
        if bought == 0:
            break
        jump(sell_year)
        for name in list(holdings):
            sell_lot(name)

    if year != HOME:
        jump(HOME)
    for name in list(holdings):
        if _price(timeline, HOME, name) is not None:
            sell_lot(name)
    return actions


def _pair_trades(seed, deadline=None):
    energy = seed["energy"]
    capital = seed["capital"]
    timeline = seed["timeline"]
    qty = _qty_map(timeline)
    holdings = {}
    year = HOME
    actions = []

    def can_reach(stops):
        here = year
        cost = 0
        for dest in stops:
            cost += abs(dest - here)
            here = dest
        return cost + abs(HOME - here) <= energy

    def jump(dest):
        nonlocal energy, year
        if dest == year:
            return
        energy -= abs(dest - year)
        actions.append(f"j-{year}-{dest}")
        year = dest

    def buy(stock, take):
        nonlocal capital
        price = _price(timeline, year, stock)
        take = min(take, qty[year][stock], capital // price)
        if take <= 0:
            return 0
        qty[year][stock] -= take
        capital -= take * price
        holdings[stock] = holdings.get(stock, 0) + take
        actions.append(f"b-{stock}-{take}")
        return take

    def sell(stock, take=None):
        nonlocal capital
        price = _price(timeline, year, stock)
        if price is None:
            return 0
        have = holdings.get(stock, 0)
        take = have if take is None else min(take, have)
        if take <= 0:
            return 0
        holdings[stock] = have - take
        capital += take * price
        actions.append(f"s-{stock}-{take}")
        return take

    while True:
        if deadline is not None and time.monotonic() >= deadline:
            break
        best = None
        for buy_year, stocks in timeline.items():
            for stock, info in stocks.items():
                avail = qty[buy_year][stock]
                buy_price = info["price"]
                if avail <= 0:
                    continue
                for sell_year, other in timeline.items():
                    sell_price = other.get(stock, {}).get("price")
                    if sell_price is None or sell_price <= buy_price:
                        continue
                    if not can_reach([buy_year, sell_year]):
                        continue
                    take = min(avail, capital // buy_price)
                    if take <= 0:
                        continue
                    profit = take * (sell_price - buy_price)
                    travel = abs(year - buy_year) + abs(buy_year - sell_year)
                    roi = sell_price / buy_price
                    score = (roi, profit, -travel)
                    if best is None or score > best[0]:
                        best = (score, buy_year, sell_year)

        if best is None:
            break
        _score, buy_year, sell_year = best
        hops = _line_path(year, buy_year, timeline) + _line_path(buy_year, sell_year, timeline)[1:]
        acts, new_cap, new_qty, new_hold = _walk(hops, capital, qty, holdings, timeline)
        if not acts or not any(item.startswith("b-") for item in acts):
            jump(buy_year)
            ranked = []
            for name, info in timeline.get(buy_year, {}).items():
                sell_price = _price(timeline, sell_year, name)
                if sell_price is None or sell_price <= info["price"] or qty[buy_year][name] <= 0:
                    continue
                ranked.append((sell_price / info["price"], name))
            ranked.sort(reverse=True)
            bought = 0
            for _roi, name in ranked:
                bought += buy(name, qty[buy_year][name])
            if bought == 0:
                break
            jump(sell_year)
            for name in list(holdings):
                sell(name)
            continue
        used = _jump_cost(acts)
        if used > energy:
            break
        energy -= used
        actions.extend(acts)
        capital, qty, holdings = new_cap, new_qty, new_hold
        year = sell_year if acts else year
        for action in acts:
            bits = action.split("-")
            if bits[0] == "j":
                year = int(bits[2])
        for name in list(holdings):
            if _price(timeline, year, name) is not None:
                future_home = _price(timeline, HOME, name)
                now = _price(timeline, year, name)
                if future_home is None or now >= future_home or not can_reach([HOME]):
                    sell(name)

    if year != HOME:
        jump(HOME)
    for name in list(holdings):
        if _price(timeline, HOME, name) is not None:
            sell(name)
    return actions


def _line_path(a, b, timeline):
    points = sorted(set(timeline) | {a, b})
    if a <= b:
        return [y for y in points if a <= y <= b]
    return [y for y in points if b <= y <= a][::-1]


def _finish(actions, holdings, timeline):
    year = HOME
    for action in actions:
        bits = action.split("-")
        if bits[0] == "j":
            year = int(bits[2])
    extra = []
    if year != HOME:
        extra.append(f"j-{year}-{HOME}")
    for name, have in holdings.items():
        if have > 0 and _price(timeline, HOME, name) is not None:
            extra.append(f"s-{name}-{have}")
    return list(actions) + extra


def _jump_cost(actions):
    total = 0
    for action in actions:
        bits = action.split("-")
        if bits[0] == "j":
            total += abs(int(bits[1]) - int(bits[2]))
    return total


def _replay(seed, actions):
    try:
        energy = int(seed["energy"])
        capital = int(seed["capital"])
        start = capital
        timeline = seed["timeline"]
        qty = _copy_qty(_qty_map(timeline))
        holdings = {}
        year = HOME
        for action in actions:
            if not isinstance(action, str):
                return None
            bits = action.split("-")
            if len(bits) < 3:
                return None
            kind = bits[0]
            if kind == "j":
                source, dest = int(bits[1]), int(bits[2])
                if source != year:
                    return None
                cost = abs(dest - source)
                if cost > energy:
                    return None
                energy -= cost
                year = dest
                continue
            name = "-".join(bits[1:-1])
            take = int(bits[-1])
            if take <= 0:
                return None
            info = timeline.get(year, {}).get(name)
            if info is None:
                return None
            if kind == "b":
                if qty[year][name] < take:
                    return None
                cost = take * info["price"]
                if cost > capital:
                    return None
                capital -= cost
                qty[year][name] -= take
                holdings[name] = holdings.get(name, 0) + take
            elif kind == "s":
                if holdings.get(name, 0) < take:
                    return None
                capital += take * info["price"]
                holdings[name] -= take
            else:
                return None
        if year != HOME:
            return None
        return capital - start
    except (TypeError, ValueError, KeyError):
        return None
