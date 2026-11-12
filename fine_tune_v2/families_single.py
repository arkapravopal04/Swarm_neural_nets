"""
fine_tune_v2/families_single.py -- single-question scenario families
(TRAINING_SPEC.md §6).

Each family makes a scenario from a random generator: a request (one question
with one exact answer, phrased from >= 5 templates with figure placeholders),
the solver's answer, the answer as a correct `run_code` program prints it, and
-- for families short enough to work by hand inside the runtime's 3-sentence /
60-word REPORT trim -- the working steps. Every number in a template is a
placeholder filled by code (D3: no teacher model).

These are structure TYPES of the benchmark's tiers 1-4 (finite analysis,
optimisation, state reconstruction, short calculations with an exception),
never its items; fine_tune_v2/validate.py checks contamination.
"""
import math
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import Optional


@dataclass
class Single:
    family: str
    domain: str
    request: str                 # the question as the person asks it
    answer: object               # float / int / str label -- the solver's value
    printed: str                 # what the gold program prints (E3 copies it)
    unit: str                    # unit words for the answer sentence ("" for labels)
    code: str                    # gold run_code program
    answer_sentence: str         # "{printed} {unit}" in a sentence, the final line
    steps: Optional[list] = None  # E1 working (<= 2 short sentences), or None
    wrong: Optional[tuple] = None  # (wrong report text, judge critique naming the error)
    tool_ok: bool = True         # False for domains where run_code is blocked
    figures: dict = field(default_factory=dict)

    def __post_init__(self):
        # One print format for every program, the same rule as fnum(): a
        # rounded value with trailing zeros trimmed, so 10.0 prints "10" and
        # 1e-05 prints "0.00001" -- what the solver's `printed` says.
        if "print(round(" in self.code:
            self.code = SHOW_HELPER + self.code.replace("print(round(", "show(round(")


SHOW_HELPER = ("def show(x):\n"
               "    print(f'{float(x):.8f}'.rstrip('0').rstrip('.'))\n\n")


def fnum(x, places=8):
    """A value as the gold program prints it: integers bare, others rounded
    to `places` decimals with trailing zeros trimmed (benchmark tolerance is
    1e-6, so 8 places keeps every answer well inside it)."""
    if isinstance(x, Fraction):
        x = x.numerator / x.denominator if x.denominator != 1 else int(x)
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    if isinstance(x, int):
        return str(x)
    s = f"{x:.{places}f}".rstrip("0").rstrip(".")
    return s


def comma(n):
    """A request-style figure: 40000 -> '40,000'."""
    return f"{n:,}" if isinstance(n, int) else str(n)


def pick(rng, items):
    return items[rng.randrange(len(items))]


def _print_code(body, expr):
    return (body.rstrip() + "\n" if body else "") + f"print(round({expr}, 8))"


# ---------------------------------------------------------------- families

def gamblers_ruin(rng, p):
    k = rng.randint(2, 6)
    n = rng.randint(k + 2, k + 6)
    num = rng.choice([2, 3, 4, 3])
    den = rng.choice([5, 7, 8, 10])
    if num >= den:
        num = den - 1
    pw = Fraction(num, den)
    q = 1 - pw
    if pw == q:
        prob = Fraction(k, n)
    else:
        r = q / pw
        prob = (1 - r ** k) / (1 - r ** n)
    phr = [
        "A player starts with {k} chips and bets one chip per round. Each round they win a chip with probability {num}/{den} and lose one otherwise. Play stops when they reach {n} chips or run out. What is the probability that they reach {n} chips first?",
        "A random walk starts at {k} on the integers 0 to {n}. Each step goes up by 1 with probability {num}/{den} and down by 1 otherwise, and the walk stops at 0 or {n}. What is the probability it stops at {n}?",
        "A trader holds {k} units and each day gains one unit with probability {num}/{den} or loses one. They stop at {n} units or at zero. Find the probability of reaching {n} units before zero.",
        "Starting from state {k}, a process moves to the next state up with probability {num}/{den} and to the next state down otherwise, with absorbing states 0 and {n}. What is the probability of absorption at {n}?",
        "A gambler with {k} tokens plays a fair-sized game where each round is won with probability {num}/{den}; a win adds a token and a loss removes one. The gambler quits on reaching {n} tokens or going broke. Give the probability of quitting with {n} tokens.",
        "On a line of positions 0 to {n}, a token begins at {k}. Each move is right with probability {num}/{den}, else left, until it hits an end. What is the chance it ends at position {n}?",
    ]
    req = phr[p % len(phr)].format(k=k, n=n, num=num, den=den)
    req += " Give the answer to at least 6 decimal places."
    code = (f"from fractions import Fraction\np = Fraction({num}, {den})\nq = 1 - p\n"
            f"k, n = {k}, {n}\n"
            "prob = Fraction(k, n) if p == q else (1 - (q / p) ** k) / (1 - (q / p) ** n)\n"
            "print(round(float(prob), 8))")
    printed = fnum(float(prob))
    wrong_val = fnum(float(Fraction(k, n)))
    return Single("gamblers_ruin", "Finance & Quantitative Analysis", req, float(prob), printed, "",
                  code, f"The probability of reaching {n} first is {printed}.",
                  wrong=(f"The probability is {k}/{n} = {wrong_val}.",
                         f"The output uses {k}/{n}, which is only right for a fair game; "
                         f"here a round is won with probability {num}/{den}."),
                  figures=dict(k=k, n=n, num=num, den=den))


def markov_steps(rng, p):
    # states A -> (A, B, done); B -> (A, B, done); expected steps from A to done
    pa = [rng.randint(1, 4) for _ in range(3)]
    pb = [rng.randint(1, 4) for _ in range(3)]
    sa, sb = sum(pa), sum(pb)
    paa, pab, pad = (Fraction(x, sa) for x in pa)
    pba, pbb, pbd = (Fraction(x, sb) for x in pb)
    # E_A = 1 + paa E_A + pab E_B ; E_B = 1 + pba E_A + pbb E_B
    det = (1 - paa) * (1 - pbb) - pab * pba
    ea = ((1 - pbb) + pab) / det
    names = pick(rng, [("idle", "busy"), ("draft", "review"), ("north", "south"),
                       ("queue", "service"), ("red", "blue")])
    a, b = names
    phr = [
        "A process is in state {a} or {b} until it finishes. From {a} it stays in {a} with probability {paa}, moves to {b} with probability {pab} and finishes with probability {pad}. From {b} it moves to {a} with probability {pba}, stays in {b} with probability {pbb} and finishes with probability {pbd}. Starting in {a}, what is the expected number of steps until it finishes?",
        "Each step a ticket is either in {a}, in {b}, or closed. From {a}: {a} {paa}, {b} {pab}, closed {pad}. From {b}: {a} {pba}, {b} {pbb}, closed {pbd}. A ticket starts in {a}. Find the expected number of steps until it is closed.",
        "A Markov chain has transient states {a} and {b} and one absorbing state. Transition probabilities: {a} to {a} {paa}, {a} to {b} {pab}, {a} to absorbing {pad}; {b} to {a} {pba}, {b} to {b} {pbb}, {b} to absorbing {pbd}. What is the expected time to absorption from {a}?",
        "A robot alternates between zones {a} and {b} and stops when it docks. In {a} it stays with probability {paa}, goes to {b} with probability {pab}, docks with probability {pad}; in {b} it goes to {a} with probability {pba}, stays with probability {pbb}, docks with probability {pbd}. Each move takes one step. Expected steps to dock, starting in {a}?",
        "From state {a} the next state is {a}, {b} or done with probabilities {paa}, {pab}, {pad}; from {b} it is {a}, {b} or done with probabilities {pba}, {pbb}, {pbd}. What is the expected number of transitions until done, starting from {a}?",
    ]
    f = lambda x: f"{x.numerator}/{x.denominator}"
    req = phr[p % len(phr)].format(a=a, b=b, paa=f(paa), pab=f(pab), pad=f(pad),
                                   pba=f(pba), pbb=f(pbb), pbd=f(pbd))
    req += " Give the answer to at least 6 decimal places."
    code = ("from fractions import Fraction as F\n"
            f"paa, pab = F({paa.numerator}, {paa.denominator}), F({pab.numerator}, {pab.denominator})\n"
            f"pba, pbb = F({pba.numerator}, {pba.denominator}), F({pbb.numerator}, {pbb.denominator})\n"
            "# E_A = 1 + paa*E_A + pab*E_B ; E_B = 1 + pba*E_A + pbb*E_B\n"
            "det = (1 - paa) * (1 - pbb) - pab * pba\n"
            "e_a = ((1 - pbb) + pab) / det\n"
            "print(round(float(e_a), 8))")
    printed = fnum(float(ea))
    return Single("markov_steps", "Theoretical Mathematics", req, float(ea), printed, "steps",
                  code, f"The expected number of steps from {a} is {printed}.",
                  figures=dict(pa=pa, pb=pb))


def dice_until_sum(rng, p):
    target = rng.randint(4, 12)
    faces = rng.choice([4, 6, 6, 8])
    e = [Fraction(0)] * (target + faces + 1)
    for s in range(target - 1, -1, -1):
        e[s] = 1 + sum(e[s + k] if s + k < target else 0 for k in range(1, faces + 1)) / faces
    ans = e[0]
    phr = [
        "A fair {faces}-sided die numbered 1 to {faces} is rolled repeatedly and the results are added. What is the expected number of rolls until the running total is at least {target}?",
        "You keep rolling a fair die with faces 1 to {faces}, summing the rolls, and stop as soon as the sum reaches {target} or more. Find the expected number of rolls.",
        "Rolls of a fair {faces}-faced die (values 1 to {faces}) are accumulated until the total first reaches at least {target}. How many rolls are needed on average?",
        "A counter starts at 0 and each turn adds the result of a fair roll of a {faces}-sided die (1 to {faces}). The game ends when the counter is {target} or higher. What is the expected number of turns?",
        "Compute the expected number of throws of a fair die with faces 1 to {faces} needed for the sum of the throws to reach at least {target}.",
    ]
    req = phr[p % len(phr)].format(faces=faces, target=target) + " Give the answer to at least 6 decimal places."
    code = ("from fractions import Fraction\n"
            f"target, faces = {target}, {faces}\n"
            "e = [Fraction(0)] * (target + faces + 1)\n"
            "for s in range(target - 1, -1, -1):\n"
            "    e[s] = 1 + sum(e[s + k] if s + k < target else 0 for k in range(1, faces + 1)) / faces\n"
            "print(round(float(e[0]), 8))")
    printed = fnum(float(ans))
    return Single("dice_until_sum", "Theoretical Mathematics", req, float(ans), printed, "rolls",
                  code, f"The expected number of rolls is {printed}.",
                  figures=dict(target=target, faces=faces))


def grid_paths(rng, p):
    rows, cols = rng.randint(3, 6), rng.randint(3, 6)
    cells = [(r, c) for r in range(rows) for c in range(cols)
             if (r, c) not in ((0, 0), (rows - 1, cols - 1))]
    blocked = sorted(rng.sample(cells, rng.randint(1, 3)))
    ways = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if (r, c) in blocked:
                continue
            if (r, c) == (0, 0):
                ways[r][c] = 1
                continue
            ways[r][c] = (ways[r - 1][c] if r else 0) + (ways[r][c - 1] if c else 0)
    ans = ways[rows - 1][cols - 1]
    bl = ", ".join(f"({r + 1}, {c + 1})" for r, c in blocked)
    phr = [
        "A grid has {rows} rows and {cols} columns. A robot starts at the top-left cell (1, 1) and must reach the bottom-right cell ({rows}, {cols}), moving only down or right one cell at a time. The cells {bl} (row, column) are blocked. How many different paths are there?",
        "On a {rows} by {cols} grid of cells, count the monotone paths (each move one cell down or one cell right) from cell (1, 1) to cell ({rows}, {cols}) that avoid the blocked cells {bl}, given as (row, column).",
        "A courier walks a street grid of {rows} rows by {cols} columns of blocks, only going south or east, from corner block (1, 1) to block ({rows}, {cols}). Blocks {bl} (row, column) are closed. How many routes are possible?",
        "How many lattice paths using only down and right steps go from (1, 1) to ({rows}, {cols}) in a {rows}x{cols} cell grid if the cells {bl} (row, column) cannot be entered?",
        "A token moves through a {rows}-row, {cols}-column board from the cell in row 1, column 1 to the cell in row {rows}, column {cols}, stepping only down or right. Cells {bl} (row, column) are off limits. Count the possible paths.",
    ]
    req = phr[p % len(phr)].format(rows=rows, cols=cols, bl=bl)
    code = (f"rows, cols = {rows}, {cols}\nblocked = {set((r, c) for r, c in blocked)!r}\n"
            "ways = [[0] * cols for _ in range(rows)]\n"
            "for r in range(rows):\n"
            "    for c in range(cols):\n"
            "        if (r, c) in blocked:\n            continue\n"
            "        ways[r][c] = 1 if (r, c) == (0, 0) else (ways[r - 1][c] if r else 0) + (ways[r][c - 1] if c else 0)\n"
            "print(ways[rows - 1][cols - 1])")
    return Single("grid_paths", "Computer Science", req, ans, str(ans), "paths",
                  code, f"There are {ans} paths.", figures=dict(rows=rows, cols=cols))


def knapsack(rng, p):
    n = 5
    w = [rng.randint(2, 9) for _ in range(n)]
    v = [rng.randint(3, 20) for _ in range(n)]
    cap = rng.randint(max(w), sum(w) - 3)
    best = 0
    for r in range(n + 1):
        for combo in combinations(range(n), r):
            if sum(w[i] for i in combo) <= cap:
                best = max(best, sum(v[i] for i in combo))
    names = pick(rng, [["tent", "stove", "rope", "lamp", "map"],
                       ["drill", "saw", "level", "clamp", "sander"],
                       ["server A", "server B", "server C", "server D", "server E"],
                       ["crate 1", "crate 2", "crate 3", "crate 4", "crate 5"]])
    items = "; ".join(f"{names[i]}: weight {w[i]} kg, value {v[i]}" for i in range(n))
    phr = [
        "A pack can carry at most {cap} kg. The items are {items}. Each item can be taken at most once. What is the largest total value that fits?",
        "Choose a subset of these items with total weight at most {cap} kg to maximise total value: {items}. What is the maximum value?",
        "A van has room for {cap} kg. Available: {items}. No item may be split or taken twice. Find the highest total value the van can carry.",
        "With a weight limit of {cap} kg and the items {items}, what is the best achievable total value if each item is either taken whole or left?",
        "Solve this 0/1 selection: limit {cap} kg; {items}. Report the maximum total value.",
    ]
    req = phr[p % len(phr)].format(cap=cap, items=items)
    code = (f"from itertools import combinations\nw = {w}\nv = {v}\ncap = {cap}\n"
            "best = 0\nfor r in range(len(w) + 1):\n"
            "    for c in combinations(range(len(w)), r):\n"
            "        if sum(w[i] for i in c) <= cap:\n"
            "            best = max(best, sum(v[i] for i in c))\nprint(best)")
    return Single("knapsack", "Computer Science", req, best, str(best), "",
                  code, f"The maximum total value is {best}.", figures=dict(cap=cap))


def makespan(rng, p):
    jobs = [rng.randint(2, 15) for _ in range(rng.randint(5, 7))]
    best = sum(jobs)
    for r in range(len(jobs) + 1):
        for combo in combinations(range(len(jobs)), r):
            a = sum(jobs[i] for i in combo)
            best = min(best, max(a, sum(jobs) - a))
    js = ", ".join(str(j) for j in jobs)
    phr = [
        "Jobs taking {js} minutes must each run on one of two identical servers, one job at a time per server. What is the earliest time, in minutes, by which all jobs can be finished?",
        "Two identical machines process the tasks with durations {js} (minutes); a task cannot be split. What is the minimum possible makespan in minutes?",
        "Split the build steps lasting {js} minutes between two runners so the last one finishes as early as possible. How many minutes does that take?",
        "Assign batch jobs of {js} minutes to two workers working in parallel, no job split. Find the smallest finishing time in minutes.",
        "Given durations {js} minutes and two parallel lanes, what is the shortest time to complete every item if each item goes wholly to one lane?",
    ]
    req = phr[p % len(phr)].format(js=js)
    code = (f"from itertools import combinations\njobs = {jobs}\ntotal = sum(jobs)\nbest = total\n"
            "for r in range(len(jobs) + 1):\n"
            "    for c in combinations(range(len(jobs)), r):\n"
            "        a = sum(jobs[i] for i in c)\n"
            "        best = min(best, max(a, total - a))\nprint(best)")
    return Single("makespan", "Software Engineering", req, best, str(best), "minutes",
                  code, f"All jobs can finish by {best} minutes.", figures=dict(jobs=jobs))


def inventory_replay(rng, p):
    start = rng.randint(80, 300)
    events, stock = [], start
    for _ in range(rng.randint(4, 6)):
        kind = rng.choice(["sold", "received", "returned", "returned_damaged"])
        q = rng.randint(3, 40)
        if kind == "sold":
            q = min(q, stock)
            stock -= q
            events.append(f"{q} sold")
        elif kind == "received":
            stock += q
            events.append(f"{q} received from the supplier")
        elif kind == "returned":
            stock += q
            events.append(f"{q} returned in good condition")
        else:
            events.append(f"{q} returned damaged")
    item = pick(rng, ["chairs", "kettles", "laptops", "helmets", "lamps"])
    ev = "; ".join(events)
    phr = [
        "A shop starts the week with {start} {item} in stock. During the week: {ev}. Damaged returns are written off and never go back into stock. How many {item} are in stock at the end of the week?",
        "Opening stock: {start} {item}. Movements in order: {ev}. Items returned damaged are scrapped, not restocked. What is the closing stock?",
        "A warehouse holds {start} {item}. Then: {ev}. Only returns in good condition are restocked. How many {item} remain?",
        "Replay this log for {item}, starting from {start} units: {ev}. A damaged return does not change stock. Final stock?",
        "Begin with {start} {item}. The ledger shows {ev}. Write-offs for damaged returns mean those units never re-enter stock. What is the count at the end?",
    ]
    req = phr[p % len(phr)].format(start=start, item=item, ev=ev)
    code_lines = [f"stock = {start}"]
    for e in events:
        q = int(e.split()[0])
        if e.endswith("sold"):
            code_lines.append(f"stock -= {q}  # sold")
        elif "damaged" in e:
            code_lines.append(f"# {q} returned damaged: written off, no change")
        else:
            code_lines.append(f"stock += {q}  # {e.split(' ', 1)[1]}")
    code_lines.append("print(stock)")
    return Single("inventory_replay", "Data Engineering", req, stock, str(stock), item,
                  "\n".join(code_lines), f"The closing stock is {stock} {item}.",
                  figures=dict(start=start))


def loan_payment(rng, p):
    principal = rng.choice([50000, 120000, 250000, 400000, 75000, 180000])
    rate = rng.choice([6, 7.5, 8, 9, 10.5, 12])
    years = rng.choice([2, 3, 5, 10])
    r = rate / 100 / 12
    n = years * 12
    pay = principal * r / (1 - (1 + r) ** -n)
    phr = [
        "A loan of {P} rupees is repaid in equal monthly instalments over {y} years at {rate}% annual interest, compounded monthly. What is the monthly instalment in rupees?",
        "What equal monthly payment repays {P} rupees in {y} years at an annual rate of {rate}% charged monthly?",
        "A borrower takes {P} rupees at {rate}% per year (monthly compounding) for {y} years with fixed monthly payments. Find the payment.",
        "Compute the monthly amortising payment on a {P}-rupee loan, {rate}% nominal annual interest compounded monthly, term {y} years.",
        "Fixed monthly instalments clear a {P} rupee loan after {y} years; interest is {rate}% a year, applied monthly. How much is each instalment?",
    ]
    req = phr[p % len(phr)].format(P=comma(principal), y=years, rate=rate) + " Give the answer to 2 decimal places."
    code = (f"P, annual, years = {principal}, {rate}, {years}\nr = annual / 100 / 12\nn = years * 12\n"
            "pay = P * r / (1 - (1 + r) ** -n)\nprint(round(pay, 2))")
    printed = fnum(round(pay, 2))
    return Single("loan_payment", "Finance & Quantitative Analysis", req, round(pay, 2), printed, "rupees",
                  code, f"The monthly instalment is {printed} rupees.",
                  wrong=(f"The instalment is {comma(principal)} / {n} = {fnum(principal / n, 2)} rupees.",
                         "The output divides the principal by the number of months and leaves out the interest entirely."),
                  figures=dict(P=principal))


def resistor_divider(rng, p):
    v = rng.choice([5, 9, 12, 24, 48])
    r1, r2 = rng.choice([100, 220, 330, 470, 1000]), rng.choice([100, 150, 220, 680, 1000])
    total = r1 + r2
    cur = Fraction(v, total)
    phr = [
        "Two resistors of {r1} ohms and {r2} ohms are connected in series across a {v} V supply. What current flows, in amperes?",
        "A {v} V source drives a series string of a {r1}-ohm and a {r2}-ohm resistor. Find the current in amperes.",
        "What is the current through a series circuit of {r1} ohms and {r2} ohms powered by {v} volts?",
        "A {r1} ohm resistor and a {r2} ohm resistor in series sit across {v} V. Give the circuit current in A.",
        "In series, resistors of {r1} and {r2} ohms carry current from a {v}-volt battery. How many amperes flow?",
    ]
    req = phr[p % len(phr)].format(v=v, r1=r1, r2=r2) + " Give the answer to at least 6 decimal places."
    printed = fnum(float(cur))
    steps = [f"The series resistance is {r1} + {r2} = {total} ohms.",
             f"The current is {v} / {total} = {printed} A."]
    code = _print_code(f"r_total = {r1} + {r2}", f"{v} / r_total")
    return Single("resistor_divider", "Electrical & Computer Engineering", req, float(cur), printed, "A",
                  code, f"The current is {printed} A.", steps=steps,
                  wrong=(f"The current is {v} / {r1} = {fnum(v / r1)} A.",
                         f"The output uses only the {r1}-ohm resistor; in series the resistances add."),
                  figures=dict(v=v, r1=r1, r2=r2))


def rc_charge(rng, p):
    r = rng.choice([1000, 2200, 4700, 10000])
    c_uf = rng.choice([10, 47, 100, 220])
    frac = rng.choice([50, 63, 80, 90, 95])
    tau = r * c_uf / 1e6
    t = -tau * math.log(1 - frac / 100)
    phr = [
        "A capacitor of {c} microfarads charges through a {r} ohm resistor from 0 V toward a fixed supply. How many seconds does it take to reach {f}% of the supply voltage?",
        "An RC circuit has R = {r} ohms and C = {c} uF, starting uncharged. Find the time in seconds for the capacitor voltage to reach {f}% of its final value.",
        "How long, in seconds, does an initially empty {c} uF capacitor take to charge to {f}% of the source voltage through {r} ohms?",
        "With {r} ohms in series with {c} microfarads, starting at zero volts, when does the capacitor reach {f}% of the step voltage? Answer in seconds.",
        "Compute the charging time to {f}% for R = {r} ohm, C = {c} microfarad, from zero initial charge, in seconds.",
    ]
    req = phr[p % len(phr)].format(r=r, c=c_uf, f=frac) + " Give the answer to at least 6 decimal places."
    code = (f"import math\nR, C = {r}, {c_uf}e-6\ntau = R * C\n"
            f"t = -tau * math.log(1 - {frac} / 100)\nprint(round(t, 8))")
    printed = fnum(t)
    return Single("rc_charge", "Electrical & Computer Engineering", req, t, printed, "seconds",
                  code, f"It takes {printed} seconds.", figures=dict(r=r, c=c_uf, f=frac))


def beam_moment(rng, p):
    load = rng.choice([2, 4, 5, 8, 10, 12])
    span = rng.choice([s for s in (3, 4, 5, 6, 8) if s != load])
    prod = load * span
    m = Fraction(prod, 4)
    phr = [
        "A simply supported beam of span {L} m carries a single point load of {P} kN at midspan. What is the maximum bending moment in kN m?",
        "Find the largest bending moment, in kN m, in a {L} m simply supported beam with a {P} kN load at its centre.",
        "A {P} kN load sits at the middle of a simply supported {L}-metre beam. Give the peak bending moment in kN m.",
        "What is the maximum moment (kN m) for a centre point load of {P} kN on a simply supported span of {L} m?",
        "A footbridge plank spans {L} m between simple supports and takes {P} kN at midspan. Maximum bending moment in kN m?",
    ]
    req = phr[p % len(phr)].format(P=load, L=span)
    printed = fnum(float(m))
    steps = [f"For a midspan point load the maximum moment is P x L / 4, and {load} x {span} = {prod}.",
             f"So the maximum moment is {prod} / 4 = {printed} kN m."]
    code = _print_code("", f"{load} * {span} / 4")
    return Single("beam_moment", "Mechanical Engineering", req, float(m), printed, "kN m",
                  code, f"The maximum bending moment is {printed} kN m.", steps=None,
                  wrong=(f"The maximum moment is {load} x {span} = {prod} kN m.",
                         "The output uses P x L, the cantilever formula; a simply supported beam with a midspan load has P x L / 4."),
                  figures=dict(P=load, L=span))


def pipe_flow(rng, p):
    d_mm = rng.choice([25, 40, 50, 80, 100])
    v = rng.choice([0.5, 1, 1.5, 2, 2.5])
    area = math.pi * (d_mm / 1000) ** 2 / 4
    q = area * v * 1000  # L/s
    phr = [
        "Water flows at {v} m/s through a pipe of {d} mm inside diameter. What is the flow rate in litres per second?",
        "Find the volumetric flow, in L/s, for a mean velocity of {v} m/s in a circular pipe with {d} mm bore.",
        "A {d} mm diameter line carries liquid at {v} metres per second. How many litres pass each second?",
        "What flow rate in litres per second corresponds to {v} m/s in a round duct {d} mm across?",
        "Compute Q in L/s for velocity {v} m/s and internal diameter {d} mm.",
    ]
    req = phr[p % len(phr)].format(v=v, d=d_mm) + " Give the answer to at least 6 decimal places."
    code = (f"import math\nd = {d_mm} / 1000\nv = {v}\narea = math.pi * d ** 2 / 4\n"
            "print(round(area * v * 1000, 8))")
    printed = fnum(q)
    return Single("pipe_flow", "Chemical Engineering & Materials", req, q, printed, "L/s",
                  code, f"The flow rate is {printed} L/s.", figures=dict(d=d_mm, v=v))


def limiting_reagent(rng, p):
    # 2 H2 + O2 -> 2 H2O style: aA + bB -> cC
    a, b, c = pick(rng, [(2, 1, 2), (1, 1, 1), (1, 2, 1), (3, 1, 2)])
    ma, mb, mc = pick(rng, [(2.016, 32.00, 18.015), (28.01, 32.00, 44.01), (16.04, 32.00, 44.01),
                            (2.016, 28.02, 17.03)])
    ga, gb = rng.choice([4, 10, 12, 20]), rng.choice([16, 32, 40, 64])
    na, nb = ga / ma, gb / mb
    ext = min(na / a, nb / b)
    gc = ext * c * mc
    phr = [
        "In the reaction {a} A + {b} B -> {c} C, A has molar mass {ma} g/mol, B {mb} g/mol and C {mc} g/mol. Starting from {ga} g of A and {gb} g of B, what is the maximum mass of C in grams?",
        "{ga} g of reactant A (molar mass {ma} g/mol) and {gb} g of reactant B ({mb} g/mol) react as {a} A + {b} B -> {c} C. C has molar mass {mc} g/mol. Find the theoretical yield of C in grams.",
        "Given {a}A + {b}B -> {c}C with molar masses A {ma}, B {mb}, C {mc} g/mol, how many grams of C can form from {ga} g A and {gb} g B?",
        "What mass of product C (molar mass {mc} g/mol) forms at most when {ga} g of A ({ma} g/mol) meets {gb} g of B ({mb} g/mol) in the reaction {a} A + {b} B -> {c} C?",
        "Find the limiting reagent and the maximum grams of C for {a} A + {b} B -> {c} C using {ga} g of A (M = {ma} g/mol) and {gb} g of B (M = {mb} g/mol); C has M = {mc} g/mol. Report the mass of C.",
    ]
    req = phr[p % len(phr)].format(a=a, b=b, c=c, ma=ma, mb=mb, mc=mc, ga=ga, gb=gb) + " Give the answer to at least 6 decimal places."
    code = (f"na = {ga} / {ma}\nnb = {gb} / {mb}\nextent = min(na / {a}, nb / {b})\n"
            f"print(round(extent * {c} * {mc}, 8))")
    printed = fnum(gc)
    return Single("limiting_reagent", "Chemical Engineering & Materials", req, gc, printed, "g",
                  code, f"At most {printed} g of C can form.", figures=dict(ga=ga, gb=gb))


def dose_calc(rng, p):
    wt = rng.choice([12, 18, 25, 40, 60, 75, 90])
    per_kg = rng.choice([5, 10, 15, 20])
    cap = rng.choice([500, 750, 1000])
    conc = rng.choice([25, 50, 100])
    dose = min(wt * per_kg, cap)
    vol = Fraction(dose, conc)
    raw = wt * per_kg
    phr = [
        "A drug is dosed at {k} mg per kg of body weight, with a maximum single dose of {cap} mg. It comes as a solution of {c} mg per mL. How many mL should a {w} kg patient receive?",
        "For a {w} kg patient, the order is {k} mg/kg up to a ceiling of {cap} mg. The stock solution is {c} mg/mL. What volume in mL is drawn up?",
        "Calculate the volume (mL) of a {c} mg/mL solution for a {w} kg person dosed at {k} mg/kg, never exceeding {cap} mg per dose.",
        "A {w}-kilogram patient needs {k} mg per kilogram, capped at {cap} mg. The vial strength is {c} mg/mL. How many millilitres?",
        "Dose rule: {k} mg/kg, maximum {cap} mg. Concentration {c} mg/mL. Weight {w} kg. Volume to give, in mL?",
    ]
    req = phr[p % len(phr)].format(k=per_kg, cap=cap, c=conc, w=wt)
    printed = fnum(float(vol))
    if raw > cap:
        steps = [f"{wt} x {per_kg} = {raw} mg is above the {cap} mg maximum, so the dose is {cap} mg.",
                 f"The volume is {cap} / {conc} = {printed} mL."]
        wrong = (f"The volume is {raw} / {conc} = {fnum(raw / conc)} mL.",
                 f"The output gives {raw} mg, which is over the {cap} mg maximum single dose.")
    else:
        steps = [f"The dose is {wt} x {per_kg} = {raw} mg, within the {cap} mg maximum.",
                 f"The volume is {raw} / {conc} = {printed} mL."]
        wrong = None
    code = _print_code(f"dose = min({wt} * {per_kg}, {cap})", f"dose / {conc}")
    return Single("dose_calc", "Biomedical & Life Sciences", req, float(vol), printed, "mL",
                  code, f"Give {printed} mL.", steps=steps, wrong=wrong,
                  figures=dict(w=wt, k=per_kg, cap=cap, c=conc))


def pipeline_hours(rng, p):
    records = rng.choice([1800000, 3600000, 7200000, 5400000, 900000])
    rate = rng.choice([250, 500, 1000, 1500, 2000])
    secs = Fraction(records, rate)
    hours = secs / 3600
    phr = [
        "A job loads {N} records at a steady {r} records per second. How many hours does the load take?",
        "An ETL step processes {r} rows per second. How long, in hours, does it need for {N} rows?",
        "At {r} records per second, how many hours does it take to ingest {N} records?",
        "A migration copies {N} records with throughput {r} records/s. Duration in hours?",
        "Find the time in hours for {N} events to be processed at {r} events per second.",
    ]
    req = phr[p % len(phr)].format(N=comma(records), r=comma(rate))
    s_txt = fnum(float(secs))
    printed = fnum(float(hours))
    steps = [f"{comma(records)} / {comma(rate)} = {s_txt} seconds.",
             f"{s_txt} / 3600 = {printed} hours."]
    code = _print_code(f"seconds = {records} / {rate}", "seconds / 3600")
    return Single("pipeline_hours", "Data Engineering", req, float(hours), printed, "hours",
                  code, f"The load takes {printed} hours.", steps=None,
                  wrong=(f"The load takes {comma(records)} / {comma(rate)} = {s_txt} hours.",
                         "The output reports seconds as hours; the division gives seconds, which still need dividing by 3600."),
                  figures=dict(N=records, r=rate))


def loop_count(rng, p):
    n = rng.randint(8, 60)
    prod = n * (n - 1)
    ans = prod // 2
    phr = [
        "A routine compares every pair of distinct items in a list of {n} items exactly once. How many comparisons does it make?",
        "for i in 1..{n}: for j in i+1..{n}: compare(i, j). How many times is compare called?",
        "In a round-robin where each of {n} teams plays every other team once, how many games are played?",
        "How many unordered pairs can be formed from {n} distinct elements?",
        "A handshake check: {n} people each shake hands once with every other person. How many handshakes happen?",
    ]
    req = phr[p % len(phr)].format(n=n)
    steps = [f"Each pair is counted once, so the count is {n} x {n - 1} / 2, and {n} x {n - 1} = {prod}.",
             f"{prod} / 2 = {ans}."]
    code = _print_code("", f"{n} * ({n} - 1) // 2")
    return Single("loop_count", "Computer Science", req, ans, str(ans), "",
                  code, f"The count is {ans}.", steps=None,
                  wrong=(f"The count is {n} x {n - 1} = {prod}.",
                         "The output counts ordered pairs; each pair is counted twice and must be halved."),
                  figures=dict(n=n))


def inclusion_exclusion(rng, p):
    n = rng.choice([100, 250, 500, 1000, 360, 840])
    a, b = rng.sample([2, 3, 4, 5, 6, 7, 9, 11], 2)
    lcm = a * b // math.gcd(a, b)
    ans = n // a + n // b - n // lcm
    phr = [
        "How many integers from 1 to {n} inclusive are divisible by {a} or by {b}?",
        "Count the whole numbers between 1 and {n} (inclusive) that are multiples of {a} or of {b}.",
        "In the range 1..{n}, how many numbers are divisible by at least one of {a} and {b}?",
        "Find the number of integers k with 1 <= k <= {n} such that {a} divides k or {b} divides k.",
        "Of the numbers 1 through {n}, how many are multiples of {a}, of {b}, or of both?",
    ]
    req = phr[p % len(phr)].format(n=n, a=a, b=b)
    code = f"n, a, b = {n}, {a}, {b}\nprint(sum(1 for k in range(1, n + 1) if k % a == 0 or k % b == 0))"
    return Single("inclusion_exclusion", "Theoretical Mathematics", req, ans, str(ans), "",
                  code, f"There are {ans} such integers.",
                  wrong=(f"There are {n // a} + {n // b} = {n // a + n // b} such integers.",
                         f"The output counts multiples of both {a} and {b} twice; they must be subtracted once."),
                  figures=dict(n=n, a=a, b=b))


def fuel_burn(rng, p):
    rate = rng.choice([600, 900, 1200, 2400, 3000])
    minutes = rng.choice([20, 35, 45, 90, 150])
    prod = rate * minutes
    burn = Fraction(prod, 60)
    phr = [
        "An engine burns fuel at {r} kg per hour. How many kg does it burn in {m} minutes?",
        "At a fuel flow of {r} kg/h, what mass of fuel is used over a {m}-minute climb?",
        "A turbine consumes {r} kilograms of fuel each hour. Find the fuel used in {m} minutes, in kg.",
        "How much fuel, in kilograms, does a {m} minute segment take at {r} kg/h?",
        "Fuel flow is constant at {r} kg per hour for {m} minutes. Total fuel burned in kg?",
    ]
    req = phr[p % len(phr)].format(r=comma(rate), m=minutes)
    printed = fnum(float(burn))
    steps = [f"{comma(rate)} x {minutes} = {comma(prod)} kg-minutes per hour.",
             f"{comma(prod)} / 60 = {printed} kg."]
    code = _print_code("", f"{rate} * {minutes} / 60")
    return Single("fuel_burn", "Aerospace & Automation", req, float(burn), printed, "kg",
                  code, f"The engine burns {printed} kg.", steps=None,
                  wrong=(f"The engine burns {comma(rate)} x {minutes} = {comma(prod)} kg.",
                         "The output multiplies an hourly rate by minutes without converting the minutes to hours."),
                  figures=dict(r=rate, m=minutes))


def thrust_accel(rng, p):
    mass = rng.choice([500, 800, 1200, 2000])
    thrust = rng.choice([12000, 15000, 25000, 30000, 40000])
    if thrust <= mass * 9.81:
        thrust = int(mass * 9.81 * 1.5 // 1000 + 1) * 1000
    a = (thrust - mass * 9.81) / mass
    phr = [
        "A vehicle of mass {m} kg lifts off vertically with {T} N of thrust. Taking g = 9.81 m/s^2 and ignoring drag, what is its initial upward acceleration in m/s^2?",
        "Find the initial vertical acceleration of a {m} kg rocket producing {T} N thrust, with g = 9.81 m/s^2 and no drag.",
        "A {m}-kilogram drone pushes {T} newtons straight up. Using g = 9.81 m/s^2, what is its upward acceleration?",
        "With thrust {T} N acting upward on a {m} kg body and gravity 9.81 m/s^2, compute the net upward acceleration.",
        "A lander of {m} kg fires its engine for {T} N of upward thrust; g = 9.81 m/s^2, drag ignored. Acceleration in m/s^2?",
    ]
    req = phr[p % len(phr)].format(m=mass, T=comma(thrust)) + " Give the answer to at least 6 decimal places."
    code = _print_code(f"m, T, g = {mass}, {thrust}, 9.81", "(T - m * g) / m")
    printed = fnum(a)
    return Single("thrust_accel", "Aerospace & Automation", req, a, printed, "m/s^2",
                  code, f"The initial upward acceleration is {printed} m/s^2.",
                  wrong=(f"The acceleration is {comma(thrust)} / {mass} = {fnum(thrust / mass)} m/s^2.",
                         "The output ignores the vehicle's weight; gravity must be subtracted from the thrust first."),
                  figures=dict(m=mass, T=thrust))


def ledger_replay(rng, p):
    bal = rng.choice([5000, 12000, 25000, 8000])
    lines, final, ids = [], bal, []
    for i in range(rng.randint(4, 6)):
        amt = rng.choice([500, 750, 1200, 2000, 3500, 300])
        kind = rng.choice(["deposit", "withdrawal"])
        ids.append((f"T{i + 1}", kind, amt))
    rev = rng.randrange(len(ids))
    for j, (tid, kind, amt) in enumerate(ids):
        lines.append(f"{tid} {kind} {comma(amt)}")
        if j != rev:
            final += amt if kind == "deposit" else -amt
    lines.append(f"{ids[rev][0]} reversed")
    ledger = "; ".join(lines)
    phr = [
        "An account opens with {b} rupees. The ledger reads: {L}. A reversed transaction is cancelled as if it never happened. What is the closing balance in rupees?",
        "Starting balance {b} rupees. Entries in order: {L}. Reversals undo the named entry. Find the final balance.",
        "Replay these entries on an opening balance of {b} rupees: {L}. A reversal cancels its transaction. Closing balance?",
        "A wallet holds {b} rupees, then: {L}. Treat a reversed entry as void. How many rupees are left?",
        "Opening balance: {b} rupees. Activity: {L}. Reversed items do not count. What is the balance at the end?",
    ]
    req = phr[p % len(phr)].format(b=comma(bal), L=ledger)
    code_lines = [f"balance = {bal}", f"entries = {[(t, k, a) for t, k, a in ids]!r}",
                  f"reversed_ids = {{{ids[rev][0]!r}}}",
                  "for tid, kind, amt in entries:",
                  "    if tid in reversed_ids:",
                  "        continue",
                  "    balance += amt if kind == 'deposit' else -amt",
                  "print(balance)"]
    return Single("ledger_replay", "Finance & Quantitative Analysis", req, final, str(final), "rupees",
                  "\n".join(code_lines), f"The closing balance is {comma(final)} rupees.",
                  figures=dict(b=bal))


def room_overlap(rng, p):
    meetings = []
    for _ in range(rng.randint(5, 7)):
        s = rng.randint(8, 16)
        e = s + rng.randint(1, 3)
        meetings.append((s, e))
    events = sorted([(s, 1) for s, _ in meetings] + [(e, -1) for _, e in meetings])
    cur = best = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    ms = ", ".join(f"{s}:00-{e}:00" for s, e in meetings)
    phr = [
        "Meetings are booked for {ms} today. A meeting ending at the hour frees its room for one starting at that hour. What is the fewest rooms needed?",
        "Given the bookings {ms}, and that back-to-back meetings can share a room, how many rooms must be available?",
        "How many conference rooms are needed to host meetings at {ms} if a room freed at an hour can be reused at that same hour?",
        "Schedule these sessions: {ms}. Sessions touching at an endpoint do not overlap. Minimum number of rooms?",
        "Find the minimum number of rooms for the time slots {ms} (an end time equal to a start time is not a clash).",
    ]
    req = phr[p % len(phr)].format(ms=ms)
    code = (f"meetings = {meetings}\n"
            "events = sorted([(s, 1) for s, _ in meetings] + [(e, -1) for _, e in meetings])\n"
            "cur = best = 0\nfor _, d in events:\n    cur += d\n    best = max(best, cur)\nprint(best)")
    return Single("room_overlap", "Computer Science", req, best, str(best), "rooms",
                  code, f"The minimum number of rooms is {best}.", figures={})


# ---------------------------------------------------- choice families (no tool)

def compliance_choice(rng, p):
    keep = rng.choice([5, 7, 10])
    age = rng.choice([a for a in range(keep - 3, keep + 5) if a != keep])  # no boundary case
    hold = rng.random() < 0.4
    consent = rng.random() < 0.5
    rec = pick(rng, ["an invoice", "a payroll record", "a customer contract", "an audit file"])
    if hold:
        correct = "C"
    elif age >= keep:
        correct = "B"
    else:
        correct = "A"
    opts = {"A": "keep it, because the retention period has not ended and no hold applies",
            "B": "delete it, because the retention period has ended and no hold applies",
            "C": "keep it, because a litigation hold overrides the retention schedule",
            "D": "delete it only with the customer's consent"}
    hold_txt = "It is subject to an active litigation hold." if hold else "No litigation hold applies."
    cons_txt = " The customer has asked for deletion." if consent else ""
    phr = [
        "Policy: records are kept for {k} years from creation and then deleted, unless a litigation hold applies, which overrides the schedule. Customer requests do not change the schedule. {rec} was created {a} years ago. {h}{c} What should be done with it?",
        "Under a {k}-year retention rule with litigation holds taking priority (and customer requests having no effect), what happens to {rec} that is {a} years old? {h}{c}",
        "The retention schedule says {k} years, then delete; an active litigation hold suspends deletion; customer wishes do not alter the schedule. {rec} is {a} years old. {h}{c} Choose the correct action.",
        "A firm keeps documents {k} years and then destroys them unless legal places a hold; requests from customers are not a basis for early deletion. Consider {rec} created {a} years ago. {h}{c} Which action is right?",
        "Apply the rule: retain {k} years from creation; litigation hold overrides; customer requests do not change retention. Record: {rec}, age {a} years. {h}{c}",
    ]
    rec_cap = rec[0].upper() + rec[1:]
    req = phr[p % len(phr)].format(k=keep, rec=rec, a=age, h=hold_txt, c=cons_txt)
    # A sentence-initial record ("... ago. {rec} was created") starts in upper case.
    req = req.replace(". " + rec, ". " + rec_cap).replace("Consider " + rec_cap, "Consider " + rec)
    req += "\n\nOptions: " + "; ".join(f"{k}) {v}" for k, v in opts.items()) + ". Answer with the letter of the correct option."
    why = {"A": f"{age} years is under the {keep}-year period and no hold applies",
           "B": f"{age} years meets the {keep}-year period and no hold applies",
           "C": "an active litigation hold overrides the schedule, whatever the record's age"}[correct]
    steps = [f"Because {why}, the answer is {correct}."]
    return Single("compliance_choice", "Legal & Compliance Analysis", req, correct, correct, "",
                  "", f"The answer is {correct}.", steps=steps, tool_ok=False, figures={})


def comms_choice(rng, p):
    limit = rng.choice([40, 50, 60])
    date = pick(rng, ["14 March", "2 June", "30 September", "11 November"])
    good = rng.choice(["A", "B", "C", "D"])
    flaws = ["it leaves out the date", "it is over the word limit", "it never states the decision"]
    letters = [l for l in "ABCD" if l != good]
    rng.shuffle(flaws)
    desc = {good: f"states the decision, gives {date}, and is {limit - 12} words"}
    for l, f in zip(letters, flaws):
        desc[l] = f"a draft where {f}"
    phr = [
        "A notice to staff must state the decision, give the date {d}, and be at most {lim} words. Which draft qualifies?",
        "Pick the message that meets all three rules: it states the decision, it includes {d}, and it has no more than {lim} words.",
        "Which email satisfies the brief (decision stated, date {d} included, {lim} words or fewer)?",
        "The announcement has three requirements: the decision, the date {d}, and a {lim}-word maximum. Choose the draft that meets them.",
        "Select the summary that states the decision, mentions {d}, and stays within {lim} words.",
    ]
    req = phr[p % len(phr)].format(d=date, lim=limit)
    req += "\n\nOptions: " + "; ".join(f"{k}) {desc[k]}" for k in "ABCD") + ". Answer with the letter of the correct option."
    steps = [f"Only option {good} meets every rule, so the answer is {good}."]
    return Single("comms_choice", "Professional Communications", req, good, good, "",
                  "", f"The answer is {good}.", steps=steps, tool_ok=True, figures={})


FAMILIES = {
    "gamblers_ruin": gamblers_ruin, "markov_steps": markov_steps, "dice_until_sum": dice_until_sum,
    "grid_paths": grid_paths, "knapsack": knapsack, "makespan": makespan,
    "inventory_replay": inventory_replay, "loan_payment": loan_payment,
    "resistor_divider": resistor_divider, "rc_charge": rc_charge, "beam_moment": beam_moment,
    "pipe_flow": pipe_flow, "limiting_reagent": limiting_reagent, "dose_calc": dose_calc,
    "pipeline_hours": pipeline_hours, "loop_count": loop_count,
    "inclusion_exclusion": inclusion_exclusion, "fuel_burn": fuel_burn,
    "thrust_accel": thrust_accel, "ledger_replay": ledger_replay, "room_overlap": room_overlap,
    "compliance_choice": compliance_choice, "comms_choice": comms_choice,
}
