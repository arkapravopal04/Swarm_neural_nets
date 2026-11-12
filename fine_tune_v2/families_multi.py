"""
fine_tune_v2/families_multi.py -- multi-part request families
(TRAINING_SPEC.md §6), for decomposition, nesting (D4), roll-ups and the
executor rows inside them.

A scenario is a tree of Parts. A leaf Part has the executor's gold REPORT; a
decomposer Part has pieces (a three-level family nests one more decomposer).
Each Part's `text` is the request's own clause, so a child task restates its
part in the request's words and figures (TRAINING_SPEC §7.2).

A leaf REPORT's FIRST sentence carries the decision and its working inline and
fits in 140 characters: a decomposer only ever sees that much of a child's
result (Orchestrator.CHILD_EXCERPT_CHARS), so roll-ups are built from first
sentences only. Prose for parts no solver writes comes from banks (D3: no
teacher model) and contains no figures of its own.
"""
from dataclasses import dataclass, field
from typing import Optional

from families_single import comma, fnum, pick, SHOW_HELPER


@dataclass
class Part:
    label: str
    text: str                        # the request's clause for this part
    deps: list = field(default_factory=list)
    role: str = "executor"           # "executor" | "decomposer"
    report: Optional[str] = None     # leaf gold REPORT (first sentence <= 140 chars)
    pieces: list = field(default_factory=list)   # [Part] for a decomposer
    code: Optional[str] = None       # E2 program for a computed leaf
    printed: Optional[str] = None    # what that program prints
    after_tool: Optional[str] = None  # E3 REPORT using the printed value
    estimate: bool = False           # E4: the report rests on a labelled estimate
    wrong: Optional[tuple] = None    # (wrong report, critique) for E6
    keywords: tuple = ()             # requirement filter for executors

    @property
    def final_report(self):
        """The leaf's last REPORT: the direct one, or the one after its tool."""
        return self.report or self.after_tool

    @property
    def summary(self):
        """What a parent decomposer can see of this child's result: the first
        sentence of a leaf's REPORT, or of a decomposer child's own roll-up,
        prefixed by the part it covered (the YOUR SUBTASKS line names it)."""
        if self.role == "decomposer":
            head = first_sentence(rollup(self.pieces))
            rest = [p.label for p in self.pieces[1:]]
            if rest:
                # The excerpt shows only the first piece's result; the others
                # are named as done, with no figures, so none goes missing silently.
                head += f" The {' and '.join(rest)} {'is' if len(rest) == 1 else 'are'} done too."
            return head
        return first_sentence(self.final_report)


@dataclass
class Multi:
    family: str
    domain: str
    request: str
    parts: list
    requirements: list               # phaser-style constraint lines
    figures: dict = field(default_factory=dict)

    def leaves(self):
        out = []

        def walk(ps):
            for p in ps:
                if p.role == "decomposer":
                    walk(p.pieces)
                else:
                    out.append(p)
        walk(self.parts)
        return out

    def decomposers(self):
        out = []

        def walk(ps, gen):
            for p in ps:
                if p.role == "decomposer":
                    out.append((p, gen))
                    walk(p.pieces, gen + 1)
        walk(self.parts, 1)
        return out


def first_sentence(text):
    if not text:
        return None
    head = text.split(". ")[0].strip()
    return head if head.endswith(".") else head + "."


def rollup(parts):
    """A decomposer's gold roll-up: one sentence per child, each the first
    sentence of that child's result (what the excerpt shows) -- no new
    figures, nothing solved that a child did not solve."""
    return " ".join(p.summary for p in parts)


# --------------------------------------------------------------- prose banks

ILLNESS = [
    "The {leader} of the ill {member}'s group stays with them and contacts a local doctor. The other groups continue with their own {leader}s, and the family is told the same day.",
    "If a {member} falls ill, their group's {leader} takes them to the nearest clinic. The rest of that group joins another group until the {leader} returns, and the family is informed.",
    "An ill {member} rests with their {leader} while a doctor is called. The remaining groups keep to the plan under their own {leader}s, and the family is contacted at once.",
    "The ill {member} is cared for by their own {leader}, who arranges medical help. Their group merges with a neighbouring group meanwhile, and the family is called.",
    "Their {leader} stays with the ill {member} and gets medical advice; the other {leader}s carry on with their groups. The family is informed before the end of the day.",
    "A {member} who falls ill is accompanied by their {leader} to a doctor. The other groups continue, and the {leader} updates the family by phone.",
]
ROLLBACK = [
    "Keep the old database read-only until the copy is verified. If checks fail, point the application back to the old database and investigate before retrying.",
    "Take a full backup before the cutover and keep the old system running. If verification fails, switch traffic back to the old system and repeat the copy later.",
    "Freeze writes on the old database, copy, then verify row counts. If any check fails, re-enable the old database and abandon the cutover for that window.",
    "Leave the source untouched during the copy. On a failed check, redirect the application to the source and schedule a new window.",
    "Snapshot the source first. If the new database fails its checks, restore service from the source and keep the snapshot for the retry.",
    "Run the old and new databases side by side until checks pass. On failure, keep serving from the old one and roll the cutover back.",
]
NOTIFY = [
    "Email every client the new address a week before the move, and update the website and invoices on moving day.",
    "Send clients a short notice with the new address and the moving date, then update letterheads and the website.",
    "Tell clients by email and phone about the new address, and change the address on invoices and the website.",
    "Write to all clients with the new address and the date it takes effect, and update every listing that shows the old one.",
    "Notify clients of the new address in writing, and update the website, signatures and invoice template on the move date.",
    "Send each client the new address with the effective date, and correct the address on the website and billing.",
]
PACKING = [
    "Pack each desk's equipment in its own labelled crate, with the owner's name and new desk number on the label.",
    "Label every crate with its owner and destination desk, and pack cables in bags taped to their devices.",
    "Give each person a crate labelled with their name and new desk, and keep shared equipment in crates marked by team.",
    "Pack and label equipment per desk, listing the contents on each crate so it can be checked on arrival.",
    "Use one labelled crate per person for their equipment, marked with the new desk, and pack shared items separately.",
    "Pack equipment desk by desk into crates labelled with owner and new location, and list fragile items on top.",
]
NETWORK = [
    "Check that every desk has a working network port, the wireless covers the whole floor, and the printer is reachable.",
    "Test each desk's wired connection, confirm wireless coverage, and verify printers and shared drives before people arrive.",
    "Confirm internet access, label each desk's network port, and test the printer and video room on the first morning.",
    "Verify the router and switches, test every port, and check that the wireless reaches meeting rooms.",
    "Make a checklist: internet line live, ports tested at every desk, wireless coverage checked, printer and shared drives working.",
    "Before moving day, test the connection, every desk port and the wireless, and make sure shared printers respond.",
]
PRESS = [
    "The press note announces the product, its price and launch date, and gives one quote and a contact for media.",
    "Write a one-page note announcing the launch, the product's main use, its price, and a media contact.",
    "The note states what the product does, its price and when it goes on sale, with a short quote and contact details.",
    "Draft a short press note: the product, the price, the launch date, one customer-facing benefit, and a contact.",
    "Announce the launch in a brief note with the product's purpose, price, availability date and a press contact.",
    "A short release covers the product, its price, the launch date and who to contact for interviews.",
]
FLYER = [
    "The flyer shows the product name, one image, the price and the launch date, with the venue at the bottom.",
    "Design a single-page flyer with the product, its price, the event date and venue, and a clear headline.",
    "The flyer carries a headline, a product photo, the price and the launch details in large type.",
    "Make a flyer with the product name, a picture, the price and where and when the launch happens.",
    "Lay out a flyer with one image, the product name, the price and the launch date and place.",
    "A one-page flyer gives the headline, a product image, the price and the launch event details.",
]
ROTA = [
    "Rotate the families through the days in a fixed order so each family waters on the same weekday every week.",
    "Assign families to days in turn and post the rota at the gate, swapping only by agreement.",
    "List the families in order and give each pair of families a day, repeating the order each week.",
    "Pair the families and give each pair a fixed watering day, kept the same every week.",
    "Set a weekly rota in which each family's day stays fixed, with swaps noted on the board.",
    "Put the families in a repeating order so every family waters on a regular day each week.",
]

WELCOME = [
    "Welcome everyone. Badges are at the front desk, sessions start on time in the rooms on your badge, and coffee is served between sessions.",
    "Thank you for joining us. Please collect your badge at registration, check your room on the schedule, and enjoy coffee in the breaks.",
    "A warm welcome to all attendees and speakers. Your badge shows your room; please arrive a few minutes early for each session.",
    "We are glad you are here. Pick up your badge, find your room on the board, and join us for coffee between sessions.",
    "Welcome to the event. Registration has your badge, the schedule lists each room, and coffee is available throughout the day.",
    "Hello and welcome. Collect your badge, follow the signs to your room, and use the breaks to meet the speakers.",
]
DOWNTIME = [
    "Machines will be out of service one at a time during their scheduled slot; please plan work around the posted timetable.",
    "Service runs this month. Each machine is down only during its slot on the board, so check the timetable before planning runs.",
    "Planned maintenance is under way. Expect each machine to be unavailable during its posted slot, and report problems to the technician on duty.",
    "Machines are being serviced in turn. The timetable at the entrance shows when each one is down.",
    "Each machine will be briefly offline for service. See the posted schedule and avoid booking work in those slots.",
    "Scheduled servicing is taking place. Please check which machines are down each day on the notice board.",
]

GROUP_TYPES = [
    ("choir", "singers", "singer", "conductors", "conductor"),
    ("football club", "players", "player", "coaches", "coach"),
    ("scout troop", "scouts", "scout", "leaders", "leader"),
    ("drama society", "actors", "actor", "directors", "director"),
    ("college", "students", "student", "lecturers", "lecturer"),
    ("swimming club", "swimmers", "swimmer", "coaches", "coach"),
]
TRANSPORT = [("coach", "train"), ("minibus", "train"), ("bus", "ferry"), ("hired van", "rail")]


def _requirement_lines(pairs):
    return [f"{label} = {value}." for label, value in pairs]


# ------------------------------------------------------------------- families

def trip_budget(rng, p):
    org, members, member, leaders, leader = pick(rng, GROUP_TYPES)
    lead_n = rng.randint(3, 6)
    per_group = rng.randint(5, 10)
    mem_n = lead_n * per_group
    people = mem_n + lead_n
    days = rng.randint(2, 4)
    flat_t, pp_t = pick(rng, [t for t in TRANSPORT if leader not in t])
    pp = rng.choice([700, 800, 900, 950, 1100, 1200])
    flat = rng.choice([25000, 30000, 36000, 42000, 48000, 55000])
    per_cost = pp * people
    if per_cost == flat:
        flat += 1000
    cheaper, dearer = ((flat_t, flat), (pp_t, per_cost)) if flat < per_cost else ((pp_t, per_cost), (flat_t, flat))
    diff = abs(per_cost - flat)
    cap = rng.choice([10000, 12000, 15000, 18000, 20000])
    est = max(100, (cap // people) // 10 * 10 - 20)
    day_cost = est * people
    assert day_cost < cap, (day_cost, cap)
    meals_total = day_cost * days
    budget = rng.choice([120000, 150000, 160000, 200000, 220000])
    intros = [
        "A {org} is taking {m} {members} and {l} {leaders} on a {d}-day trip with a budget of {b} rupees.",
        "A {d}-day trip is planned for a {org}: {m} {members} and {l} {leaders}, with {b} rupees to spend.",
        "Our {org} has {b} rupees for a {d}-day trip with {m} {members} and {l} {leaders}.",
        "For a {d}-day outing, a {org} of {m} {members} and {l} {leaders} has a total budget of {b} rupees.",
        "A {org} with {m} {members} and {l} {leaders} is going away for {d} days on a budget of {b} rupees.",
    ]
    t_txt = f"choose between a {flat_t} that costs {comma(flat)} rupees in total and a {pp_t} at {comma(pp)} rupees per person"
    m_txt = f"plan meals so that each day stays under {comma(cap)} rupees"
    g_txt = f"split the {members} into groups with one {leader} each"
    i_txt = f"decide what to do if a {member} falls ill during the trip"
    request = intros[p % len(intros)].format(org=org, m=mem_n, members=members, l=lead_n,
                                             leaders=leaders, d=days, b=comma(budget))
    request += f" {t_txt[0].upper() + t_txt[1:]}, {m_txt}, {g_txt}, and {i_txt}."
    transport = Part("transport", t_txt[0].upper() + t_txt[1:] + ".", keywords=("transport", flat_t, pp_t),
                     report=(f"Take the {cheaper[0]}: {comma(cheaper[1])} rupees against {comma(dearer[1])} for the "
                             f"{dearer[0]}. The {pp_t} is {comma(pp)} x {people} = {comma(per_cost)} rupees for all "
                             f"{people} travellers, so the {cheaper[0]} saves {comma(diff)} rupees."),
                     wrong=(f"Take the {pp_t if pp * mem_n < flat else flat_t}: the {pp_t} is {comma(pp)} x {mem_n} = "
                            f"{comma(pp * mem_n)} rupees.",
                            f"The output counts only the {mem_n} {members}; the {lead_n} {leaders} travel too, "
                            f"so the {pp_t} is priced for {people} people."))
    meals = Part("meals", m_txt[0].upper() + m_txt[1:] + ".", estimate=True, keywords=("meal", "day"),
                 report=(f"Meals at about {est} rupees per person per day (estimated) cost {est} x {people} = "
                         f"{comma(day_cost)} rupees a day. That is under the {comma(cap)} rupee daily limit, and over "
                         f"{days} days meals come to {comma(meals_total)} rupees."))
    groups = Part("groups", g_txt[0].upper() + g_txt[1:] + ".", keywords=("group",),
                  report=(f"Make {lead_n} groups of {per_group} {members}, one {leader} each: {mem_n} / {lead_n} = "
                          f"{per_group}. Every {member} is in exactly one group."),
                  wrong=(f"Make {per_group} groups of {lead_n} {members}.",
                         f"The output swaps the numbers: {lead_n} {leaders} means {lead_n} groups, of {per_group} each."))
    illness = Part("illness", i_txt[0].upper() + i_txt[1:] + ".", keywords=("ill",),
                   report=pick(rng, ILLNESS).format(leader=leader, member=member))
    reqs = _requirement_lines([("Total budget", f"{comma(budget)} rupees"),
                               (f"{flat_t.capitalize()} cost", f"{comma(flat)} rupees total"),
                               (f"{pp_t.capitalize()} cost", f"{comma(pp)} rupees per person"),
                               ("Daily meal cost", f"under {comma(cap)} rupees"),
                               ("Group leaders", f"one {leader} per group")])
    return Multi("trip_budget", "General Discourse", request, [transport, meals, groups, illness], reqs,
                 figures=dict(people=people, budget=budget))


def event_terminal(rng, p):
    guests = rng.choice([60, 80, 120, 150, 200])
    hall_flat = rng.choice([30000, 40000, 45000, 60000])
    hall_pp = rng.choice([250, 300, 350, 400])
    pp_total = hall_pp * guests
    venue = min(hall_flat, pp_total)
    venue_name = "the flat-fee hall" if hall_flat <= pp_total else "the per-guest hall"
    cater_pp = rng.choice([400, 450, 500, 600])
    cater = cater_pp * guests
    decor = rng.choice([5000, 7500, 10000, 12000])
    total = venue + cater + decor
    budget = total + rng.choice([-8000, 5000, 12000, 20000])
    within = total <= budget
    evt = pick(rng, ["annual dinner", "alumni reunion", "awards night", "charity gala", "team celebration"])
    intros = [
        "We are organising {art} {e} for {g} guests with a budget of {b} rupees.",
        "{Art} {e} for {g} guests must be planned within {b} rupees.",
        "{Art} {e} with {g} guests has a budget of {b} rupees.",
        "Our {e} will host {g} guests and we have {b} rupees.",
        "For {art} {e} of {g} guests, the total budget is {b} rupees.",
    ]
    v_txt = (f"choose between a hall at {comma(hall_flat)} rupees flat and a hall at {hall_pp} rupees per guest")
    c_txt = f"cost the catering at {cater_pp} rupees per guest"
    d_txt = f"include decorations at {comma(decor)} rupees"
    k_txt = "check that the total stays within the budget"
    art = "an" if evt[0] in "aeiou" else "a"
    request = intros[p % len(intros)].format(e=evt, g=guests, b=comma(budget), art=art, Art=art.capitalize())
    request += f" {v_txt[0].upper() + v_txt[1:]}, {c_txt}, {d_txt}, and {k_txt}."
    vp = Part("venue", v_txt[0].upper() + v_txt[1:] + ".", keywords=("hall", "venue"),
              report=(f"Choose {venue_name}: {comma(venue)} rupees, since {hall_pp} x {guests} = {comma(pp_total)} "
                      f"against {comma(hall_flat)} flat."))
    cp = Part("catering", c_txt[0].upper() + c_txt[1:] + ".", keywords=("catering",),
              report=f"Catering costs {cater_pp} x {guests} = {comma(cater)} rupees.")
    dp = Part("decor", d_txt[0].upper() + d_txt[1:] + ".", keywords=("decoration",),
              report=f"Decorations cost {comma(decor)} rupees, as given.")
    verdict = (f"within the {comma(budget)} budget by {comma(budget - total)}" if within
               else f"over the {comma(budget)} budget by {comma(total - budget)}")
    kp = Part("check", k_txt[0].upper() + k_txt[1:] + ".", deps=["venue", "catering", "decor"],
              keywords=("budget", "total"),
              report=(f"The total is {comma(venue)} + {comma(cater)} = {comma(venue + cater)}, and "
                      f"{comma(venue + cater)} + {comma(decor)} = {comma(total)} rupees, {verdict} rupees."))
    reqs = _requirement_lines([("Total budget", f"{comma(budget)} rupees"), ("Guests", str(guests)),
                               ("Catering", f"{cater_pp} rupees per guest"), ("Decorations", f"{comma(decor)} rupees")])
    return Multi("event_terminal", "Finance & Quantitative Analysis", request, [vp, cp, dp, kp], reqs,
                 figures=dict(total=total))


def staged_circuit(rng, p):
    v = 230
    r = rng.choice([23, 46, 115, 230])
    i = v // r
    pw = v * i
    hours = rng.choice([2, 3, 4, 6, 8])
    rate = rng.choice([6, 7, 8, 9])
    cost = pw * hours / 1000 * rate
    dev = pick(rng, ["heater", "kiln", "water boiler", "oven", "dryer"])
    intros = [
        "A {dev} has a resistance of {r} ohms and runs from a {v} V supply for {h} hours a day; electricity costs {c} rupees per kWh.",
        "Our {dev} ({r} ohms) is connected to {v} volts and used {h} hours daily, with power priced at {c} rupees per kWh.",
        "A {r}-ohm {dev} operates on {v} V for {h} hours each day at an electricity price of {c} rupees per kWh.",
        "Consider a {dev} of {r} ohms on a {v} V mains supply, used for {h} hours a day, electricity {c} rupees per kWh.",
        "A {dev} with {r} ohms resistance runs {h} hours a day from {v} V; each kWh costs {c} rupees.",
    ]
    request = intros[p % len(intros)].format(dev=dev, r=r, v=v, h=hours, c=rate)
    request += " Find the current it draws, then its power, then its daily electricity cost."
    cur = Part("current", f"Find the current the {dev} draws.", keywords=("current",),
               report=f"The current is {v} / {r} = {i} A.")
    power = Part("power", f"Find the power of the {dev}.", deps=["current"], keywords=("power",),
                 report=f"The power is {v} x {i} = {comma(pw)} W, using the {i} A current.")
    code = SHOW_HELPER + f"power_w = {pw}\nhours = {hours}\nrate = {rate}\nshow(round(power_w * hours / 1000 * rate, 8))"
    printed = fnum(cost)
    costp = Part("cost", f"Find the daily electricity cost of the {dev}.", deps=["power"], keywords=("cost", "electricity"),
                 code=code, printed=printed,
                 after_tool=(f"The daily cost is {printed} rupees: {comma(pw)} W for {hours} hours at {rate} rupees "
                             f"per kWh."),
                 report=None)
    reqs = _requirement_lines([("Supply voltage", f"{v} V"), ("Resistance", f"{r} ohms"),
                               ("Daily use", f"{hours} hours"), ("Electricity price", f"{rate} rupees per kWh")])
    return Multi("staged_circuit", "Electrical & Computer Engineering", request, [cur, power, costp], reqs,
                 figures=dict(i=i, pw=pw))


def office_move(rng, p):
    n = rng.choice([18, 24, 30, 36, 42, 50])
    k = rng.choice([4, 6, 8])
    tables = -(-n // k)
    team = pick(rng, ["design team", "finance team", "support team", "research group", "sales team"])
    intros = [
        "Our {t} of {n} people is moving to a new office.",
        "A {t} with {n} staff is relocating to new premises.",
        "We are moving the {n}-person {t} to a new office.",
        "The {t} ({n} people) moves to a new office next month.",
        "A new office is ready for our {t} of {n} people.",
    ]
    p_txt = "pack and label the equipment"
    s_txt = f"set up the new office by drawing a seating plan for {n} people at tables of {k} and preparing a network checklist"
    c_txt = "send clients the new address"
    request = intros[p % len(intros)].format(t=team, n=n) + f" Please {p_txt}, {s_txt}, and {c_txt}."
    seat = Part("seating", f"Draw a seating plan for {n} people at tables of {k}.", keywords=("seat", "table"),
                report=(f"Use {tables} tables of {k}: {n} / {k} rounds up to {tables}. "
                        f"Seat each team together and leave the spare seats at the last table."
                        if n % k else
                        f"Use {tables} tables of {k}: {n} / {k} = {tables}. Seat each team together; "
                        f"every table is full."))
    net = Part("network", "Prepare a network checklist for the new office.", keywords=("network",),
               report=pick(rng, NETWORK))
    setup = Part("setup", s_txt[0].upper() + s_txt[1:] + ".", role="decomposer", pieces=[seat, net],
                 keywords=("office", "seat", "network"))
    pack = Part("packing", "Pack and label the equipment.", keywords=("pack", "label"), report=pick(rng, PACKING))
    notify = Part("clients", "Send clients the new address.", keywords=("client", "address"), report=pick(rng, NOTIFY))
    reqs = _requirement_lines([("Team size", f"{n} people"), ("Table size", f"{k} people per table")])
    return Multi("office_move", "Professional Communications", request, [pack, setup, notify], reqs,
                 figures=dict(n=n, k=k))


def product_launch(rng, p):
    cost = rng.choice([400, 600, 800, 1200, 1500])
    margin = rng.choice([20, 25, 30, 40, 50])
    price = cost * (100 + margin) // 100
    guests = rng.choice([50, 80, 100, 120])
    flat = rng.choice([20000, 25000, 30000])
    per = rng.choice([200, 250, 300])
    per_total = per * guests
    venue_cost = min(flat, per_total)
    which = "the flat-fee venue" if flat <= per_total else "the per-guest venue"
    staff = rng.choice([12, 16, 20, 24, 30])
    gsize = rng.choice([4, 5, 6])
    groups = -(-staff // gsize)
    prod = pick(rng, ["water filter", "desk lamp", "bike lock", "travel mug", "garden sensor"])
    intros = [
        "We are launching a new {p}.",
        "Our company is bringing a new {p} to market.",
        "A new {p} goes on sale next quarter.",
        "The launch of our new {p} is being planned.",
        "The team is preparing to launch a {p}.",
    ]
    pr_txt = f"set the price at the unit cost of {comma(cost)} rupees plus a {margin}% margin"
    ev_txt = (f"run the launch event by booking a venue for {guests} guests (a venue at {comma(flat)} rupees flat or one "
              f"at {per} rupees per guest) and preparing the materials, a press note and a flyer")
    tr_txt = f"train the {staff}-person sales team in groups of {gsize}"
    request = intros[p % len(intros)].format(p=prod) + f" Please {pr_txt}, {ev_txt}, and {tr_txt}."
    price_p = Part("price", pr_txt[0].upper() + pr_txt[1:] + ".", keywords=("price", "margin"),
                   code=SHOW_HELPER + f"cost = {cost}\nmargin = {margin}\nshow(round(cost * (100 + margin) / 100, 8))",
                   printed=fnum(price),
                   after_tool=(f"The price is {fnum(price)} rupees: the unit cost of {comma(cost)} rupees with "
                               f"a {margin}% margin."))
    venue = Part("venue", f"Book a venue for {guests} guests: one at {comma(flat)} rupees flat or one at {per} rupees per guest.",
                 keywords=("venue",),
                 report=(f"Book {which}: {comma(venue_cost)} rupees, since {per} x {guests} = {comma(per_total)} "
                         f"against {comma(flat)} flat."))
    press = Part("press", "Write the press note for the launch.", keywords=("press",), report=pick(rng, PRESS))
    flyer = Part("flyer", "Design the flyer for the launch.", keywords=("flyer",), report=pick(rng, FLYER))
    materials = Part("materials", "Prepare the launch materials: a press note and a flyer.", role="decomposer",
                     pieces=[press, flyer], keywords=("material", "press", "flyer"))
    event = Part("event", ev_txt[0].upper() + ev_txt[1:] + ".", role="decomposer", pieces=[venue, materials],
                 keywords=("event", "venue"))
    train = Part("training", tr_txt[0].upper() + tr_txt[1:] + ".", keywords=("train", "sales"),
                 report=(f"Run {groups} training groups: {staff} / {gsize} "
                         + (f"rounds up to {groups}. " if staff % gsize else f"= {groups}. ") +
                         f"Each group gets the same session on the product and its price."))
    reqs = _requirement_lines([("Unit cost", f"{comma(cost)} rupees"), ("Margin", f"{margin}%"),
                               ("Guests", str(guests)), ("Sales team", f"{staff} people in groups of {gsize}")])
    return Multi("product_launch", "Professional Communications", request, [price_p, event, train], reqs,
                 figures=dict(price=price))


def data_migration(rng, p):
    records = rng.choice([7200000, 10800000, 14400000, 21600000])
    rate = rng.choice([500, 1000, 2000])
    hours = float(fnum(records / rate / 3600))
    secs = records // rate
    need = hours + 1
    w1, w2 = rng.choice([4, 6, 8]), rng.choice([10, 12, 14])
    choice = "the Saturday window" if need <= w1 else ("the Sunday window" if need <= w2 else None)
    intros = [
        "We must migrate a database of {N} records, copied at {r} records per second.",
        "A {N}-record database is being moved to a new server at a copy rate of {r} records per second.",
        "Our database holds {N} records and the copy tool moves {r} records per second.",
        "The migration covers {N} records; the copy runs at {r} records per second.",
        "The migration copies {N} records at a steady {r} records per second.",
    ]
    e_txt = "estimate the copy time in hours"
    c_txt = (f"choose the shorter cutover window (Saturday, {w1} hours, or Sunday, {w2} hours) that fits the "
             f"copy time plus a 1-hour check")
    r_txt = "write a rollback plan"
    request = intros[p % len(intros)].format(N=comma(records), r=comma(rate)) + f" Please {e_txt}, {c_txt}, and {r_txt}."
    est = Part("estimate", "Estimate the copy time in hours.", keywords=("copy", "time"),
               code=SHOW_HELPER + f"records = {records}\nrate = {rate}\nshow(round(records / rate / 3600, 8))",
               printed=fnum(records / rate / 3600),
               after_tool=(f"The copy takes {fnum(records / rate / 3600)} hours for {comma(records)} records at "
                           f"{comma(rate)} records per second."))
    pick_txt = (f"Use {choice}: the {fnum(hours)}-hour copy plus the 1-hour check needs {fnum(need)} hours."
                if choice else
                f"Neither window fits: the {fnum(hours)}-hour copy plus the 1-hour check needs {fnum(need)} hours, "
                f"more than {w2}.")
    win = Part("window", c_txt[0].upper() + c_txt[1:] + ".", deps=["estimate"], keywords=("window", "cutover"),
               report=pick_txt)
    rb = Part("rollback", "Write a rollback plan.", keywords=("rollback",), report=pick(rng, ROLLBACK))
    reqs = _requirement_lines([("Records", comma(records)), ("Copy rate", f"{comma(rate)} records per second"),
                               ("Saturday window", f"{w1} hours"), ("Sunday window", f"{w2} hours")])
    return Multi("data_migration", "Data Engineering", request, [est, win, rb], reqs, figures=dict(hours=hours))


def lab_study(rng, p):
    groups = rng.choice([2, 3, 4])
    per = rng.choice([10, 12, 15, 20])
    people = groups * per
    samples = rng.choice([2, 3, 4])
    price = rng.choice([150, 200, 250, 400])
    total_samples = people * samples
    cost = total_samples * price
    per_day = rng.choice([20, 30, 40, 60])
    days = -(-total_samples // per_day)
    intros = [
        "A lab study enrols {n} participants split equally into {g} groups.",
        "Our study has {n} participants in {g} equal groups.",
        "A trial of {n} participants is divided evenly into {g} groups.",
        "We are running a study with {n} participants across {g} equal groups.",
        "{n} participants join a lab study in {g} groups of equal size.",
    ]
    request = intros[p % len(intros)].format(n=people, g=groups)
    request += (f" Each participant gives {samples} samples, each sample costs {price} rupees to process, and the lab "
                f"processes {per_day} samples a day. Find the group size, the total processing cost, and the number of "
                f"days processing takes.")
    gp = Part("size", "Find the group size.", keywords=("group",),
              report=f"Each group has {people} / {groups} = {per} participants.")
    cp = Part("cost", "Find the total processing cost.", keywords=("cost",),
              report=(f"Processing costs {comma(cost)} rupees: {people} x {samples} = {total_samples} samples at "
                      f"{price} rupees each."))
    dp = Part("days", "Find the number of days processing takes.", keywords=("day",),
              report=(f"Processing takes {days} day{'s' if days != 1 else ''}: {total_samples} samples at "
                      f"{per_day} a day" + (", rounded up to whole days." if total_samples % per_day else ".")))
    reqs = _requirement_lines([("Participants", str(people)), ("Groups", str(groups)),
                               ("Samples per participant", str(samples)), ("Lab capacity", f"{per_day} samples per day")])
    return Multi("lab_study", "Biomedical & Life Sciences", request, [gp, cp, dp], reqs, figures=dict(cost=cost))


def garden_plan(rng, p):
    plot = rng.choice([10, 12, 15, 20])
    plots = rng.choice([12, 18, 24, 30])
    area = plot * plots
    fam = rng.choice([3, 4, 6])
    while plots % fam:
        fam -= 1
    each = plots // fam
    water = rng.choice([50, 80, 100, 120])
    monthly = plots * water
    intros = [
        "A community garden of {a} square metres is divided into plots of {p} square metres.",
        "We have {a} square metres of garden to split into {p}-square-metre plots.",
        "A {a} square metre garden will be laid out in plots of {p} square metres each.",
        "Our shared garden covers {a} square metres, cut into plots of {p} square metres.",
        "The garden is {a} square metres and each plot is {p} square metres.",
    ]
    request = intros[p % len(intros)].format(a=area, p=plot)
    request += (f" {fam} families share the plots equally and water costs {water} rupees per plot per month. "
                f"Find the number of plots, how many plots each family gets, the monthly water cost, and set a "
                f"watering rota.")
    np_ = Part("plots", "Find the number of plots.", keywords=("plot",),
               report=f"There are {area} / {plot} = {plots} plots.")
    ep = Part("share", "Find how many plots each family gets.", deps=["plots"], keywords=("family", "plot"),
              report=f"Each family gets {plots} / {fam} = {each} plots, using the {plots} plots.")
    wp = Part("water", "Find the monthly water cost.", deps=["plots"], keywords=("water",),
              report=f"Water costs {plots} x {water} = {comma(monthly)} rupees a month.")
    rp = Part("rota", "Set a watering rota.", keywords=("rota", "water"), report=pick(rng, ROTA))
    reqs = _requirement_lines([("Garden area", f"{area} square metres"), ("Plot size", f"{plot} square metres"),
                               ("Families", str(fam)), ("Water", f"{water} rupees per plot per month")])
    return Multi("garden_plan", "General Discourse", request, [np_, ep, wp, rp], reqs, figures=dict(plots=plots))


COFFEE_EST = [100, 120, 150]


def conference_logistics(rng, p):
    attendees = rng.choice([120, 150, 180, 240, 300])
    speakers = rng.choice([6, 8, 10, 12])
    rooms = rng.choice([3, 4, 5, 6])
    while attendees % rooms:
        rooms -= 1
    per_room = attendees // rooms
    badges = attendees + speakers
    days = rng.choice([1, 2, 3])
    cup = rng.choice(COFFEE_EST)
    coffee_cap = rng.choice([30000, 40000, 50000, 60000])
    est_cups = 2
    coffee_day = cup * est_cups * badges
    while coffee_day >= coffee_cap:
        cup -= 10
        coffee_day = cup * est_cups * badges
    name = pick(rng, ["data summit", "design forum", "health workshop", "teachers' conference", "robotics meetup"])
    intros = [
        "A {d}-day {n} expects {a} attendees and {s} speakers, with sessions split across {r} rooms.",
        "Our {n} runs for {d} days with {a} attendees, {s} speakers and {r} session rooms.",
        "The {n} ({d} days) will host {a} attendees and {s} speakers in {r} rooms.",
        "For the {d}-day {n} there are {a} attendees, {s} speakers and {r} rooms.",
        "{a} attendees and {s} speakers are coming to a {d}-day {n} that uses {r} rooms.",
    ]
    request = intros[p % len(intros)].format(d=days, n=name, a=attendees, s=speakers, r=rooms)
    request += (f" Please count the name badges needed for everyone, split the attendees equally across the rooms, "
                f"budget coffee so each day stays under {comma(coffee_cap)} rupees, and write the welcome message.")
    bp = Part("badges", "Count the name badges needed for everyone.", keywords=("badge",),
              report=f"Print {badges} badges: {attendees} attendees + {speakers} speakers = {badges}.")
    rp = Part("rooms", "Split the attendees equally across the rooms.", keywords=("room",),
              report=f"Put {per_room} attendees in each room: {attendees} / {rooms} = {per_room}.")
    cp = Part("coffee", f"Budget coffee so each day stays under {comma(coffee_cap)} rupees.", estimate=True,
              keywords=("coffee",),
              report=(f"Coffee costs about {comma(coffee_day)} rupees a day, under the {comma(coffee_cap)} limit. "
                      f"That assumes about {cup} rupees a cup (estimated) and 2 cups each: {cup} x 2 = {cup * 2} per "
                      f"person, and {cup * 2} x {badges} = {comma(coffee_day)}."))
    wp = Part("welcome", "Write the welcome message.", keywords=("welcome",), report=pick(rng, WELCOME))
    reqs = _requirement_lines([("Attendees", str(attendees)), ("Speakers", str(speakers)),
                               ("Rooms", str(rooms)), ("Daily coffee cost", f"under {comma(coffee_cap)} rupees")])
    return Multi("conference_logistics", "Professional Communications", request, [bp, rp, cp, wp], reqs,
                 figures=dict(badges=badges))


def maintenance_plan(rng, p):
    machines = rng.choice([12, 16, 20, 24, 30])
    techs = rng.choice([2, 3, 4])
    while machines % techs:
        techs -= 1
    each = machines // techs
    part_cost = rng.choice([1500, 2200, 3000, 4500])
    parts_total = machines * part_cost
    plant = pick(rng, ["bottling plant", "print shop", "textile mill", "packing line", "workshop"])
    intros = [
        "Our {pl} has {m} machines due for service and {t} technicians.",
        "A {pl} must service {m} machines this month with {t} technicians.",
        "{m} machines in the {pl} need their quarterly service; {t} technicians are available.",
        "The {pl} has {t} technicians to service its {m} machines.",
        "Service is due on {m} machines at the {pl}, shared by {t} technicians.",
    ]
    request = intros[p % len(intros)].format(pl=plant, m=machines, t=techs)
    request += (f" Each machine needs one spare part at {comma(part_cost)} rupees. Please split the machines equally "
                f"among the technicians, cost the spare parts, and write the downtime notice for staff.")
    sp = Part("split", "Split the machines equally among the technicians.", keywords=("technician",),
              report=f"Give each technician {each} machines: {machines} / {techs} = {each}.")
    cp = Part("parts", "Cost the spare parts.", keywords=("part",),
              report=f"Spare parts cost {machines} x {comma(part_cost)} = {comma(parts_total)} rupees.")
    dp = Part("notice", "Write the downtime notice for staff.", keywords=("notice", "downtime"),
              report=pick(rng, DOWNTIME))
    reqs = _requirement_lines([("Machines", str(machines)), ("Technicians", str(techs)),
                               ("Spare part", f"{comma(part_cost)} rupees per machine")])
    return Multi("maintenance_plan", "Mechanical Engineering", request, [sp, cp, dp], reqs,
                 figures=dict(parts_total=parts_total))


STALL_NOTES = [
    "Give every vendor a numbered pitch, a contact name and an arrival time, and send the list a week ahead.",
    "List each vendor with their pitch number, what they sell and their set-up time.",
    "Make a vendor sheet with names, pitch numbers and phone numbers, and share it with the gate team.",
    "Record each vendor's pitch, goods and arrival slot, and post the list at the entrance.",
    "Assign pitch numbers to vendors, note their goods, and confirm each one by phone.",
    "Keep a vendor list with pitch, goods and contact, and send it to every stallholder.",
]
SAFETY = [
    "Mark first-aid points at both ends of the ground, keep the exits clear, and brief volunteers on whom to call.",
    "Set up a first-aid tent near the entrance, keep walkways and exits open, and give stewards a radio each.",
    "Place first aid by the main path, mark the exits, and run a short briefing for volunteers before opening.",
    "Keep two clear exits, put a first-aid table at the gate, and make sure every steward knows the emergency number.",
    "Assign a first-aider, mark exits and keep them open, and tell volunteers where to send anyone who needs help.",
    "Plan first-aid cover, clear exit routes and a steward briefing on lost children and emergencies.",
]


def festival_plan(rng, p):
    stalls = rng.choice([24, 30, 36, 40, 48])
    rows = rng.choice([3, 4, 6])
    while stalls % rows:
        rows -= 1
    per_row = stalls // rows
    fee = rng.choice([500, 750, 1000, 1200])
    income = stalls * fee
    name = pick(rng, ["spring fair", "food festival", "craft market", "harvest fete", "book fair"])
    intros = [
        "Our {n} will have {s} stalls laid out in {r} rows.",
        "The {n} plans {s} stalls arranged in {r} rows.",
        "{s} stalls in {r} rows are planned for the {n}.",
        "A {n} is being organised with {s} stalls set out in {r} rows.",
        "The {n} has room for {s} stalls in {r} rows.",
    ]
    request = intros[p % len(intros)].format(n=name, s=stalls, r=rows)
    request += (f" Each stall pays {fee} rupees. Please organise the stalls by planning the layout of the rows and "
                f"preparing the vendor list, plan the safety cover, and work out the stall income.")
    layout = Part("layout", f"Plan the layout of {stalls} stalls in {rows} rows.", keywords=("stall", "row"),
                  report=f"Put {per_row} stalls in each row: {stalls} / {rows} = {per_row}.")
    vendors = Part("vendors", "Prepare the vendor list.", keywords=("vendor",), report=pick(rng, STALL_NOTES))
    org = Part("stalls", "Organise the stalls by planning the layout of the rows and preparing the vendor list.",
               role="decomposer", pieces=[layout, vendors], keywords=("stall", "vendor", "row"))
    safety = Part("safety", "Plan the safety cover.", keywords=("safety",), report=pick(rng, SAFETY))
    money = Part("income", "Work out the stall income.", keywords=("income", "pay"),
                 report=f"Stall income is {stalls} x {fee} = {comma(income)} rupees.")
    reqs = _requirement_lines([("Stalls", str(stalls)), ("Rows", str(rows)), ("Stall fee", f"{fee} rupees")])
    return Multi("festival_plan", "General Discourse", request, [org, safety, money], reqs,
                 figures=dict(income=income))


FAMILIES = {
    "trip_budget": trip_budget, "event_terminal": event_terminal, "staged_circuit": staged_circuit,
    "office_move": office_move, "product_launch": product_launch, "data_migration": data_migration,
    "lab_study": lab_study, "garden_plan": garden_plan,
    "conference_logistics": conference_logistics, "maintenance_plan": maintenance_plan,
    "festival_plan": festival_plan,
}
