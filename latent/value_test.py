"""Value test: A works out a value in its notes, B must reuse it.

Arms: note400 / text_full (B reads text), mem_full (B reads A's whole memory),
and last / writer / reader cuts of A's memory. See README.md, "Value test".
"""
import json
import os
import random
import re
import time
from contextlib import nullcontext

# ---------------------------------------------------------------- synthetic A texts

def _f(rng, lo, hi, dec):
    return f"{rng.uniform(lo, hi):.{dec}f}"


# Distractor paragraphs. None of these use a target name below.
BANK = [
    lambda r: (lambda q, d: f"Taking the flow rate as {q} L/s and the pipe bore as {d} mm, the mean velocity works out to about {float(q) / 1000 / (3.14159 * (float(d) / 2000) ** 2):.2f} m/s, which sits comfortably inside the range we wanted for the supply line. I checked the units twice because the vendor sheet mixes litres and cubic metres.")(_f(r, 1.5, 6.0, 1), _f(r, 30, 80, 0)),
    lambda r: (lambda v, L: f"For the Reynolds number I used a kinematic viscosity of 1.0e-6 m2/s, a velocity of {v} m/s and a characteristic length of {L} m. That gives roughly {float(v) * float(L) / 1e-6:,.0f}, so the flow is clearly turbulent and the laminar correlation is out.")(_f(r, 0.8, 3.0, 2), _f(r, 0.03, 0.09, 3)),
    lambda r: (lambda m, dt: f"The heat load on the jacket follows from a mass flow of {m} kg/s, a specific heat of 4.18 kJ/kgK and a temperature rise of {dt} K, so about {float(m) * 4.18 * float(dt):.1f} kW has to be rejected by the chiller. That is below the nameplate rating with some margin left.")(_f(r, 0.2, 1.5, 2), _f(r, 4, 15, 1)),
    lambda r: (lambda P, L: f"Beam deflection at midspan: point load {P} kN, span {L} m, E = 200 GPa and I = 8.36e-6 m4. Using PL^3/48EI the deflection comes to about {float(P) * 1000 * float(L) ** 3 / (48 * 200e9 * 8.36e-6) * 1000:.2f} mm, well under the span/360 limit.")(_f(r, 5, 25, 1), _f(r, 2.0, 4.5, 2)),
    lambda r: (lambda p, n: f"The motor torque was estimated from a shaft power of {p} kW at {n} rpm. Torque is power over angular speed, which gives about {float(p) * 1000 / (float(n) * 2 * 3.14159 / 60):.1f} N m at the coupling. The gearbox service factor of 1.4 still leaves headroom.")(_f(r, 3, 30, 1), _f(r, 900, 1800, 0)),
    lambda r: (lambda V, I: f"On the electrical side, the supply voltage is {V} V and the measured line current is {I} A. The apparent power is therefore about {float(V) * float(I) * 1.732 / 1000:.2f} kVA for the three-phase feed, which the existing breaker can carry.")(_f(r, 380, 415, 0), _f(r, 8, 40, 1)),
    lambda r: (lambda h, A: f"Convective heat transfer from the housing: with a film coefficient of {h} W/m2K over an exposed area of {A} m2 and a 20 K difference to ambient, the loss is about {float(h) * float(A) * 20:.0f} W. Radiation adds a little but I left it out for now.")(_f(r, 5, 25, 1), _f(r, 0.4, 2.5, 2)),
    lambda r: (lambda F, A: f"Contact stress check on the support pad: a load of {F} kN spread over {A} cm2 gives a mean bearing pressure of {float(F) * 10 / float(A):.2f} MPa. The pad material is rated well above that, so no change there.")(_f(r, 10, 60, 1), _f(r, 40, 150, 0)),
    lambda r: (lambda Q, H: f"The pump duty point is {Q} m3/h at a head of {H} m. With an assumed efficiency of 0.72 the hydraulic power is about {float(Q) / 3600 * 1000 * 9.81 * float(H) / 0.72 / 1000:.2f} kW at the shaft, so the 7.5 kW frame is adequate.")(_f(r, 10, 60, 1), _f(r, 8, 35, 1)),
    lambda r: (lambda m, a: f"For the lifting lug I took a mass of {m} kg and a dynamic factor of {a}. The design load is then about {float(m) * 9.81 * float(a) / 1000:.2f} kN per lug when two lugs share the lift evenly, which matches the earlier hand estimate.")(_f(r, 200, 900, 0), _f(r, 1.2, 1.8, 2)),
    lambda r: (lambda L, dT: f"Thermal expansion of the {L} m steel run over a {dT} K swing, with a coefficient of 12e-6 per K, comes to about {float(L) * 1000 * 12e-6 * float(dT):.2f} mm. The expansion loop drawn on sheet 4 takes up more than that.")(_f(r, 6, 40, 1), _f(r, 30, 90, 0)),
    lambda r: (lambda c, t: f"The sampling run used a concentration of {c} mg/L in the feed and a contact time of {t} min. Removal was close to linear over that window, so I kept the first-order rate constant from the lab sheet rather than refitting it.")(_f(r, 2, 30, 1), _f(r, 10, 45, 0)),
    lambda r: (lambda f, k: f"Natural frequency of the mounting frame: with a lumped mass of {f} kg and a stiffness of {k} kN/m, the first mode sits near {(float(k) * 1000 / float(f)) ** 0.5 / (2 * 3.14159):.1f} Hz. That is away from the running speed, so no detuning is needed.")(_f(r, 50, 400, 0), _f(r, 200, 2000, 0)),
    lambda r: (lambda p, d: f"Hoop stress in the vessel shell: internal pressure {p} bar, mean diameter {d} mm and a 6 mm wall give about {float(p) * 0.1 * float(d) / (2 * 6):.1f} MPa. The allowable for the plate grade is higher, and the corrosion allowance was already taken off.")(_f(r, 4, 16, 1), _f(r, 300, 900, 0)),
    lambda r: (lambda u, w: f"Wind load on the panel used a design wind speed of {u} m/s and a panel width of {w} m. The dynamic pressure is about {0.613 * float(u) ** 2:.0f} Pa, and with a force coefficient of 1.2 the line load on the top rail stays modest.")(_f(r, 20, 40, 1), _f(r, 1.0, 3.0, 2)),
    lambda r: (lambda s, n: f"The belt drive runs at a belt speed of {s} m/s with {n} belts in parallel. Checking the catalogue power per belt at that speed, the drive carries the load with about 15 percent spare, which is what we agreed on last week.")(_f(r, 10, 28, 1), r.randint(2, 6)),
    lambda r: (lambda e, t: f"Energy use for the batch: the heater draws {e} kW for {t} hours per cycle, so a cycle costs about {float(e) * float(t):.1f} kWh. Insulating the lid would trim that, but it is outside the scope of this review.")(_f(r, 3, 18, 1), _f(r, 1.5, 6.0, 1)),
    lambda r: (lambda g, m: f"Air change rate for the room: a volume of {g} m3 and a supply of {m} m3/h give about {float(m) / float(g):.1f} changes per hour, above the minimum in the brief. The diffuser layout from the earlier markup is kept.")(_f(r, 80, 400, 0), _f(r, 400, 2400, 0)),
]

# Targets: (name, unit, lo, hi, decimals). Names never appear in BANK.
TARGETS = [
    ("coating thickness", "micrometres", 120, 480, 1),
    ("settling velocity", "m/s", 0.01, 0.09, 4),
    ("bolt preload", "kN", 11, 39, 2),
    ("weld throat size", "mm", 3.1, 9.9, 2),
    ("spring rate", "N/mm", 120, 880, 1),
    ("dew point", "degrees C", 6, 19, 2),
    ("bearing life", "hours", 2100, 9800, 1),
    ("tank residence time", "minutes", 21, 98, 2),
    ("cable sag", "m", 0.4, 2.9, 3),
    ("gear backlash", "mm", 0.05, 0.45, 3),
    ("filter pressure drop", "kPa", 1.1, 8.9, 3),
    ("flywheel inertia", "kg m2", 0.11, 0.99, 4),
]
PLACEMENTS = ["early", "middle", "late"]
LEAD = "Next, the {name}, using the same inputs as above and the factors from the standard."


def _value(rng, lo, hi, dec):
    while True:  # non-round: last digit not 0
        v = f"{rng.uniform(lo, hi):.{dec}f}"
        if v[-1] != "0":
            return v


def _count(text, value):
    return len(re.findall(r"(?<![\d.])" + re.escape(value) + r"(?![\d])", text))


def make_pairs(seed=0):
    rng = random.Random(seed)
    pairs = []
    for name, unit, lo, hi, dec in TARGETS:
        for placement in PLACEMENTS:
            value = _value(rng, lo, hi, dec)
            target = LEAD.format(name=name) + f" So the {name} comes out to {value} {unit}."
            while True:
                order = list(range(len(BANK)))
                rng.shuffle(order)
                paras = [BANK[i](rng) for i in order]
                if not any(_count(p, value) for p in paras):
                    break
            if placement == "early":
                paras.insert(1, target)
            elif placement == "late":
                paras.insert(len(paras) - 1, target)
            else:  # the paragraph boundary closest to the middle
                total = sum(len(p) + 2 for p in paras)
                acc, best = 0, (1e9, 0)
                for i, p in enumerate(paras):
                    best = min(best, (abs(acc - total / 2), i))
                    acc += len(p) + 2
                paras.insert(best[1], target)
            a_text = "\n\n".join(paras)
            pairs.append(dict(
                id=f"{name.replace(' ', '_')}_{placement}", name=name, value=value,
                unit=unit, placement=placement, a_text=a_text,
                question=f"What value did you get for the {name}? Reply with only the number."))
    return pairs


# ---------------------------------------------------------------- arms and scoring

def default_arms():
    arms = [dict(name="note400", kind="text"), dict(name="text_full", kind="text"),
            dict(name="mem_full", kind="mem", sel=None)]
    for n in (32, 128):
        for sel in ("last", "writer", "reader"):
            arms.append(dict(name=f"{sel}_{n}", kind="mem", sel=sel, n=n, restamp=True))
    for sel in ("last", "writer", "reader"):
        arms.append(dict(name=f"{sel}_128_orig", kind="mem", sel=sel, n=128, restamp=False))
    return arms


A_HEAD = "<|im_start|>user\nWork through the calculations for this design review.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def b_turn(content):
    return "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def is_correct(reply, value):
    m = NUM.search(reply)
    if not m:
        return False
    got = m.group(0).replace(",", "").strip()
    try:
        g, v = float(got), float(value)
    except ValueError:
        return False
    return got == value or abs(g - v) <= 1e-9 * max(abs(v), 1e-12)


# ---------------------------------------------------------------- the run

def run_value_test(model, tok, out_dir, arms=None, pairs=None):
    import torch
    import kv

    arms = arms or default_arms()
    pairs = pairs or make_pairs()
    dev = next(model.parameters()).device
    enc = lambda s: tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "value_rows.jsonl")
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    done = {(r["id"], r["arm"]) for r in rows}
    ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()

    with ctx, torch.no_grad(), open(path, "a", encoding="utf-8") as out:
        for p in pairs:
            todo = [a for a in arms if (p["id"], a["name"]) not in done]
            if not todo:
                continue
            a_mem = writer = reader = None
            b_mem_ids = enc(b_turn(p["question"]))
            for arm in todo:
                t0 = time.time()
                if arm["kind"] == "text":
                    notes = p["a_text"][-400:] if arm["name"] == "note400" else p["a_text"]
                    ids = enc(b_turn("Notes from earlier:\n" + notes + "\n\n" + p["question"]))
                    mem, _, _ = kv.run(model, ids[:, :1])  # start from scratch, decode the rest
                    new, _ = kv.decode(model, mem, ids[:, 1:], 16, stop)
                    kept, a_slots = None, None
                else:
                    if a_mem is None:
                        a_mem, writer = kv.build(model, enc(A_HEAD + p["a_text"] + "<|im_end|>\n"), window=32)
                    mem = a_mem
                    if arm["sel"] is not None:
                        lo = kv.SINKS
                        if arm["sel"] == "last":
                            idx = kv.pick_last(a_mem, arm["n"], lo=lo)
                        else:
                            if arm["sel"] == "reader" and reader is None:
                                reader = kv.reader_scores(model, a_mem, b_mem_ids)
                            s = writer if arm["sel"] == "writer" else reader
                            idx = kv.pick_top(s, arm["n"], lo=lo)
                        mem = kv.keep(a_mem, idx)
                        if arm["restamp"]:
                            mem = kv.restamp(model, mem, start=0)
                    kept, a_slots = mem.slots, a_mem.slots
                    new, _ = kv.decode(model, mem, b_mem_ids, 16, stop)
                reply = tok.decode(new, skip_special_tokens=True)
                row = dict(id=p["id"], arm=arm["name"], name=p["name"], placement=p["placement"],
                           value=p["value"], reply=reply, correct=is_correct(reply, p["value"]),
                           kept_slots=kept, a_slots=a_slots, seconds=round(time.time() - t0, 3))
                out.write(json.dumps(row) + "\n")
                out.flush()
                rows.append(row)

    return summarize(rows, arms, pairs)


def summarize(rows, arms, pairs):
    ids = {p["id"] for p in pairs}
    summary = {}
    print(f"{'arm':<16}{'all':>7}{'early':>7}{'middle':>8}{'late':>7}{'kept':>8}{'n':>5}")
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm["name"] and r["id"] in ids]
        if not rs:
            continue
        acc = lambda xs: sum(r["correct"] for r in xs) / len(xs) if xs else float("nan")
        by = {pl: acc([r for r in rs if r["placement"] == pl]) for pl in PLACEMENTS}
        ks = [r["kept_slots"] for r in rs if r["kept_slots"] is not None]
        kept = sum(ks) / len(ks) if ks else None
        summary[arm["name"]] = dict(acc=acc(rs), **by, kept=kept, n=len(rs))
        print(f"{arm['name']:<16}{acc(rs):>7.2f}{by['early']:>7.2f}{by['middle']:>8.2f}{by['late']:>7.2f}"
              f"{(f'{kept:.0f}' if kept is not None else '-'):>8}{len(rs):>5}")
    return summary


if __name__ == "__main__":
    ps = make_pairs()
    lens = [len(p["a_text"]) for p in ps]
    print(f"pairs: {len(ps)}, distinct names: {len({p['name'] for p in ps})}")
    print(f"a_text chars: min {min(lens)}, max {max(lens)} (~{min(lens) // 4}-{max(lens) // 4} tokens at 4 chars/token)")
    ok = True
    for p in ps:
        sent = f"So the {p['name']} comes out to {p['value']} {p['unit']}."
        frac = p["a_text"].find(sent) / len(p["a_text"])
        want = {"early": frac < 0.12, "middle": 0.4 < frac < 0.6, "late": frac > 0.85}[p["placement"]]
        once = p["a_text"].count(sent) == 1 and _count(p["a_text"], p["value"]) == 1
        if not (want and once):
            ok = False
            print("BAD", p["id"], round(frac, 3), _count(p["a_text"], p["value"]))
    for pl in PLACEMENTS:
        fr = [p["a_text"].find(p["value"]) / len(p["a_text"]) for p in ps if p["placement"] == pl]
        print(f"{pl:<7} target at {min(fr):.2f}-{max(fr):.2f} of text")
    print("in last 400 chars:", {pl: sum(p["value"] in p["a_text"][-400:] for p in ps if p["placement"] == pl) for pl in PLACEMENTS})
    print("all targets once and placed:", ok)
    print("example:", ps[0]["id"], "|", ps[0]["value"], ps[0]["unit"], "|", ps[0]["question"])
