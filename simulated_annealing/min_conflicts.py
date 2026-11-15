"""
Min-Conflicts Hill Climbing: comparison model from the peer-reviewed paper
    Minton, Johnston, Philips & Laird (1992), Artificial Intelligence 58, 161-205.

Author: Akaela Espina

How it follows the paper:
    1. Start from a complete (conflicting) assignment, the same start that
       simulated_annealing.py uses for the same seed.
    2. Each step, pick a variable (course) that is in a hard conflict.
    3. Give it the value (room, timeslot) with the minimum cost; break ties randomly.
Adaptation for our problem: the paper only has hard constraints, so the value is
chosen by our full weighted cost (hard x 10000 + soft). Once no hard conflicts
remain, any course may be picked, so the soft penalty keeps improving.
Stops when the best cost has not improved for --patience steps, or at the limits.

Usage (from the repo top folder):
    python simulated_annealing/min_conflicts.py
    python simulated_annealing/min_conflicts.py --scales small --runs 1
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent      # simulated_annealing/
REPO_ROOT = SCRIPT_DIR.parent                       # repo top folder
sys.path.insert(0, str(REPO_ROOT))                  # so "src" is found from anywhere

from src.scheduler_core import load_instance, create_initial_state

HARD_WEIGHT = 10000  # same default as TimetableState.evaluate()


class LocalCost:
    """Cost terms that involve ONE course, so a value can be scored without
    re-scoring the whole timetable. Mirrors TimetableState.evaluate() exactly:
    total(after) - total(before) == local(new value) - local(old value)."""

    def __init__(self, problem):
        self.courses = problem["courses"]
        self.rooms = problem["rooms"]
        self.slots = problem["timeslots"]
        self.cohort_members = {}
        self.instructor_members = {}
        for c_id, c in self.courses.items():
            self.cohort_members.setdefault(c["student_cohort"], []).append(c_id)
            self.instructor_members.setdefault(c["instructor_id"], []).append(c_id)

    def context(self, assignments, c_id):
        """Everything about the OTHER courses that c_id's cost depends on."""
        course = self.courses[c_id]
        by_slot = {}
        for other, (r, t) in assignments.items():
            if other != c_id:
                by_slot.setdefault(t, []).append((other, r))
        cohort_orders = {"MWF": [], "TR": []}
        for other in self.cohort_members[course["student_cohort"]]:
            if other != c_id:
                t = assignments[other][1]
                cohort_orders[self.slots[t]["day_group"]].append(self.slots[t]["slot_order"])
        inst_bldgs = {"MWF": set(), "TR": set()}
        for other in self.instructor_members[course["instructor_id"]]:
            if other != c_id:
                r, t = assignments[other]
                inst_bldgs[self.slots[t]["day_group"]].add(self.rooms[r]["building"])
        return by_slot, cohort_orders, inst_bldgs

    def cost(self, c_id, value, ctx):
        """(hard, soft) of every term that involves course c_id at value (room, slot)."""
        by_slot, cohort_orders, inst_bldgs = ctx
        course = self.courses[c_id]
        r, t = value
        slot = self.slots[t]
        hard = soft = 0
        for other, r2 in by_slot.get(t, ()):                       # pairwise clashes
            oc = self.courses[other]
            hard += (r2 == r) + (oc["instructor_id"] == course["instructor_id"]) \
                + (oc["student_cohort"] == course["student_cohort"])
        if self.rooms[r]["capacity"] < course["enrollment"]:       # capacity
            hard += 1
        if t not in course["preferred_slots"]:                     # individual prefs
            soft += 5
        if slot["day_group"] not in course["preferred_days"]:
            soft += 4
        if slot["is_undesirable"]:
            soft += 3
        for day in ("MWF", "TR"):                                  # cohort gaps
            orders = list(cohort_orders[day])
            if slot["day_group"] == day:
                orders.append(slot["slot_order"])
            orders.sort()
            soft += 5 * sum(1 for i in range(len(orders) - 1) if orders[i + 1] - orders[i] > 1)
        for day in ("MWF", "TR"):                                  # instructor buildings
            bldgs = set(inst_bldgs[day])
            if slot["day_group"] == day:
                bldgs.add(self.rooms[r]["building"])
            if len(bldgs) > 1:
                soft += 4
        return hard, soft


def conflicted_courses(assignments, courses):
    """Courses involved in at least one pairwise hard clash."""
    by_slot = {}
    for c_id, (r, t) in assignments.items():
        by_slot.setdefault(t, []).append((c_id, r))
    bad = set()
    for group in by_slot.values():
        for i in range(len(group)):
            a, ra = group[i]
            for j in range(i + 1, len(group)):
                b, rb = group[j]
                if (ra == rb or courses[a]["instructor_id"] == courses[b]["instructor_id"]
                        or courses[a]["student_cohort"] == courses[b]["student_cohort"]):
                    bad.update((a, b))
    return list(bad)


def min_conflicts(problem, seed=42, patience=500, max_steps=50_000,
                  time_limit=300.0, history_every=10):
    start = time.perf_counter()
    state = create_initial_state(problem, seed=seed)      # same start as SA
    rng = random.Random(seed + 20_000)
    local = LocalCost(problem)
    courses = problem["courses"]
    slot_ids = list(problem["timeslots"].keys())
    values = {c_id: [(r_id, t) for r_id, r in problem["rooms"].items()
                     if r["capacity"] >= c["enrollment"] for t in slot_ids]
              for c_id, c in courses.items()}

    total, hard, soft = state.evaluate()
    initial = (total, hard, soft)
    best = (total, hard, soft)
    best_assign = dict(state.assignments)
    steps = evaluated = changes = since_best = 0
    history = [(0, total, hard, soft, 0)]
    stop_reason = "no_improvement"

    while True:
        conflicted = conflicted_courses(state.assignments, courses)
        c_id = rng.choice(conflicted) if conflicted else rng.choice(list(courses))
        ctx = local.context(state.assignments, c_id)
        old = state.assignments[c_id]
        old_h, old_s = local.cost(c_id, old, ctx)

        best_vals, best_cost = [], None
        for v in values[c_id]:                               # min-conflicts value
            h, s = local.cost(c_id, v, ctx)
            evaluated += 1
            c = h * HARD_WEIGHT + s
            if best_cost is None or c < best_cost:
                best_vals, best_cost = [(v, h, s)], c
            elif c == best_cost:
                best_vals.append((v, h, s))
        v, h, s = rng.choice(best_vals)                      # random tie-break

        if v != old:
            state.assignments[c_id] = v
            hard += h - old_h
            soft += s - old_s
            total = hard * HARD_WEIGHT + soft
            changes += 1
        steps += 1

        if total < best[0]:
            best, best_assign, since_best = (total, hard, soft), dict(state.assignments), 0
        else:
            since_best += 1
        if steps % history_every == 0:
            history.append((evaluated, best[0], best[1], best[2], steps))

        if best[1] == 0 and since_best >= patience:
            break
        if steps >= max_steps:
            stop_reason = "max_steps"
            break
        if time.perf_counter() - start > time_limit:
            stop_reason = "time_limit"
            break

    history.append((evaluated, best[0], best[1], best[2], steps))
    state.assignments = best_assign
    check = state.evaluate()                                 # full re-score as a safety check
    assert check == best, f"local cost drifted from evaluate(): {check} vs {best}"
    return {
        "best_total": best[0], "best_hard": best[1], "best_soft": best[2],
        "feasible": best[1] == 0,
        "initial_total": initial[0], "initial_hard": initial[1], "initial_soft": initial[2],
        "steps": steps, "changes": changes, "states_evaluated": evaluated,
        "stop_reason": stop_reason, "runtime_s": time.perf_counter() - start,
        "history": history,
    }


def main():
    p = argparse.ArgumentParser(description="Min-conflicts comparison model (Minton et al. 1992)")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    p.add_argument("--out-dir", default=str(SCRIPT_DIR / "results"))
    p.add_argument("--scales", nargs="+", default=["small", "medium", "large"])
    p.add_argument("--runs", type=int, default=5, help="runs per instance (seeds 100, 101, ...)")
    p.add_argument("--patience", type=int, default=500, help="stop after this many steps without improvement")
    p.add_argument("--max-steps", type=int, default=50_000)
    p.add_argument("--time-limit", type=float, default=300.0, help="seconds per run")
    args = p.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for s in args.scales for f in data_dir.glob(f"{s}_seed*.json"))
    if not files:
        raise SystemExit(f"No instance files found in {data_dir}.")

    fields = ["instance", "scale", "run_seed", "initial_total", "initial_hard", "initial_soft",
              "best_total", "best_hard", "best_soft", "feasible", "steps", "changes",
              "states_evaluated", "stop_reason", "runtime_s"]
    rows, histories = [], {}
    for f in files:
        problem = load_instance(str(f))
        scale = problem["metadata"].get("scale", f.stem.split("_")[0])
        for run in range(args.runs):
            seed = 100 + run                                  # same seeds as SA
            r = min_conflicts(problem, seed=seed, patience=args.patience,
                              max_steps=args.max_steps, time_limit=args.time_limit)
            row = {k: r[k] for k in fields if k in r}
            row.update(instance=f.stem, scale=scale, run_seed=seed, runtime_s=round(r["runtime_s"], 3))
            rows.append(row)
            histories[f"{f.stem}_run{seed}"] = r["history"]
            print(f"{f.stem:20s} seed={seed}  hard {r['initial_hard']:>3} -> {r['best_hard']:<3} "
                  f"soft {r['initial_soft']:>4} -> {r['best_soft']:<4} "
                  f"steps={r['steps']:>5}  evaluated={r['states_evaluated']:>7}  "
                  f"{r['runtime_s']:.1f}s  ({r['stop_reason']})")

    with open(out_dir / "mc_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "mc_history.json", "w") as fh:
        json.dump(histories, fh)
    print(f"\nWrote {out_dir / 'mc_results.csv'} and {out_dir / 'mc_history.json'}")


if __name__ == "__main__":
    main()
