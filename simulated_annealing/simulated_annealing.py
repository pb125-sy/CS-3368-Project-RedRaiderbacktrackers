"""
Simulated Annealing with Geometric Cooling (base model + 1 improvement)
Author: Akaela Espina

Improvement over steepest-ascent hill climbing (base model):
    stochastic acceptance. A worse neighbor is accepted with probability
    exp(-delta / T), so the search can leave local minima and plateaus.
    T starts high and shrinks geometrically: T <- alpha * T.

Uses the team's shared scheduler_core.py for the state, the initial
state, and the cost function, so results compare fairly across models.

Usage (from the repo top folder):
    python simulated_annealing/simulated_annealing.py                  # all instances, 5 runs each
    python simulated_annealing/simulated_annealing.py --scales small   # one size only
    python simulated_annealing/simulated_annealing.py --runs 3 --time-limit 60
"""

import argparse
import csv
import json
import math
import random
import time
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent      # simulated_annealing/
REPO_ROOT = SCRIPT_DIR.parent                       # repo top folder
sys.path.insert(0, str(REPO_ROOT))                  # so "src" is found from anywhere

from src.scheduler_core import load_instance, create_initial_state


# Neighborhood: ONE random move per step (annealing never needs the full list)
# Same two move types as TimetableState.get_neighbors(): reassign and swap.

class RandomNeighborhood:
    def __init__(self, problem: dict, rng: random.Random):
        self.rng = rng
        self.course_ids = list(problem["courses"].keys())
        self.timeslot_ids = list(problem["timeslots"].keys())
        self.rooms = problem["rooms"]
        self.courses = problem["courses"]
        # rooms big enough for each course, computed once
        self.valid_rooms = {
            c_id: [r_id for r_id, r in self.rooms.items()
                   if r["capacity"] >= c["enrollment"]]
            for c_id, c in self.courses.items()
        }

    def _reassign(self, state):
        c_id = self.rng.choice(self.course_ids)
        current = state.assignments[c_id]
        occupied = set(state.assignments.values())
        for _ in range(20):
            target = (self.rng.choice(self.valid_rooms[c_id]),
                      self.rng.choice(self.timeslot_ids))
            if target != current and target not in occupied:
                nxt = state.copy()
                nxt.assignments[c_id] = target
                return ("reassign", c_id, current, target), nxt
        return None

    def _swap(self, state):
        for _ in range(20):
            a, b = self.rng.sample(self.course_ids, 2)
            r_a, _ = state.assignments[a]
            r_b, _ = state.assignments[b]
            if (state.assignments[a] != state.assignments[b]
                    and self.rooms[r_b]["capacity"] >= self.courses[a]["enrollment"]
                    and self.rooms[r_a]["capacity"] >= self.courses[b]["enrollment"]):
                nxt = state.copy()
                nxt.swap(a, b)
                return ("swap", a, b), nxt
        return None

    def sample(self, state):
        first, second = ((self._reassign, self._swap) if self.rng.random() < 0.5
                         else (self._swap, self._reassign))
        return first(state) or second(state)


# Simulated annealing

def initial_temperature(state, neighborhood, cost, samples=200, accept_prob=0.8):
    """Pick T0 so a typical uphill move is accepted ~80% of the time at the start."""
    uphill = []
    for _ in range(samples):
        result = neighborhood.sample(state)
        if result:
            delta = result[1].evaluate()[0] - cost
            if delta > 0:
                uphill.append(delta)
    if not uphill:
        return 1.0
    return -(sum(uphill) / len(uphill)) / math.log(accept_prob)


def simulated_annealing(problem, seed=42, alpha=0.95, iters_per_temp=100,
                        t_min=0.01, max_iters=200_000, time_limit=300.0,
                        history_every=100):
    start = time.perf_counter()
    current = create_initial_state(problem, seed=seed)
    rng = random.Random(seed + 10_000)          # separate stream from the initial state
    neighborhood = RandomNeighborhood(problem, rng)

    cur_cost, cur_hard, cur_soft = current.evaluate()
    best, best_eval = current.copy(), (cur_cost, cur_hard, cur_soft)
    initial_eval = best_eval

    T = initial_temperature(current, neighborhood, cur_cost)
    T0 = T
    iters = accepted = uphill_accepted = 0
    history = [(0, best_eval[0], best_eval[1], best_eval[2], T)]
    stop_reason = "t_min"

    while T > t_min:
        for _ in range(iters_per_temp):
            result = neighborhood.sample(current)
            iters += 1
            if result is None:
                continue
            _, candidate = result
            cand_cost, cand_hard, cand_soft = candidate.evaluate()
            delta = cand_cost - cur_cost

            # stochastic acceptance: always take better, sometimes take worse
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current = candidate
                cur_cost, cur_hard, cur_soft = cand_cost, cand_hard, cand_soft
                accepted += 1
                if delta > 0:
                    uphill_accepted += 1
                if cur_cost < best_eval[0]:
                    best, best_eval = current.copy(), (cur_cost, cur_hard, cur_soft)

            if iters % history_every == 0:
                history.append((iters, best_eval[0], best_eval[1], best_eval[2], T))
            if iters >= max_iters:
                stop_reason = "max_iters"
                break
            if time.perf_counter() - start > time_limit:
                stop_reason = "time_limit"
                break
        if stop_reason != "t_min":
            break
        T *= alpha                               # geometric cooling

    history.append((iters, best_eval[0], best_eval[1], best_eval[2], T))
    return {
        "best_state": best,
        "best_total": best_eval[0],
        "best_hard": best_eval[1],
        "best_soft": best_eval[2],
        "feasible": best_eval[1] == 0,
        "initial_total": initial_eval[0],
        "initial_hard": initial_eval[1],
        "initial_soft": initial_eval[2],
        "iterations": iters,
        "accepted": accepted,
        "uphill_accepted": uphill_accepted,
        "T0": T0,
        "stop_reason": stop_reason,
        "runtime_s": time.perf_counter() - start,
        "history": history,
    }


# Experiment runner: every instance x several seeds -> results/ folder 

def main():
    p = argparse.ArgumentParser(description="Simulated annealing timetabler")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    p.add_argument("--out-dir", default=str(SCRIPT_DIR / "results"))
    p.add_argument("--scales", nargs="+", default=["small", "medium", "large"])
    p.add_argument("--runs", type=int, default=5, help="independent runs per instance")
    p.add_argument("--alpha", type=float, default=0.95, help="geometric cooling rate")
    p.add_argument("--iters-per-temp", type=int, default=100)
    p.add_argument("--t-min", type=float, default=0.01)
    p.add_argument("--max-iters", type=int, default=200_000)
    p.add_argument("--time-limit", type=float, default=300.0, help="seconds per run")
    args = p.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for s in args.scales for f in data_dir.glob(f"{s}_seed*.json"))
    if not files:
        raise SystemExit(f"No instance files found in {data_dir}. Run generate_instances.py first.")

    fields = ["instance", "scale", "run_seed", "initial_total", "initial_hard",
              "initial_soft", "best_total", "best_hard", "best_soft", "feasible",
              "iterations", "accepted", "uphill_accepted", "T0", "stop_reason", "runtime_s"]
    rows, histories = [], {}

    for f in files:
        problem = load_instance(str(f))
        scale = problem["metadata"].get("scale", f.stem.split("_")[0])
        for run in range(args.runs):
            seed = 100 + run
            r = simulated_annealing(problem, seed=seed, alpha=args.alpha,
                                    iters_per_temp=args.iters_per_temp, t_min=args.t_min,
                                    max_iters=args.max_iters, time_limit=args.time_limit)
            row = {k: r[k] for k in fields if k in r}
            row.update(instance=f.stem, scale=scale, run_seed=seed,
                       T0=round(r["T0"], 2), runtime_s=round(r["runtime_s"], 3))
            rows.append(row)
            histories[f"{f.stem}_run{seed}"] = r["history"]
            print(f"{f.stem:20s} seed={seed}  hard {r['initial_hard']:>3} -> {r['best_hard']:<3} "
                  f"soft {r['initial_soft']:>4} -> {r['best_soft']:<4} "
                  f"iters={r['iterations']:>6}  {r['runtime_s']:.1f}s  ({r['stop_reason']})")

    with open(out_dir / "sa_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "sa_history.json", "w") as fh:
        json.dump(histories, fh)

    # summary per scale: the numbers for the results table
    print("\nSummary (mean over all instances and runs)")
    print(f"{'scale':8s} {'feasible %':>10s} {'hard':>6s} {'soft':>7s} {'iters':>8s} {'time s':>8s}")
    for s in args.scales:
        rs = [r for r in rows if r["scale"] == s]
        if not rs:
            continue
        n = len(rs)
        print(f"{s:8s} {100 * sum(r['feasible'] for r in rs) / n:10.0f} "
              f"{sum(r['best_hard'] for r in rs) / n:6.2f} "
              f"{sum(r['best_soft'] for r in rs) / n:7.1f} "
              f"{sum(r['iterations'] for r in rs) / n:8.0f} "
              f"{sum(r['runtime_s'] for r in rs) / n:8.2f}")
    print(f"\nWrote {out_dir / 'sa_results.csv'} and {out_dir / 'sa_history.json'}")


if __name__ == "__main__":
    main()

