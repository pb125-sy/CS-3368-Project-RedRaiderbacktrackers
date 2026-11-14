from src import scheduler_core

import json
import time


def hill_climbing(state, time_limit=300):
    start = time.perf_counter()
    score = state.evaluate()[0]
    steps = 0
    checked = 0
    history = [score]
    reason = "Time limit"

    while time.perf_counter() - start < time_limit:
        best_state = state
        best_score = score

        # Check all neighbors and keep the lowest cost.
        for move, neighbor in state.get_neighbors():
            if time.perf_counter() - start >= time_limit:
                break
            new_score = neighbor.evaluate()[0]
            checked += 1
            if new_score < best_score:
                best_state = neighbor
                best_score = new_score

        # Only accept a move after checking the full neighborhood.
        if time.perf_counter() - start >= time_limit:
            break
        if best_score == score:
            reason = "No better neighbor"
            break
        state = best_state
        score = best_score
        steps += 1
        history.append(score)

    results = {
        "steps": steps,
        "neighbors_checked": checked,
        "seconds": time.perf_counter() - start,
        "stop_reason": reason,
        "cost_history": history,
    }
    return state, results


if __name__ == "__main__":
    # Change this file name to try another instance.
    file_name = "data/small_seed42.json"
    seed = 42
    problem = scheduler_core.load_instance(file_name)
    state = scheduler_core.create_initial_state(problem, seed)
    initial = state.evaluate()
    state, results = hill_climbing(state)
    total, hard, soft = state.evaluate()
    results["instance"] = file_name
    results["seed"] = seed
    results["initial_cost"] = initial[0]
    results["total_cost"] = total
    results["hard_violations"] = hard
    results["soft_cost"] = soft
    results["schedule"] = state.assignments
    print("Total cost:", total)
    print("Hard violations:", hard)
    print("Soft cost:", soft)
    print("Steps:", results["steps"])
    print("Seconds:", round(results["seconds"], 3))
    print("Stopped because:", results["stop_reason"])
    with open("result.json", "w") as file:
        json.dump(results, file, indent=2)
