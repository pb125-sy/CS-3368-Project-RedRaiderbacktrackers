from src import scheduler_core

import json
import time
import matplotlib.pyplot as plt

# time_limit is set 10 seconds!
def hill_climbing(state, time_limit=10):
    start = time.perf_counter()
    score = state.evaluate()[0]
    steps = 0
    checked = 0
    history = [score]
    reason = "Time limit"

    while time.perf_counter() - start < time_limit:
        best_state = state
        best_score = score

        for move, neighbor in state.get_neighbors():
            if time.perf_counter() - start >= time_limit:
                break
            new_score = neighbor.evaluate()[0]
            checked += 1
            if new_score < best_score:
                best_state = neighbor
                best_score = new_score

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

    # run 1: small_seed42 
    file_name = "../data/small_seed42.json"
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
    results["schedule"] = str(state.assignments)

    print("run 1 -> cost:", total, "| steps:", results["steps"], "| secs:", round(results["seconds"], 3))

    with open("result_1.json", "w") as file:
        json.dump(results, file, indent=2)

    # run 2: medium_seed42
    file_name = "../data/medium_seed42.json"
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
    results["schedule"] = str(state.assignments)

    print("run 2 -> cost:", total, "| steps:", results["steps"], "| secs:", round(results["seconds"], 3))

    with open("result_2.json", "w") as file:
        json.dump(results, file, indent=2)

    # run 3: large_seed42
    file_name = "../data/large_seed42.json"
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
    results["schedule"] = str(state.assignments)

    print("run 3 -> cost:", total, "| steps:", results["steps"], "| secs:", round(results["seconds"], 3))

    with open("result_3.json", "w") as file:
        json.dump(results, file, indent=2)

    # Visual Representation Export
    names = []
    before = []
    after = []
    
    for i in (1, 2, 3):
        with open("result_" + str(i) + ".json") as file:
            r = json.load(file)
        names.append(r["instance"].split("/")[-1])
        before.append(r["initial_cost"])
        after.append(r["total_cost"])

    xs = [0, 1, 2]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([x - 0.2 for x in xs], before, 0.4, label="Before")
    ax.bar([x + 0.2 for x in xs], after, 0.4, label="After")
    
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("cost (smaller is better)")
    ax.set_title("Before vs After")
    ax.legend()
    
    fig.tight_layout()
    fig.savefig("graph.png", dpi=150)
    print("Saved graph.png")
