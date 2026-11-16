import json
import time
import matplotlib.pyplot as plt
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent      
REPO_ROOT = SCRIPT_DIR.parent                       # repo top folder
sys.path.insert(0, str(REPO_ROOT))                  # so "src" is found from anywhere

from src.scheduler_core import load_instance, create_initial_state


# time_limit is set 300 seconds to avoid inflated hard score for large datasets
def hill_climbing(state, time_limit=300):
    start = time.perf_counter()
    score, initial_hard, initial_soft = state.evaluate()
    initial_score=score
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
        "initial_score": initial_score,
        "initial_hard": initial_hard,
        "initial_soft": initial_soft,
        "is_feasible": (state.evaluate()[1] == 0),
        "best_hard": state.evaluate()[1],
        "best_soft": state.evaluate()[2],
        "neighbors_checked": checked,
        "seconds": time.perf_counter() - start,
        "stop_reason": reason,
        "cost_history": history,
    }
    return state, results

def run_benchmark(instance_path: str, seed=42, time_limit=300) -> dict:
    problem = load_instance(instance_path)
    state = create_initial_state(problem, seed)
    state, results=hill_climbing(state,time_limit)
    return state, results
    
if __name__ == "__main__":

    scale=["small", "medium", "large"]
    feasible=[]
    mean_h=[]
    mean_s=[]
    mean_iter=[]
    mean_time=[]
    mean_initial=[]
    mean_final=[]

    for x in range(3):
        sum_f=0
        sum_h=0
        sum_s=0
        sum_iter=0
        sum_time=0
        sum_initial=0
        sum_final=0
        y_1=42
        for y in range(5):
            filepath=f"data/{scale[x]}_seed{y_1}.json"
            seed=100
            for z in range(5):
                new_state, results = run_benchmark(filepath,seed)
                if results["is_feasible"]:
                    sum_f+=1
                sum_h+=results["best_hard"]
                sum_s+=results["best_soft"]
                sum_iter+=results["steps"]
                sum_time+=results["seconds"]
                sum_initial+=results["initial_score"]  
                sum_final+=new_state.evaluate()[0]
                print(f"{filepath}\tseed={seed}\thard {results["initial_hard"]} -> {results["best_hard"]}\tsoft {results["initial_soft"]} -> {results["best_soft"]}\titers= {results["steps"]}\t{results["seconds"]}")
                seed+=1
                if ((x==y==z==0) or ((x==1) and (y==z==0)) or ((x==2) and (y==z==0))):
                    with open(f"result_{x+1}.json", "w") as file:
                        json.dump(results, file, indent=2)
            y_1+=1
        feasible.append(100*sum_f/25)
        mean_h.append(sum_h/25)
        mean_s.append(sum_s/25)
        mean_iter.append(sum_iter/25)
        mean_time.append(sum_time/25)
        mean_initial.append(sum_initial/25)
        mean_final.append(sum_final/25)
    print("Summary:")
    print("scale\tfeasible %\thard\tsoft\titers\ttime s")
    for i in range(3):
        print(f"{scale[i]}\t{feasible[i]}\t\t{mean_h[i]:.1f}\t{mean_s[i]:.1f}\t{mean_iter[i]:.1f}\t{mean_time[i]}")

    summary_list=[]
    with open(f"result_summary.json", "w") as file:
        for x in range(3):
            summary_list.append({
                "scale": scale[x],
                "feasible": feasible[x],
                "mean_hard": mean_h[x],
                "mean_soft": mean_s[x],
                "mean_iterations": mean_iter[x],
                "mean_time": mean_time[x],
                "mean_initial": mean_initial[x],
                "mean_final": mean_final[x]
            })
        json.dump(summary_list, file, indent=2)
    # Visual Representation Export
    names = scale
    
    

    xs = [0, 1, 2]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([x - 0.2 for x in xs], mean_initial, 0.4, label="Before")
    ax.bar([x + 0.2 for x in xs], mean_final, 0.4, label="After")
    
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("cost (smaller is better)")
    ax.set_title("Before vs After")
    ax.legend()
    
    fig.tight_layout()
    fig.savefig("graph.png", dpi=150)
    print("Saved graph.png")
