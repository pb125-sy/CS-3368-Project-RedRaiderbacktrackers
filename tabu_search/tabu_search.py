import time
from collections import deque
import random
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent      
REPO_ROOT = SCRIPT_DIR.parent                       # repo top folder
sys.path.insert(0, str(REPO_ROOT))   

from src.scheduler_core import TimetableState, load_instance, create_initial_state


class tabu_search:
    def __init__(self, tenure: int, max_no_improve: int = 50, timeout_sec: float = 300.0):
        # Stores tabu tenure (e.g., ceil(2n / 3)), stopping threshold, and run budget
        self.tenure=tenure
        self.max_no_improve=max_no_improve
        self.timeout_sec=timeout_sec

    def _is_tabu(self, move_key: tuple, tabu_list: deque) -> bool:
        # Checks if a move (or its reverse swap) is currently forbidden in tabu_list
        if (move_key in tabu_list):
            return True
        return False

    def _get_tabu_key(self, move: tuple, current_state: TimetableState) -> tuple:
        # Normalizes a move descriptor so reverse moves are caught (e.g., sorting swap pairs)
        if move[0]=="reassign":
            room, slot = current_state.assignments[move[1]]
            return ((move[1],room,slot))
        if move[0]=="swap":
            return (tuple(sorted((move[1],move[2]))))

    def solve(self, initial_state: TimetableState) -> dict:
        # Main search loop: evaluates neighbors, checks tabu status & aspiration criterion,
        # updates best_state, tracks iterations/runtime, and returns results dictionary
        current_state=initial_state
        initial_hard=initial_state.evaluate()[1]
        initial_soft=initial_state.evaluate()[2]
        best_state=current_state
        tabu_list=deque(maxlen=self.tenure)
        best_cost = current_state.evaluate()[0]
        no_improve_count=0
        iterations=0
        cost_hist=[best_cost]
        states_expanded=0
        start = time.perf_counter()
        while (time.perf_counter()-start < self.timeout_sec and no_improve_count<self.max_no_improve):
            neighbors = current_state.get_neighbors()
            neighbors_capped = random.sample(neighbors, min(len(neighbors), 500))
            curr_best_cost=math.inf
            best_neighbor=None
            for move, candidate_state in neighbors_capped:
                candidate_cost = candidate_state.evaluate()[0]
                states_expanded+=1
                curr_key = self._get_tabu_key(move, candidate_state)
                if not self._is_tabu(curr_key, tabu_list) or candidate_cost<best_cost:
                    if candidate_cost<curr_best_cost:
                        curr_best_cost=candidate_cost
                        best_neighbor=candidate_state
                        best_move=move
            
            if (best_neighbor==None):
                break    
            if (curr_best_cost<best_cost):
                best_cost=curr_best_cost
                no_improve_count=0
                best_state=best_neighbor
            else:
                no_improve_count+=1
            tabu_list.append(self._get_tabu_key(best_move, current_state))
            current_state=best_neighbor
            cost_hist.append(best_cost)
            iterations+=1
        _, best_hard, best_soft = best_state.evaluate()
        elapsed_time=time.perf_counter()-start
        return {
            "model": "Tabu Search",
            "total_cost": best_cost,
            "initial_hard": initial_hard,
            "hard_violations": best_hard,
            "initial_soft": initial_soft,
            "soft_penalty": best_soft,
            "is_feasible": (best_hard==0),
            "iterations": iterations,
            "states_expanded":states_expanded,
            "runtime_sec":elapsed_time,
            "best_state": best_state,
            "cost_hist": cost_hist
        }

def run_benchmark(instance_path: str, seed: int = 42) -> dict:
    # Loads a JSON instance, builds initial_state, runs TabuSearchSolver, and formats metrics
    problem_data=load_instance(instance_path)
    initial_state = create_initial_state(problem_data, seed)
    dynamic_tenure = math.ceil(2*len(problem_data["courses"])/3)
    tabu = tabu_search(dynamic_tenure)
    result = tabu.solve(initial_state)
    return result

if __name__ == "__main__":
    scale = ["small", "medium", "large"]
    feasible=[]
    mean_h=[]
    mean_s=[]
    mean_iter=[]
    mean_time=[]

    for x in range(3):
        sum_f=0
        sum_h=0
        sum_s=0
        sum_iter=0
        sum_time=0
        y_1=42
        for y in range(5):
            filepath= f"data/{scale[x]}_seed{y_1}.json"
            seed=100
            for z in range(5):
                result=run_benchmark(filepath, seed)
                if result["is_feasible"]:
                    sum_f+=1
                sum_h+=result["hard_violations"]
                sum_s+=result["soft_penalty"]
                sum_iter+=result["iterations"]
                sum_time+=result["runtime_sec"]
                print(f"{filepath}\tseed={seed}\thard {result["initial_hard"]} -> {result["hard_violations"]}\tsoft {result["initial_soft"]} -> {result["soft_penalty"]}\titers= {result["iterations"]}\t{result["runtime_sec"]}")
                seed+=1
            y_1+=1
        feasible.append(100*(sum_f/25))
        mean_h.append(sum_h/25)
        mean_s.append(sum_s/25)
        mean_iter.append(sum_iter/25)
        mean_time.append(sum_time/25)

    print("Summary:")
    print("scale\tfeasible %\thard\tsoft\titers\ttime s")
    for x in range(3):
        print(f"{scale[x]}\t{feasible[x]}\t{mean_h[x]}\t{mean_s[x]}\t{mean_iter[x]}\t{mean_time[x]}")
    