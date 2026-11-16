from hill_climbing import hill_climbing
from simulated_annealing import simulated_annealing
from tabu_search import tabu_search
import matplotlib.pyplot as plt
import numpy as np
'''
representative_path="data/large_seed46.json"

seed=42
hc_hist=hill_climbing.run_benchmark(representative_path, seed)[1]["cost_history"]
sa_res=simulated_annealing.run_benchmark(representative_path, seed)
sa_hist=[item[1] for item in sa_res["history"]]
ts_hist=tabu_search.run_benchmark(representative_path, seed)["cost_hist"]

x_hc = range(len(hc_hist))
x_sa = [item[0] for item in sa_res["history"]]
x_ts = range(len(ts_hist))

fig, ax = plt.subplots(3,1,figsize=(5, 10),sharey=True)

ax[0].plot(x_hc, hc_hist, color="darkorange", linestyle="--")
ax[1].plot(x_sa, sa_hist, color="teal", linestyle=":")
ax[2].plot(x_ts, ts_hist, color="royalblue", linestyle="-")

ax[0].set_yscale("log")
# Customization
ax[0].set_title("Hill Climbing")
ax[0].set_xlabel("Iterations")
ax[0].set_ylabel("Best Cost")
ax[0].grid(True)

ax[1].set_yscale("log")
# Customization
ax[1].set_title("Simulated Annealing")
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Best Cost")
ax[1].grid(True)

ax[2].set_yscale("log")
# Customization
ax[2].set_title("Tabu Search")
ax[2].set_xlabel("Iterations")
ax[2].set_ylabel("Best Cost")
ax[2].grid(True)

plt.tight_layout()
# Display the chart
plt.savefig("convergence_comparison.png", dpi=200)
print("convergence")
'''
scale=["small"]
soft_hc=[]
hard_hc=[]
soft_sa=[]
hard_sa=[]
soft_ts=[]
hard_ts=[]

for x in range(len(scale)):
    seedval=42
    sum_soft_hc=0
    sum_hard_hc=0
    sum_soft_sa=0
    sum_hard_sa=0
    sum_soft_ts=0
    sum_hard_ts=0
    for y in range(5):
        filepath=f"data/{scale[x]}_seed{seedval}.json"
        seed_v=100
        for z in range(5):
            res1=hill_climbing.run_benchmark(filepath, seed_v)[1]
            res2=simulated_annealing.run_benchmark(filepath, seed_v)
            res3=tabu_search.run_benchmark(filepath, seed_v)
            sum_soft_hc+=res1["best_soft"]
            sum_hard_hc+=res1["best_hard"]
            sum_soft_sa+=res2["best_soft"]
            sum_hard_sa+=res2["best_hard"]
            sum_soft_ts+=res3["soft_penalty"]
            sum_hard_ts+=res3["hard_violations"]
            seed_v+=1
        seedval+=1
    soft_hc.append(sum_soft_hc/25)
    hard_hc.append(sum_hard_hc/25)
    soft_sa.append(sum_soft_sa/25)
    hard_sa.append(sum_hard_sa/25)
    soft_ts.append(sum_soft_ts/25)
    hard_ts.append(sum_hard_ts/25)

fig, ax = plt.subplots(2,1,figsize=(8, 10))

x=np.arange(len(scale))
ax[0].bar(x-0.2,soft_hc,width=0.2,label='Hill Climbing', color='#1f77b4')
ax[0].bar(x,soft_sa,width=0.2,label='Simulated Annealing', color='#ff7f0e')
ax[0].bar(x+0.2,soft_ts,width=0.2,label='Tabu Search', color='#2ca02c')


ax[0].set_xlabel('Scale')
ax[0].set_ylabel('Best Soft Penalty')
ax[0].set_title('Soft Penalty Comparison')

ax[0].set_xticks(x)
ax[0].set_xticklabels(scale)

# Add a legend
ax[0].legend()

ax[1].bar(x-0.2,hard_hc,width=0.2,label='Hill Climbing', color='#1f77b4')
ax[1].bar(x,hard_sa,width=0.2,label='Simulated Annealing', color='#ff7f0e')
ax[1].bar(x+0.2,hard_ts,width=0.2,label='Tabu Search', color='#2ca02c')


ax[1].set_xlabel('Scale')
ax[1].set_ylabel('Best Hard Count')
ax[1].set_title('Hard Count Comparison')

ax[1].set_xticks(x)
ax[1].set_xticklabels(scale)

# Add a legend
ax[1].legend()

# 6. Display the chart
plt.tight_layout()
plt.savefig("cost_comparison.png", dpi=200)