# CS-3368-Project-RedRaiderbacktrackers

# University Course Timetabling Optimization

**Team:** RedRaiderBacktrackers (Pawanjit Birdi, Akaela Espina, Camila Verdezoto)  
**Course:** CS 3368 – Artificial Intelligence  

## Overview
This project solves the University Course Timetabling Problem (constraint scheduling problem) using local search and optimization. All solvers optimize a shared objective function that strictly prioritizes eliminating hard constraint violations (room conflicts, instructor overlaps, student cohort clashes, and room capacity limits) before minimizing soft constraint penalties (instructor/course time preferences, student schedule gaps, and instructor building changes).

## Repository Structure
* `src/scheduler_core.py` – Shared problem loader, initial state generator, `TimetableState` representation, constraint evaluator, and neighborhood move operators (`reassign` and `swap`).
* `hill_climbing.py` – Base model implementing Steepest-Ascent Hill Climbing (Camila Verdezoto).
* `simulated_annealing.py` – Base + 1 improvement implementing Simulated Annealing with calibrated initial temperature and geometric cooling (Akaela Espina).
* `tabu_search.py` – Base + 2 improvements implementing Tabu Search with dynamic tabu tenure ($k = \lceil 2n/3 \rceil$), aspiration criterion, and candidate neighborhood sampling (Pawanjit Birdi).
* `metrics_report.py` – Comparative benchmark runner and visualization script across problem tiers.
* `data/` – Synthetic problem instances across three scales (`small` $n=25$, `medium` $n=50$, `large` $n=100$) and five seeds (`seed42`–`seed46`).

## Requirements
* Python 3.10+
* Standard library modules (`json`, `math`, `random`, `time`, `collections`, `pathlib`, `argparse`, `csv`)
* Third-party libraries for plotting: `matplotlib`, `numpy`

Install plotting dependencies if needed:
```bash
pip install matplotlib numpy
```

## How to Run
All scripts automatically resolve the repository root path, so they can be executed directly from any working directory:

```bash
# Run individual solvers across benchmark instances
python hill_climbing.py
python simulated_annealing.py
python tabu_search.py

# Generate comparative convergence and cost charts
python metrics_report.py
```