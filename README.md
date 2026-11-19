# Evolution of Cooperation in a Multi-Level Continuous Prisoner's Dilemma

Simulation code for our Game Theory project. We implement the two-level spatial Prisoner's Dilemma model of Ispolatov, Simon, and Doebeli (2026) and extend it from a ring of groups to other network structures.

**Authors:** Dev Sonawane, Taksh Patel, Divy Makwana. SVNIT, Surat, India.

**Repository:** https://github.com/Taksh9347/Evolution-of-Cooperation-in-Multi-Level-Continuous-Prisoner-s-Dilemma

**Report:** [`paper/GT_Project_Report.pdf`](paper/GT_Project_Report.pdf) ([`.docx`](paper/GT_Project_Report.docx))

**Base paper:** Y. Ispolatov, B. Simon, and M. Doebeli, "Evolution of cooperation in spatially structured group-selection models of the continuous Prisoner's Dilemma," 2026, [arXiv:2606.03071](https://arxiv.org/abs/2606.03071). A copy is in [`paper/`](paper/).

---

## Overview

In the continuous Prisoner's Dilemma, each player picks a cooperation level `x` between 0 and 1 and gets the payoff

```
P(x, y) = B*y - C*x        (B > C)
```

Lowering your own `x` always raises your payoff, so in a well-mixed population cooperation goes to zero.

The model has two levels:

| Level | Who plays | Strategy | Payoff decides |
|---|---|---|---|
| Within a group (about 100 individuals) | every individual against every other in its group | own `x` | birth rates |
| Between groups | each group against its neighbor groups on a graph | group mean `x̄` | which groups split and which go extinct |

Cooperation loses at both levels when interactions are well mixed. The base paper shows that it can persist when groups sit on a ring.

**What we added:**

- Six graph types instead of only a ring: ring, random regular, honeycomb, prism, Möbius ladder, and square torus.
- A sweep over the number of neighbors `k` (2 to 6).
- The paper's tests of global selection (`σ`), within-group selection (`s`), and fission randomness (`r`), repeated on the new graphs.
- A test of the number of groups (100, 200, 400).

## Repository Structure

```
.
├── gsim.py            # Model: within-group dynamics, fission, BD/DB updates, graphs
├── experiments.py     # Checks, parameter sweeps, figures; resumes from saved results
├── main.ipynb         # Notebook that runs the checks, a quick test, and the full run
├── requirements.txt   # Python dependencies
└── paper/             # Base paper and our report (PDF and DOCX)
```

`results/` and `figures/` are created when you run the experiments. They are listed in `.gitignore`.

## Environment Setup

### Requirements

- Python 3.10 (we used 3.10.11)
- Packages in `requirements.txt`: `numpy`, `scipy`, `networkx`, `matplotlib`, `pandas`, `joblib>=1.3`

### Install

```bash
git clone https://github.com/Taksh9347/Evolution-of-Cooperation-in-Multi-Level-Continuous-Prisoner-s-Dilemma.git
cd Evolution-of-Cooperation-in-Multi-Level-Continuous-Prisoner-s-Dilemma
python -m venv venv
```

Activate the environment:

```bash
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

To use the notebook with this environment, register it as a Jupyter kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name venv --display-name "Python (GT_Project)"
```

### Hardware Used

The full run used an Intel Core i5-13500HX (14 cores, 20 threads) with `--jobs 14`. It took 1 h 9 min. Set `--jobs` to the number of cores you want to use. The default, `-1`, uses all cores.

## Usage

Run all commands from the repository folder.

```bash
# 1. Checks against the base paper (takes under a minute)
python experiments.py check

# 2. Quick test of one experiment (small and fast, for debugging)
python experiments.py e2 --quick --jobs 1

# 3. Full run used in the report (1,058 simulations)
python experiments.py all --G 200 --T 1500 --reps 8 --jobs 14
```

You can also run a single experiment, for example `python experiments.py e5 --G 200 --T 1500 --reps 8`. The notebook `main.ipynb` runs the same commands.

**Resume:** each finished run is written to `results/<experiment>.csv` straight away. If a run stops, start the same command again and only the missing runs will execute.

### Experiments

| Name | Varied | Fixed | Runs |
|---|---|---|---|
| `check` | Three checks: decay rate vs. the paper's appendix formula, within-group decay, neutral drift | | |
| `e2` | 6 graphs × `B` = 1.5, 2.5, 3.5, 4.5, 6 | BD, `σ` = 1, `s` = 0.0005, `r` = 2 | 240 |
| `e3` | ring, random `k`=3, honeycomb × BD, DB × `σ` = 0, 0.5, 1, 2, 4 | `B` = 4.5, `s` = 0.0005, `r` = 2 | 240 |
| `e4` | ring, random `k`=3 × `s` = 0.0001 to 0.003 × `r` = 0.5 to 5 | BD, `B` = 4.5, `σ` = 1 | 256 |
| `e5` | random graph `k` = 2 to 6 × `B` | BD, `σ` = 1, `s` = 0.0005, `r` = 2 | 200 |
| `e6` | number of groups `G` = 100, 200, 400 × `B` | random `k`=3, BD, `σ` = 1 | 120 |
| `snap` | two single runs with time series and graph snapshots | | 2 |

Each setting uses 8 seeds (0 to 7).

### Main Parameters (`gsim.Params`)

| Parameter | Meaning | Default |
|---|---|---|
| `B`, `C` | benefit and cost | 2.5, 1 |
| `beta`, `delta` | birth and death rate scales (group size = `beta/delta` = 100) | 1000, 10 |
| `s` | within-group selection strength | 0.0005 |
| `R` | mutation coefficient | 1e-6 |
| `r` | fission randomness | 1 |
| `sigma` | global selection strength (softmax) | 1 |
| `rule` | `"BD"` or `"DB"` | `"BD"` |
| `local` | `"select"` (best or worst neighbor) or `"random"` | `"select"` |
| `m` | number of cooperation bins | 10 |
| `G` | number of groups | 200 |
| `topology` | `ring`, `rrg`, `prism`, `mobius`, `honeycomb`, `torus`, `rrg4` | `rrg` |
| `k` | degree, used by `rrg` | 3 |
| `T`, `dt` | simulated time and integration step | 3000, 0.05 |

## Dataset Resources

This project does not use an external dataset. All data is generated by the simulation, and the same command with the same seeds gives the same values.

**Output files** in `results/`, one CSV per experiment. Each row is one run: the run's parameters and seed, followed by:

| Column | Meaning |
|---|---|
| `final_mean` | mean cooperation over groups and over the last 20% of the run |
| `final_corr` | neighbor correlation of group cooperation (covariance across edges / variance) |
| `final_var_between` | variance of group cooperation between groups |
| `runtime_s` | wall-clock time of the run in seconds |

**Notes for reading the values:**

- Cooperation is reported between 0.05 and 0.95, which are the centers of the lowest and highest bins. A value of 0.05 means that cooperation is lost.
- At the end of each sweep, `experiments.py` prints a table of the mean and standard deviation over the 8 repeats of each setting.

**External resources:**

- Base paper: [arXiv:2606.03071](https://arxiv.org/abs/2606.03071)
- [NetworkX documentation](https://networkx.org/documentation/stable/) for the graph generators used in `gsim.make_graph`

## Main Results

The full results are in the report. In short:

- Cooperation persisted on all six graphs once the benefit `B` was high enough.
- The benefit needed rose with the number of neighbors `k`. On random regular graphs, cooperation appeared at about `B/C > k`.
- Graphs with the same `k` gave different results. The prism and Möbius ladder were below the random graph and the honeycomb.
- Strict global selection (large `σ`) lowered cooperation. Under DB, cooperation was highest at `σ` = 0.
- Stronger within-group selection lowered cooperation. More fission randomness helped when within-group selection was strong and hurt when it was weak.
- More groups gave more cooperation.

## Limitations

- The main runs did not test random neighbor choice (`local="random"`) with `σ` > 0, so the base paper's claim that local selection is needed is not tested here.
- We used 200 groups and `T` = 1500, against 1000 groups and `T` = 50 000 in the base paper.
- With 10 bins, final values cluster about 0.1 apart, and some runs were still changing near the end.

## References

1. Y. Ispolatov, B. Simon, and M. Doebeli, "Evolution of cooperation in spatially structured group-selection models of the continuous Prisoner's Dilemma," 2026, arXiv:2606.03071.
2. H. Ohtsuki, C. Hauert, E. Lieberman, and M. A. Nowak, "A simple rule for the evolution of cooperation on graphs and social networks," *Nature*, vol. 441, pp. 502–505, 2006.
3. B. Simon, Y. Ispolatov, and M. Doebeli, "Fission as a source of variation for group selection," *Evolution*, vol. 78, no. 9, pp. 1583–1593, 2024.

The full reference list is in the report.
