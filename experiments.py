"""
experiments.py -- sanity checks, parameter sweeps and figures, with live progress reporting.

Every finished run prints one line:  its parameters, its result, how long it took, elapsed time,
the ETA of the current sweep and the ETA of the whole job.  At the end of each sweep a summary table is printed.
Partial results are saved if you press Ctrl-C.

Usage (from the folder containing gsim.py):
    python experiments.py check                    # validation against the paper's appendix + neutral drift
    python experiments.py e2 --quick               # topology x B/C   (quick = tiny, for debugging)
    python experiments.py e2 --G 200 --T 1500 --reps 8 --jobs 8
    python experiments.py all --G 200 --T 1500 --reps 8 --jobs 8
Experiments: e2 topology x B | e3 sigma sweep | e4 (s,r) heatmap | e5 degree sweep | e6 finite size | snap
Needs joblib >= 1.3 (results are streamed as soon as each run finishes).

Checkpoint / resume: every finished run is appended to results/<experiment>.csv immediately. If you start the
same command again (after a crash, Ctrl-C or reboot), runs whose parameters+seed are already in that CSV are skipped
and only the missing ones are executed. Changing any parameter (G, T, grids ...) or raising --reps simply makes
the new combinations count as 'missing'; old rows with other parameters stay in the file but are ignored.
"""
import argparse, csv, io, itertools, os, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from joblib import Parallel, delayed
from gsim import Params, simulate, integrate, trait_axis, fmt_time

RES, FIG = "results", "figures"
for d in (RES, FIG):
    os.makedirs(d, exist_ok=True)

LABEL = {"ring": "ring (k=2)", "rrg": "random 3-regular", "prism": "prism / circular ladder (k=3)",
         "mobius": "Moebius ladder (k=3)", "honeycomb": "honeycomb (k=3)", "rrg4": "random 4-regular",
         "torus": "square torus (k=4)"}
RESULT_COLS = ["final_mean", "final_corr", "final_var_between", "runtime_s"]


# ------------------------------------------------------------------ global progress tracker
class Tracker:
    """Counts runs over the whole job so that a total ETA can be shown. dry=True only counts."""
    def __init__(self):
        self.dry, self.total, self.done, self.t0 = False, 0, 0, time.time()

TR = Tracker()


def _grid_runs(grid, base, reps):
    keys = list(grid)
    return keys, [{**base, **dict(zip(keys, v)), "seed": rep}
                  for v in itertools.product(*grid.values()) for rep in range(reps)]


def _run(kw):
    o = simulate(Params(**kw))                      # silent inside workers
    return {**kw, **{c: o[c] for c in RESULT_COLS}}


def _norm(v):
    """Canonical form of a parameter value, so values survive a CSV round trip (2.5 == '2.5', 1e-4 == 0.0001)."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(f"{float(v):.12g}")
    return str(v)


def _key(row, cols):
    return tuple(_norm(row[c]) for c in cols)


def _read_existing(path, cols):
    """Read results/<name>.csv for resuming. Returns (DataFrame or None, status) with status in
    'none' (no usable file) | 'ok' | 'mismatch' (columns differ from this experiment -> cannot be reused).
    A half-written last line (killed mid-write) is ignored."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None, "none"
    with open(path, newline="") as f:
        txt = f.read()
    if not txt.endswith("\n"):
        txt = txt[:txt.rfind("\n") + 1]
    if not txt.strip():
        return None, "none"
    try:
        df = pd.read_csv(io.StringIO(txt), on_bad_lines="skip")
    except Exception:
        return None, "mismatch"
    if set(df.columns) != set(cols + RESULT_COLS):
        return None, "mismatch"
    return df.dropna(subset=cols), "ok"


def _repair_tail(path):
    """Physically cut a half-written last line so that appending starts on a clean line."""
    with open(path, "rb") as f:
        data = f.read()
    if data and not data.endswith(b"\n"):
        with open(path, "wb") as f:
            f.write(data[:data.rfind(b"\n") + 1])


def sweep(name, grid, base, reps, jobs):
    keys, todo = _grid_runs(grid, base, reps)
    cols = list(todo[0])
    path = f"{RES}/{name}.csv"
    todo_keys = [_key(kw, cols) for kw in todo]
    todo_set = set(todo_keys)

    # ---- which (parameters + seed) combinations are already on disk?
    old, status = _read_existing(path, cols)
    done = {_key(r, cols) for r in old.to_dict("records")} if status == "ok" else set()
    remaining = [kw for kw, k in zip(todo, todo_keys) if k not in done]

    if TR.dry:                                      # planning pass: count only the runs still to do
        TR.total += len(remaining)
        return pd.DataFrame(columns=cols + RESULT_COLS)

    n_tot, n_skip = len(todo), len(todo) - len(remaining)
    if status == "mismatch":                        # old file has other columns: keep it, start a fresh one
        bak = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        os.replace(path, bak)
        print(f"!! {path} has different columns than this experiment -> moved to {bak}, starting a new file", flush=True)
    elif status == "ok":
        _repair_tail(path)
    fields = list(old.columns) if status == "ok" else cols + RESULT_COLS

    print(f"\n=== {name}: {n_tot} runs | {n_skip} already in {path} (skipped), {len(remaining)} to run "
          f"| job so far {TR.done}/{TR.total} ===", flush=True)

    t0, n = time.time(), 0
    if remaining:
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        f = open(path, "a", newline="")
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader(); f.flush()
        gen = Parallel(n_jobs=jobs, return_as="generator_unordered")(delayed(_run)(kw) for kw in remaining)
        try:
            for res in gen:
                w.writerow({c: res[c] for c in fields})      # checkpoint: on disk before anything else
                f.flush(); os.fsync(f.fileno())
                n += 1
                TR.done += 1
                el = time.time() - t0
                eta_sweep = el / n * (len(remaining) - n)
                eta_total = max(0.0, (time.time() - TR.t0) / TR.done * (TR.total - TR.done))
                lab = " ".join(f"{k}={res[k]:g}" if isinstance(res[k], (int, float)) else f"{k}={res[k]}" for k in keys)
                print(f"[{name} {n_skip + n}/{n_tot}] {lab} seed={res['seed']} | coop={res['final_mean']:.3f} "
                      f"corr={res['final_corr']:.3f} var_between={res['final_var_between']:.4f} | "
                      f"run {fmt_time(res['runtime_s'])} | elapsed {fmt_time(el)} | sweep ETA {fmt_time(eta_sweep)} | "
                      f"job {TR.done}/{TR.total} ETA {fmt_time(eta_total)}", flush=True)
        except KeyboardInterrupt:
            print(f"\n!! interrupted after {n_skip + n}/{n_tot} runs -- all finished runs are saved in {path}; "
                  f"rerun the same command to resume", flush=True)
        finally:
            f.close()
    else:
        print(f"--- {name}: nothing to run, all results already on disk", flush=True)

    # ---- the data of THIS experiment = rows of the CSV that belong to the current parameter grid
    all_df, st = _read_existing(path, cols)
    if st == "ok" and len(all_df):
        all_df = all_df.copy()
        all_df["_k"] = [_key(r, cols) for r in all_df.to_dict("records")]
        df = (all_df[all_df["_k"].isin(todo_set)].drop_duplicates("_k", keep="last")
              .drop(columns="_k").reset_index(drop=True))
    else:
        df = pd.DataFrame(columns=cols + RESULT_COLS)

    wall = time.time() - t0
    cpu = df["runtime_s"].sum() if len(df) else 0.0
    print(f"--- {name}: {len(df)}/{n_tot} runs complete | this session: {n} runs in {fmt_time(wall)} | "
          f"mean per run {fmt_time(cpu / max(len(df), 1))} -> {path}")
    if len(df):
        agg = df.groupby(keys)[["final_mean", "final_corr"]].agg(["mean", "std"])
        agg.columns = ["coop_mean", "coop_sd", "corr_mean", "corr_sd"]
        agg["n"] = df.groupby(keys).size()
        print(agg.to_string(float_format=lambda v: f"{v:.3f}"), flush=True)
    return df


# ------------------------------------------------------------------ plotting
def lines(df, x, y, hue, fname, xlabel=None, ylabel=None, title=None, label_map=None):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for h, d in df.groupby(hue):
        g = d.groupby(x)[y].agg(["mean", "std"]).reset_index()
        ax.plot(g[x], g["mean"], marker="o", ms=4, label=(label_map or {}).get(h, str(h)))
        ax.fill_between(g[x], g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.15)
    ax.set_xlabel(xlabel or x); ax.set_ylabel(ylabel or y); ax.set_title(title or "")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{FIG}/{fname}.png", dpi=200); plt.close(fig)
    print(f"    saved {FIG}/{fname}.png")


def heat(df, x, y, z, fname, title=""):
    if df.empty:
        return
    piv = df.groupby([y, x])[z].mean().unstack(x)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    im = ax.imshow(piv.values, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{v:g}" for v in piv.columns], rotation=45)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{v:g}" for v in piv.index])
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title)
    fig.colorbar(im, label="steady-state mean cooperation"); fig.tight_layout()
    fig.savefig(f"{FIG}/{fname}.png", dpi=200); plt.close(fig)
    print(f"    saved {FIG}/{fname}.png")


# ------------------------------------------------------------------ sanity checks
def check():
    """(1) Within-group-only rate vs appendix eq. A.5; (2) cooperation decays; (3) neutral drift."""
    t0 = time.time()
    P = Params(s=1e-3, R=0.0)
    xs = trait_axis(P.m, P.x_mode)
    rng = np.random.default_rng(1)
    p = rng.random((1, P.m)); p /= p.sum()
    xb = p @ xs; var = (p * xs ** 2).sum() - xb ** 2
    pred = -P.beta * P.s * P.C * var / (1 + P.s * xb * (P.B - P.C))      # eq. A.5 (weak selection)
    q = p.copy(); integrate(q, xs, P, 1e-4)
    num = ((q @ xs) - xb) / 1e-4
    print(f"[check 1/3] d xbar/dt  numerical = {num[0]:.5f}   appendix formula = {pred[0]:.5f}"
          f"   ({fmt_time(time.time() - t0)})", flush=True)
    t1 = time.time()
    o = simulate(Params(events=False, s=1e-3, T=100, G=20))
    print(f"[check 2/3] within-group only: mean cooperation {o['mean'][0]:.3f} -> {o['mean'][-1]:.3f} "
          f"(should fall to ~floor)   ({fmt_time(o['runtime_s'])})", flush=True)
    ms = []
    for s in range(3):
        o = simulate(Params(topology="rrg", local="random", s=0, sigma=0, G=200, T=300, seed=s))
        ms.append(o["final_mean"])
        print(f"[check 3/3] neutral Bd run {s + 1}/3: final mean {o['final_mean']:.3f}  "
              f"({fmt_time(o['runtime_s'])}; ETA for this check {fmt_time((time.time() - t1) / (s + 1) * (2 - s))})",
              flush=True)
    print(f"            neutral drift gives different means per seed {np.round(ms, 2)} -> no systematic push, so always use replicates")


# ------------------------------------------------------------------ experiments
def e2(a):   # topology x B/C  (main novelty figure)
    df = sweep("e2_topology_B", {"topology": ["ring", "rrg", "prism", "mobius", "honeycomb", "torus"],
                                 "B": a.Bgrid}, dict(rule="BD", sigma=1.0, s=5e-4, r=2.0, **a.base), a.reps, a.jobs)
    lines(df, "B", "final_mean", "topology", "fig_e2_topology_vs_B", "benefit B (C=1)", "steady-state cooperation",
          "BD update: cooperation vs benefit/cost by topology", LABEL)
    lines(df, "B", "final_corr", "topology", "fig_e2_corr_vs_B", "benefit B (C=1)", "neighbour correlation",
          label_map=LABEL)


def e3(a):   # global selection strength sigma, BD and DB, k=3 vs ring
    df = sweep("e3_sigma", {"topology": ["ring", "rrg", "honeycomb"], "rule": ["BD", "DB"], "sigma": a.sgrid},
               dict(B=a.Bfix, s=5e-4, r=2.0, **a.base), a.reps, a.jobs)
    for rule, d in df.groupby("rule"):
        lines(d, "sigma", "final_mean", "topology", f"fig_e3_sigma_{rule}", "global selection sigma",
              "steady-state cooperation", f"{rule} update", LABEL)
        lines(d, "sigma", "final_corr", "topology", f"fig_e3_corr_{rule}", "global selection sigma",
              "neighbour correlation", f"{rule} update", LABEL)


def e4(a):   # (s, r) phase diagram, random 3-regular vs ring
    for top in ("rrg", "ring"):
        df = sweep(f"e4_sr_{top}", {"s": a.s_grid, "r": a.r_grid},
                   dict(topology=top, B=a.Bfix, rule="BD", sigma=1.0, **a.base), a.reps, a.jobs)
        heat(df, "s", "r", "final_mean", f"fig_e4_heat_{top}", f"{LABEL[top]}: cooperation")


def e5(a):   # degree sweep -> critical B(k)
    df = sweep("e5_degree", {"k": [2, 3, 4, 5, 6], "B": a.Bgrid},
               dict(topology="rrg", rule="BD", sigma=1.0, s=5e-4, r=2.0, **a.base), a.reps, a.jobs)
    lines(df, "B", "final_mean", "k", "fig_e5_degree", "benefit B (C=1)", "steady-state cooperation",
          "random k-regular graphs: threshold moves with degree")


def e6(a):   # finite-size check
    df = sweep("e6_size", {"G": [100, 200, 400], "B": a.Bgrid},
               dict(topology="rrg", rule="BD", sigma=1.0, s=5e-4, r=2.0, **{k: v for k, v in a.base.items() if k != "G"}),
               a.reps, a.jobs)
    lines(df, "B", "final_mean", "G", "fig_e6_size", "benefit B (C=1)", "steady-state cooperation")


def snap(a):  # time series + graph snapshot (two single runs, with in-run progress)
    if TR.dry:
        TR.total += 2
        return
    print("\n=== snap: 2 single runs with in-run progress ===", flush=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, top in zip(axs[:2], ["rrg", "honeycomb"]):
        o = simulate(Params(topology=top, B=a.Bfix, rule="BD", sigma=1.0, s=5e-4, r=2.0, **a.base),
                     return_state=True, progress=5)
        TR.done += 1
        el = time.time() - TR.t0
        print(f"[snap {top}] final coop={o['final_mean']:.3f} corr={o['final_corr']:.3f} | run {fmt_time(o['runtime_s'])}"
              f" | job {TR.done}/{TR.total} ETA {fmt_time(el / TR.done * (TR.total - TR.done))}", flush=True)
        g = o["graph"]
        pos = nx.spring_layout(g, seed=1, iterations=100) if top == "rrg" else nx.kamada_kawai_layout(g)
        nx.draw(g, pos, ax=ax, node_color=o["xbar"], cmap="viridis", vmin=0, vmax=1, node_size=40, width=0.4)
        ax.set_title(f"{LABEL[top]}: group cooperation at end")
        axs[2].plot(o["t"], o["mean"], label=LABEL[top])
    axs[2].set_xlabel("time"); axs[2].set_ylabel("mean cooperation"); axs[2].legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_snapshots.png", dpi=200); plt.close(fig)
    print(f"    saved {FIG}/fig_snapshots.png")


# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check", "e2", "e3", "e4", "e5", "e6", "snap", "all"])
    ap.add_argument("--G", type=int, default=200); ap.add_argument("--T", type=float, default=1500)
    ap.add_argument("--reps", type=int, default=8); ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.G, a.T, a.reps = 60, 150, 1
    a.base = dict(G=a.G, T=a.T)
    a.Bgrid = [1.5, 2.5, 3.5, 4.5, 6.0] if not a.quick else [2.5, 4.5]
    a.sgrid = [0, 0.5, 1, 2, 4] if not a.quick else [0, 2]
    a.s_grid = [1e-4, 3e-4, 1e-3, 3e-3] if not a.quick else [1e-4, 1e-3]
    a.r_grid = [0.5, 1, 2, 5] if not a.quick else [1, 5]
    a.Bfix = 4.5                      # set after looking at e2 (a B where k=3 cooperation exists but is not saturated)

    funcs = {"check": lambda _: check(), "e2": e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6, "snap": snap}
    todo = ["check", "e2", "e3", "e4", "e5", "e6", "snap"] if a.what == "all" else [a.what]

    TR.dry = True                      # planning pass: count the runs so a job-wide ETA is possible
    for w in todo:
        if w != "check":
            funcs[w](a)
    TR.dry = False
    if TR.total:
        print(f"Planned: {TR.total} simulation runs (G={a.G}, T={a.T:g}, reps={a.reps}, jobs={a.jobs}). "
              f"ETA appears after the first run finishes and is rough until a few runs are done.", flush=True)
    t_all = TR.t0 = time.time()
    for w in todo:
        funcs[w](a)
        if w == "check":               # the check is not part of the counted runs, so restart the ETA clock
            TR.t0 = time.time()
    print(f"\nAll done in {fmt_time(time.time() - t_all)}.", flush=True)
