"""
gsim.py  --  spatial multi-level continuous Prisoner's Dilemma on k-regular graphs.

Hybrid method (Ispolatov, Simon & Doebeli 2026):
  * inside each group: deterministic dynamics of the trait distribution p_k (fractions in m bins)
  * between groups: discrete fission/extinction events driven by group payoffs on a graph

Key simplification (derived in the plan): because the mean per-capita birth rate is fixed at beta,
group size obeys d(eta)/dt = eta*(beta - delta*eta) independently of the traits, so size sits at
eta* = beta/delta and the composition p_k = n_k/eta obeys
    dp_k/dt = beta * p_k * (exp(s*psi_k)/Z - 1) + (R/h^2) * Lap(p)_k ,   Z = sum_j p_j exp(s*psi_j)
    psi_k   = B*xbar - C*x_k
"""
import time
from dataclasses import dataclass, asdict
import numpy as np
import networkx as nx


def fmt_time(sec):
    """Seconds -> '1h02m03s' / '4m05s' / '12.3s'."""
    if sec is None or not np.isfinite(sec):
        return "?"
    if sec < 60:
        return f"{sec:.1f}s"
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, sc = divmod(rem, 60)
    return f"{h}h{m:02d}m{sc:02d}s" if h else f"{m}m{sc:02d}s"


# ----------------------------------------------------------------------------- parameters
@dataclass
class Params:
    B: float = 2.5          # benefit
    C: float = 1.0          # cost
    beta: float = 1000.0    # birth-rate scale
    delta: float = 10.0     # death-rate scale  (group size = beta/delta)
    s: float = 5e-4         # within-group selection strength (s=0 -> no within-group selection)
    R: float = 1e-6         # mutation (diffusion) coefficient
    r: float = 1.0          # fission randomisation
    sigma: float = 1.0      # global (softmax) selection strength
    rule: str = "BD"        # "BD" or "DB"
    local: str = "select"   # "select" (best/worst neighbour) or "random" (Bd / Db)
    m: int = 10             # trait bins
    x_mode: str = "centers" # "centers": (k+.5)/m   |  "grid": k/(m-1)
    G: int = 200            # number of groups
    topology: str = "rrg"   # rrg | ring | prism | mobius | honeycomb | rrg4 | torus
    k: int = 3              # degree (used by rrg only)
    T: float = 3000.0       # simulated time (each group updated ~once per time unit)
    dt: float = 0.05        # integration step of the within-group dynamics
    rec: float = 10.0       # recording interval
    events: bool = True     # False -> switch group-level events off (baseline check)
    seed: int = 0


# ----------------------------------------------------------------------------- topologies
def make_graph(kind, G, k=3, seed=0):
    """Return a networkx graph with nodes 0..G-1. All 'k=3' kinds are exactly 3-regular."""
    if kind == "rrg":                                   # random k-regular graph (needs G*k even)
        g = nx.random_regular_graph(k, G, seed=seed)
        while not nx.is_connected(g):
            seed += 1000
            g = nx.random_regular_graph(k, G, seed=seed)
    elif kind == "rrg4":
        g = nx.random_regular_graph(4, G, seed=seed)
    elif kind == "ring":                                # paper's baseline, k = 2
        g = nx.cycle_graph(G)
    elif kind == "prism":                               # circular ladder, 3-regular, G even
        g = nx.circular_ladder_graph(G // 2)
    elif kind == "mobius":                              # Moebius ladder, 3-regular, G even
        g = nx.circulant_graph(G, [1, G // 2])
    elif kind == "honeycomb":                           # periodic hexagonal lattice, 3-regular
        rows = max(2, int(round(np.sqrt(G / 2) / 2)) * 2)   # rows, cols even
        g = nx.hexagonal_lattice_graph(rows, rows, periodic=True)
    elif kind == "torus":                               # 4-regular square lattice (paper's 2-D case)
        L = int(round(np.sqrt(G)))
        g = nx.grid_2d_graph(L, L, periodic=True)
    else:
        raise ValueError(kind)
    g = nx.convert_node_labels_to_integers(g)
    deg = {d for _, d in g.degree()}
    assert len(deg) == 1, f"graph not regular: degrees {deg}"
    return g


def neighbour_array(g):
    k = next(iter(dict(g.degree()).values()))
    return np.array([sorted(g.neighbors(i)) for i in range(g.number_of_nodes())], dtype=np.int64), k


# ----------------------------------------------------------------------------- model pieces
def trait_axis(m, mode):
    return (np.arange(m) + 0.5) / m if mode == "centers" else np.linspace(0, 1, m)


def integrate(p, xs, P, dt):
    """One explicit Euler step of the within-group dynamics for all groups at once. p: (G, m)."""
    xbar = p @ xs
    psi = P.B * xbar[:, None] - P.C * xs[None, :]
    w = np.exp(P.s * (psi - psi.max(axis=1, keepdims=True)))      # shift cancels in the ratio
    ratio = w / (p * w).sum(axis=1, keepdims=True)
    dp = P.beta * p * (ratio - 1.0)
    if P.R > 0:
        h2 = (xs[1] - xs[0]) ** 2
        lap = np.empty_like(p)
        lap[:, 1:-1] = p[:, :-2] - 2 * p[:, 1:-1] + p[:, 2:]
        lap[:, 0] = p[:, 1] - p[:, 0]                                # reflecting ends
        lap[:, -1] = p[:, -2] - p[:, -1]
        dp += (P.R / h2) * lap
    p += dt * dp
    np.clip(p, 0.0, None, out=p)
    p /= p.sum(axis=1, keepdims=True)


def fission(p_parent, r, eta, rng):
    """Split a parent composition into two daughters (eq. 8): w1 ~ N(w/2, r^2 w/4) clipped to [0, w]."""
    w = eta * p_parent
    w1 = np.clip(rng.normal(w / 2, r * np.sqrt(w) / 2), 0.0, w)
    w2 = w - w1
    s1, s2 = w1.sum(), w2.sum()
    d1 = w1 / s1 if s1 > 0 else p_parent.copy()
    d2 = w2 / s2 if s2 > 0 else p_parent.copy()
    return d1, d2


def softmax_pick(pay, sigma, sign, rng):
    """Index drawn with prob ~ exp(sign*sigma*pay). sign=+1 (BD: good groups), -1 (DB: bad groups)."""
    if sigma == 0:
        return int(rng.integers(len(pay)))
    z = sign * sigma * pay
    w = np.exp(z - z.max())
    c = np.cumsum(w)
    return int(min(np.searchsorted(c, rng.random() * c[-1]), len(pay) - 1))


def pick_local(nb, pay_nb, want, mode, rng):
    """Choose among neighbours: 'min' (worst) or 'max' (best) payoff, random tie-break; or random."""
    if mode == "random":
        return int(nb[rng.integers(len(nb))])
    target = pay_nb.min() if want == "min" else pay_nb.max()
    cand = nb[np.isclose(pay_nb, target, rtol=0, atol=1e-12)]
    return int(cand[rng.integers(len(cand))])


# ----------------------------------------------------------------------------- simulation
def simulate(P: Params, return_state=False, progress=0):
    """Run one simulation. progress = number of progress lines to print during the run (0 = silent).
    Each progress line shows simulated time, current mean cooperation, elapsed wall time and ETA.
    The returned dict always contains 'runtime_s' (wall-clock seconds for this run)."""
    t_start = time.time()
    rng = np.random.default_rng(P.seed)
    g = make_graph(P.topology, P.G, P.k, seed=P.seed)
    nbr, k = neighbour_array(g)
    G = g.number_of_nodes()
    xs = trait_axis(P.m, P.x_mode)
    eta = P.beta / P.delta

    p = rng.random((G, P.m)) + 1e-3
    p /= p.sum(axis=1, keepdims=True)
    xbar = p @ xs
    pay = P.B * xbar[nbr].mean(axis=1) - P.C * xbar

    edges = np.array(g.edges())

    def stats(t):
        xb = xbar
        mu, var = xb.mean(), xb.var()
        corr = np.mean((xb[edges[:, 0]] - mu) * (xb[edges[:, 1]] - mu)) / var if var > 1e-14 else np.nan
        within = ((p * xs ** 2).sum(1) - xb ** 2).mean()
        return (t, mu, var, corr, within)

    rows, t, next_rec, ev_acc = [stats(0.0)], 0.0, P.rec, 0.0
    n_steps = int(round(P.T / P.dt))
    sign = +1 if P.rule == "BD" else -1

    print_every = max(1, n_steps // progress) if progress else 0
    for step in range(n_steps):
        integrate(p, xs, P, P.dt)
        t += P.dt
        xbar = p @ xs
        pay = P.B * xbar[nbr].mean(axis=1) - P.C * xbar

        if P.events:
            ev_acc += G * P.dt
            n_ev = int(ev_acc)
            ev_acc -= n_ev
            for _e in range(n_ev):
                i = softmax_pick(pay, P.sigma, sign, rng)
                if P.rule == "BD":      # i fissions; a daughter replaces its (worst) neighbour j
                    j = pick_local(nbr[i], pay[nbr[i]], "min", P.local, rng)
                    p[i], p[j] = fission(p[i], P.r, eta, rng)
                else:                   # DB: i dies; its (best) neighbour j fissions into j and i
                    j = pick_local(nbr[i], pay[nbr[i]], "max", P.local, rng)
                    p[j], p[i] = fission(p[j], P.r, eta, rng)
                # incremental update of xbar / payoffs around the two changed groups
                xbar[i], xbar[j] = p[i] @ xs, p[j] @ xs
                aff = np.unique(np.concatenate(([i, j], nbr[i], nbr[j])))
                pay[aff] = P.B * xbar[nbr[aff]].mean(axis=1) - P.C * xbar[aff]

        if t >= next_rec - 1e-9:
            rows.append(stats(t))
            next_rec += P.rec

        if print_every and (step + 1) % print_every == 0:
            frac = (step + 1) / n_steps
            el = time.time() - t_start
            print(f"    [{P.topology} {P.rule} sigma={P.sigma:g} B={P.B:g}] t={t:.0f}/{P.T:.0f} ({frac:.0%})"
                  f"  mean coop={xbar.mean():.3f}  elapsed {fmt_time(el)}  ETA {fmt_time(el / frac - el)}",
                  flush=True)

    arr = np.array(rows)
    out = dict(t=arr[:, 0], mean=arr[:, 1], var_between=arr[:, 2], corr=arr[:, 3], var_within=arr[:, 4],
               params=asdict(P))
    tail = arr[:, 0] >= 0.8 * P.T                       # steady state = last 20 % of the run
    out["final_mean"] = float(arr[tail, 1].mean())
    out["final_corr"] = float(np.nanmean(arr[tail, 3]))
    out["final_var_between"] = float(arr[tail, 2].mean())
    out["runtime_s"] = time.time() - t_start
    if return_state:
        out.update(p=p.copy(), xbar=xbar.copy(), graph=g, xs=xs)
    return out
