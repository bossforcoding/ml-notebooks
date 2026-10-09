"""Discrete Bayesian networks with pandas and NumPy.

Structure scoring (BIC), hill-climbing structure learning, Bayesian parameter
estimation with a Dirichlet prior, exact inference on the full joint table and
forward (logic) sampling. Meant for networks with a handful of variables, where
the joint distribution fits comfortably in memory.
"""

import itertools

import networkx as nx
import numpy as np
import pandas as pd


def family_bic(data, child, parents, states):
    """BIC contribution of one node: maximized log-likelihood minus a complexity penalty."""
    n = len(data)
    r = len(states[child])
    q = int(np.prod([len(states[p]) for p in parents])) if parents else 1
    if parents:
        counts = data.groupby(list(parents) + [child], observed=True).size()
        parent_totals = counts.groupby(level=list(range(len(parents))), observed=True).transform("sum")
        loglik = float((counts * np.log(counts / parent_totals)).sum())
    else:
        counts = data[child].value_counts()
        loglik = float((counts * np.log(counts / n)).sum())
    return loglik - 0.5 * np.log(n) * q * (r - 1)


def bic(data, dag, states):
    return sum(family_bic(data, v, sorted(dag.predecessors(v)), states) for v in dag.nodes)


def hill_climbing(data, states, max_parents=3, start=None, verbose=False):
    """Greedy search over DAGs: add, remove or reverse one edge at a time while BIC improves."""
    dag = start.copy() if start is not None else nx.DiGraph()
    dag.add_nodes_from(states)
    cache = {}

    def score(v, parents):
        key = (v, tuple(sorted(parents)))
        if key not in cache:
            cache[key] = family_bic(data, v, sorted(parents), states)
        return cache[key]

    current = {v: score(v, dag.predecessors(v)) for v in dag.nodes}
    while True:
        best_gain, best_move = 1e-9, None
        for u, v in itertools.permutations(dag.nodes, 2):
            pv = set(dag.predecessors(v))
            if dag.has_edge(u, v):
                gain = score(v, pv - {u}) - current[v]                       # remove u -> v
                if gain > best_gain:
                    best_gain, best_move = gain, ("remove", u, v)
                pu = set(dag.predecessors(u))                                  # reverse u -> v
                if len(pu) < max_parents:
                    g = dag.copy(); g.remove_edge(u, v); g.add_edge(v, u)
                    if nx.is_directed_acyclic_graph(g):
                        gain = score(v, pv - {u}) - current[v] + score(u, pu | {v}) - current[u]
                        if gain > best_gain:
                            best_gain, best_move = gain, ("reverse", u, v)
            elif not dag.has_edge(v, u) and len(pv) < max_parents:              # add u -> v
                if not nx.has_path(dag, v, u):
                    gain = score(v, pv | {u}) - current[v]
                    if gain > best_gain:
                        best_gain, best_move = gain, ("add", u, v)
        if best_move is None:
            return dag
        op, u, v = best_move
        if op == "add":
            dag.add_edge(u, v)
        elif op == "remove":
            dag.remove_edge(u, v)
        else:
            dag.remove_edge(u, v)
            dag.add_edge(v, u)
            current[u] = score(u, dag.predecessors(u))
        current[v] = score(v, dag.predecessors(v))
        if verbose:
            print(f"{op:<8} {u} -> {v}   (+{best_gain:.1f})")


class DiscreteBN:
    """A fitted discrete Bayesian network with exact inference over the joint table."""

    def __init__(self, dag, states):
        self.dag = dag
        self.states = states
        self.order = list(nx.topological_sort(dag))

    def fit(self, data, prior_count=1.0):
        """Posterior mean CPDs under a uniform Dirichlet prior with `prior_count` pseudo-counts per configuration."""
        self.cpds = {}
        for v in self.order:
            parents = sorted(self.dag.predecessors(v))
            shape = [len(self.states[p]) for p in parents] + [len(self.states[v])]
            counts = np.zeros(shape)
            idx = [data[c].map({s: i for i, s in enumerate(self.states[c])}).to_numpy() for c in parents + [v]]
            np.add.at(counts, tuple(idx), 1)
            alpha = prior_count / np.prod(shape)
            cpd = (counts + alpha) / (counts + alpha).sum(-1, keepdims=True)
            self.cpds[v] = (parents, cpd)
        self._build_joint()
        return self

    def _build_joint(self):
        names = list(self.states)
        letters = dict(zip(names, "abcdefghijklmnopqrstuvwxyz"))
        terms, operands = [], []
        for v, (parents, cpd) in self.cpds.items():
            terms.append("".join(letters[p] for p in parents) + letters[v])
            operands.append(cpd)
        self.names = names
        self.joint = np.einsum(",".join(terms) + "->" + "".join(letters[n] for n in names), *operands)

    def query(self, target, evidence=None):
        """P(target | evidence) by summing the joint table."""
        table = self.joint
        for var, value in (evidence or {}).items():
            axis = self.names.index(var)
            mask = np.zeros(len(self.states[var]))
            mask[self.states[var].index(value)] = 1
            shape = [1] * table.ndim
            shape[axis] = -1
            table = table * mask.reshape(shape)
        axis = self.names.index(target)
        marginal = table.sum(axis=tuple(i for i in range(table.ndim) if i != axis))
        return pd.Series(marginal / marginal.sum(), index=self.states[target], name=target)

    def sample(self, n, seed=0):
        """Forward (ancestral) sampling in topological order."""
        rng = np.random.default_rng(seed)
        out = {}
        for v in self.order:
            parents, cpd = self.cpds[v]
            probs = cpd[tuple(out[p] for p in parents)] if parents else np.broadcast_to(cpd, (n, cpd.shape[-1]))
            u = rng.random((n, 1))
            out[v] = (probs.cumsum(-1) < u).sum(-1).clip(max=len(self.states[v]) - 1)
        return pd.DataFrame({v: np.array(self.states[v])[out[v]] for v in self.order})

    def logic_sampling_query(self, target, evidence, n, seed=0):
        """Approximate P(target | evidence) by rejection: keep the samples that match the evidence."""
        s = self.sample(n, seed)
        keep = np.all([s[k] == v for k, v in evidence.items()], axis=0)
        return s.loc[keep, target].value_counts(normalize=True).reindex(self.states[target], fill_value=0.0), int(keep.sum())
