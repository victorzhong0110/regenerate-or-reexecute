"""Kill test for regenerate-vs-reexecute. Thresholds fixed before any run.

Decision rule, fixed before looking at results:
- Each correct program is run 30 times. Compare its sample sd (ddof=1) with ε.
- If every correct-program sd is < ε, the topic ends.
- If a correct-program sd is ≥ ε, compare two policies on that correct program,
  200 macro-replications:
  internal = raise that program's own repetition count until the 95% half-width
             is ≤ ε, then return the pooled estimate.
  external = re-execute the default one-shot program until the 95% half-width
             of the replications' mean is ≤ ε, then return that mean.
- False accept = |estimate - task truth| > ε.
- The topic ends unless, on a correct program, internal's false-accept count
  exceeds external's by more than 10 out of 200. A gap of 10 or fewer is a tie.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

N_RUNS = 30
N_POLICY = 200
TIE_COUNT = 10
MAX_REPS = 5000

# Tolerances committed before execution. They are absolute errors on the
# number the program is asked to report.
EPS = {
    "bootstrap_se": 0.01,  # SE of a mean, in outcome units
    "permutation_p": 0.01,  # two-sided permutation p-value
    "random_split_acc": 0.02,  # test accuracy of one 70/30 split
}


def halfwidth(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.inf
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def summarize_runs(values: np.ndarray, truth: float, eps: float) -> dict:
    sd = float(np.std(values, ddof=1))
    # Approximate SE of a sample sd, used only as a stability note.
    se_sd = sd / np.sqrt(2 * (len(values) - 1))
    mean = float(np.mean(values))
    return {
        "n": int(len(values)),
        "mean": mean,
        "sd": sd,
        "se_sd": float(se_sd),
        "abs_bias": float(abs(mean - truth)),
        "sd_below_eps": bool(sd < eps),
        "bias_above_eps": bool(abs(mean - truth) > eps),
    }


@dataclass
class Task:
    name: str
    eps: float
    truth: float
    truth_note: str


def bootstrap_task():
    rng = np.random.default_rng(11)
    x = rng.normal(5.0, 2.0, size=40)
    mu2 = float(np.mean((x - x.mean()) ** 2))
    truth = float(np.sqrt(mu2 / x.size))
    task = Task(
        "bootstrap_se",
        EPS["bootstrap_se"],
        truth,
        "sd of the bootstrap mean under the empirical distribution",
    )

    def one_shot(seed: int, b: int = 1000) -> float:
        rs = np.random.default_rng(seed)
        idx = rs.integers(0, x.size, size=(b, x.size))
        means = x[idx].mean(axis=1)
        return float(np.std(means, ddof=1))

    def wrong_shot(seed: int, b: int = 1000) -> float:
        # Reports the Monte Carlo error of the bootstrap mean, not the SE.
        rs = np.random.default_rng(seed)
        idx = rs.integers(0, x.size, size=(b, x.size))
        means = x[idx].mean(axis=1)
        return float(np.std(means, ddof=1) / np.sqrt(b))

    def internal(seed: int) -> tuple[float, int]:
        rs = np.random.default_rng(seed)
        chunks = []
        total_b = 0
        estimate = np.nan
        for _ in range(MAX_REPS):
            idx = rs.integers(0, x.size, size=(1000, x.size))
            chunks.append(x[idx].mean(axis=1))
            total_b += 1000
            means = np.concatenate(chunks)
            estimate = float(np.std(means, ddof=1))
            mc_half = 1.96 * estimate / np.sqrt(2 * total_b)
            if mc_half <= task.eps:
                return estimate, total_b
        return float(estimate), total_b

    def external(seed: int) -> tuple[float, int]:
        rs = np.random.default_rng(seed)
        vals = []
        for k in range(1, MAX_REPS + 1):
            vals.append(one_shot(int(rs.integers(0, 2**31 - 1))))
            if halfwidth(np.asarray(vals)) <= task.eps:
                return float(np.mean(vals)), k
        return float(np.mean(vals)), len(vals)

    return task, one_shot, wrong_shot, internal, external


def permutation_task():
    rng = np.random.default_rng(12)
    a = rng.normal(0.0, 1.0, size=15)
    b = rng.normal(0.5, 1.0, size=15)
    pooled = np.concatenate([a, b])
    obs = float(abs(a.mean() - b.mean()))
    obs_signed = float(b.mean() - a.mean())

    def counts(n_perm: int, seed: int, sided: str) -> int:
        rs = np.random.default_rng(seed)
        got = 0
        left = n_perm
        while left:
            m = min(20_000, left)
            order = rs.random((m, pooled.size)).argsort(axis=1)
            samp = pooled[order]
            diff = samp[:, : a.size].mean(axis=1) - samp[:, a.size :].mean(axis=1)
            if sided == "two":
                got += int(np.count_nonzero(np.abs(diff) >= obs - 1e-12))
            else:
                got += int(np.count_nonzero((-diff) >= obs_signed - 1e-12))
            left -= m
        return got

    gold_n = 200_000
    gold_count = counts(gold_n, seed=0, sided="two")
    truth = (gold_count + 1) / (gold_n + 1)
    task = Task(
        "permutation_p",
        EPS["permutation_p"],
        float(truth),
        f"add-one two-sided p from {gold_n} permutations; gold binomial se "
        f"{np.sqrt(truth * (1 - truth) / gold_n):.5f}",
    )

    def one_shot(seed: int, n_perm: int = 999) -> float:
        c = counts(n_perm, seed, "two")
        return (c + 1) / (n_perm + 1)

    def wrong_shot(seed: int, n_perm: int = 999) -> float:
        # One-sided p in the observed direction. The task asks for two-sided.
        c = counts(n_perm, seed, "one")
        return (c + 1) / (n_perm + 1)

    def internal(seed: int) -> tuple[float, int]:
        rs = np.random.default_rng(seed)
        total = 0
        total_count = 0
        estimate = np.nan
        for _ in range(MAX_REPS):
            child = int(rs.integers(0, 2**31 - 1))
            total_count += counts(999, child, "two")
            total += 999
            estimate = (total_count + 1) / (total + 1)
            mc_half = 1.96 * np.sqrt(estimate * (1 - estimate) / total)
            if mc_half <= task.eps:
                return float(estimate), total
        return float(estimate), total

    def external(seed: int) -> tuple[float, int]:
        rs = np.random.default_rng(seed)
        vals = []
        for k in range(1, MAX_REPS + 1):
            vals.append(one_shot(int(rs.integers(0, 2**31 - 1))))
            if halfwidth(np.asarray(vals, dtype=float)) <= task.eps:
                return float(np.mean(vals)), k
        return float(np.mean(vals)), len(vals)

    return task, one_shot, wrong_shot, internal, external


def split_task():
    rng = np.random.default_rng(13)
    n = 80
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([x1, x2])
    logits = 0.8 * x1 - 0.4 * x2
    y = rng.binomial(1, 1 / (1 + np.exp(-logits)))
    n_test = 24

    def accuracy(seed: int, on_train: bool) -> float:
        rs = np.random.default_rng(seed)
        order = rs.permutation(n)
        test, train = order[:n_test], order[n_test:]
        Xtr, ytr = X[train], y[train]
        Xte, yte = X[test], y[test]
        target_X, target_y = (Xtr, ytr) if on_train else (Xte, yte)
        if np.unique(ytr).size < 2:
            pred = np.full(target_y.shape, ytr[0])
            return float(np.mean(pred == target_y))
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
        clf.fit(Xtr, ytr)
        pred = clf.predict(target_X)
        return float(np.mean(pred == target_y))

    gold = np.array([accuracy(10_000 + i, on_train=False) for i in range(4000)])
    truth = float(gold.mean())
    task = Task(
        "random_split_acc",
        EPS["random_split_acc"],
        truth,
        f"mean test accuracy over 4000 splits; gold se {gold.std(ddof=1) / np.sqrt(4000):.5f}",
    )

    def one_shot(seed: int) -> float:
        return accuracy(seed, on_train=False)

    def wrong_shot(seed: int) -> float:
        # Scores the training fold. The task asks for the test fold.
        return accuracy(seed, on_train=True)

    def _until(seed: int, shot) -> tuple[float, int]:
        rs = np.random.default_rng(seed)
        vals = []
        for k in range(1, MAX_REPS + 1):
            vals.append(shot(int(rs.integers(0, 2**31 - 1))))
            if halfwidth(np.asarray(vals, dtype=float)) <= task.eps:
                return float(np.mean(vals)), k
        return float(np.mean(vals)), len(vals)

    def internal(seed: int) -> tuple[float, int]:
        return _until(seed, one_shot)

    def external(seed: int) -> tuple[float, int]:
        return _until(seed + 17, one_shot)

    return task, one_shot, wrong_shot, internal, external


def eval_policy(fn, truth: float, eps: float, n: int, seed0: int) -> dict:
    errs = 0
    reps = []
    t0 = time.perf_counter()
    for i in range(n):
        est, k = fn(seed0 + i)
        reps.append(k)
        if abs(est - truth) > eps:
            errs += 1
    elapsed = time.perf_counter() - t0
    return {
        "false_accept": errs / n,
        "false_accept_count": errs,
        "n": n,
        "median_reps": float(np.median(reps)),
        "mean_reps": float(np.mean(reps)),
        "seconds": elapsed,
    }


def main():
    report = {"eps": EPS, "n_runs": N_RUNS, "n_policy": N_POLICY, "tie_count": TIE_COUNT}
    tasks_out = []

    builders = [bootstrap_task, permutation_task, split_task]
    for build in builders:
        task, one_shot, wrong_shot, internal, external = build()
        print(f"running {task.name}", flush=True)
        t0 = time.perf_counter()
        correct_vals = np.array([one_shot(1000 + i) for i in range(N_RUNS)])
        wrong_vals = np.array([wrong_shot(2000 + i) for i in range(N_RUNS)])
        run_seconds = time.perf_counter() - t0
        correct = summarize_runs(correct_vals, task.truth, task.eps)
        wrong = summarize_runs(wrong_vals, task.truth, task.eps)
        entry = {
            "task": task.name,
            "eps": task.eps,
            "truth": task.truth,
            "truth_note": task.truth_note,
            "correct_runs": correct,
            "wrong_runs": wrong,
            "correct_values": correct_vals.tolist(),
            "wrong_values": wrong_vals.tolist(),
            "seconds_for_60_default_runs": run_seconds,
        }
        if not correct["sd_below_eps"]:
            entry["internal"] = eval_policy(internal, task.truth, task.eps, N_POLICY, 50_000)
            entry["external"] = eval_policy(external, task.truth, task.eps, N_POLICY, 80_000)
            gap = entry["internal"]["false_accept_count"] - entry["external"]["false_accept_count"]
            entry["internal_minus_external_count"] = gap
            entry["internal_not_materially_higher"] = bool(gap <= TIE_COUNT)
        tasks_out.append(entry)
        print(
            f"  correct sd={correct['sd']:.5f} eps={task.eps} below={correct['sd_below_eps']}",
            flush=True,
        )

    triggered = [t for t in tasks_out if not t["correct_runs"]["sd_below_eps"]]
    if not triggered:
        verdict = "end"
        reason = "all three correct programs have sample sd < ε"
    elif all(t["internal_not_materially_higher"] for t in triggered):
        verdict = "end"
        reason = (
            "where sd ≥ ε, raising internal repetitions is within 10/200 "
            "false accepts of external re-execution"
        )
    else:
        verdict = "not_end"
        reason = "internal false accepts exceed external by more than 10/200"
    report["tasks"] = tasks_out
    report["verdict"] = verdict
    report["reason"] = reason
    out = Path(__file__).resolve().parent / "data" / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(report, f, indent=2)
    print("WROTE", out)
    print("VERDICT", verdict)
    print(reason)


if __name__ == "__main__":
    main()
