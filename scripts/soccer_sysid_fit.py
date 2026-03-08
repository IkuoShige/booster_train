#!/usr/bin/env python3
"""Fit soccer-ball contact parameters from drop/rolling trajectories.

This script provides a light-weight approximation of the paper's CMA-ES
identification workflow and exports profile(s) consumable by training configs.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class BallParams:
    static_friction: float
    dynamic_friction: float
    restitution: float
    linear_damping: float
    angular_damping: float


BOUNDS = {
    "static_friction": (0.0, 1.0),
    "dynamic_friction": (0.0, 1.0),
    "restitution": (0.0, 1.0),
    "linear_damping": (0.0, 5.0),
    "angular_damping": (0.0, 5.0),
}


def _clamp_params(values: np.ndarray) -> np.ndarray:
    clipped = values.copy()
    for i, key in enumerate(BOUNDS.keys()):
        lo, hi = BOUNDS[key]
        clipped[i] = np.clip(clipped[i], lo, hi)
    return clipped


def _load_curve(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns [time,value] in {path}")
    return data[:, 0], data[:, 1]


def _simulate_drop(times: np.ndarray, p: BallParams) -> np.ndarray:
    dt = 0.005
    g = 9.81
    z = 1.0
    vz = 0.0
    out = np.zeros_like(times)
    t_idx = 0

    t = 0.0
    max_t = float(times[-1])
    while t <= max_t + 1e-9:
        while t_idx < len(times) and t >= times[t_idx] - 1e-9:
            out[t_idx] = z
            t_idx += 1

        vz -= g * dt
        vz *= math.exp(-p.linear_damping * dt)
        z += vz * dt
        if z < 0.0:
            z = 0.0
            if vz < 0.0:
                vz = -p.restitution * vz
        t += dt

    return out


def _estimate_initial_speed(times: np.ndarray, dist: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    dt = max(times[1] - times[0], 1e-4)
    return max((dist[1] - dist[0]) / dt, 0.0)


def _simulate_roll(times: np.ndarray, p: BallParams, v0: float) -> np.ndarray:
    dt = 0.005
    g = 9.81
    v = float(v0)
    x = 0.0
    out = np.zeros_like(times)
    t_idx = 0

    t = 0.0
    max_t = float(times[-1])
    while t <= max_t + 1e-9:
        while t_idx < len(times) and t >= times[t_idx] - 1e-9:
            out[t_idx] = x
            t_idx += 1

        # Lightweight deceleration surrogate with friction + damping terms.
        acc = g * p.dynamic_friction + p.linear_damping * v + 0.05 * p.angular_damping * v
        v = max(v - acc * dt, 0.0)
        x += v * dt
        t += dt

    return out


def _loss(values: np.ndarray, t_drop: np.ndarray, y_drop: np.ndarray, t_roll: np.ndarray, y_roll: np.ndarray) -> float:
    p = BallParams(*values.tolist())

    sim_drop = _simulate_drop(t_drop, p)
    v0 = _estimate_initial_speed(t_roll, y_roll)
    sim_roll = _simulate_roll(t_roll, p, v0)

    loss_drop = np.mean((sim_drop - y_drop) ** 2)
    loss_roll = np.mean((sim_roll - y_roll) ** 2)
    return float(loss_drop + loss_roll)


def _run_random_search(
    init: np.ndarray,
    t_drop: np.ndarray,
    y_drop: np.ndarray,
    t_roll: np.ndarray,
    y_roll: np.ndarray,
    max_iters: int,
    population: int,
    sigma: float,
) -> tuple[np.ndarray, float]:
    best = init.copy()
    best_loss = _loss(best, t_drop, y_drop, t_roll, y_roll)

    for _ in range(max_iters):
        for _ in range(population):
            cand = _clamp_params(best + np.random.randn(best.shape[0]) * sigma)
            cand_loss = _loss(cand, t_drop, y_drop, t_roll, y_roll)
            if cand_loss < best_loss:
                best = cand
                best_loss = cand_loss
        sigma *= 0.99

    return best, best_loss


def _run_cma(
    init: np.ndarray,
    t_drop: np.ndarray,
    y_drop: np.ndarray,
    t_roll: np.ndarray,
    y_roll: np.ndarray,
    max_iters: int,
    population: int,
    sigma: float,
) -> tuple[np.ndarray, float, str]:
    try:
        import cma
    except ImportError:
        best, best_loss = _run_random_search(
            init,
            t_drop,
            y_drop,
            t_roll,
            y_roll,
            max_iters=max_iters,
            population=population,
            sigma=sigma,
        )
        return best, best_loss, "random_search"

    opts = {
        "popsize": population,
        "bounds": [
            [BOUNDS[k][0] for k in BOUNDS],
            [BOUNDS[k][1] for k in BOUNDS],
        ],
        "verbose": -9,
    }
    es = cma.CMAEvolutionStrategy(init.tolist(), sigma, opts)

    for _ in range(max_iters):
        xs = es.ask()
        losses = [_loss(np.asarray(x), t_drop, y_drop, t_roll, y_roll) for x in xs]
        es.tell(xs, losses)

    best = _clamp_params(np.asarray(es.result.xbest))
    best_loss = _loss(best, t_drop, y_drop, t_roll, y_roll)
    return best, best_loss, "cma_es"


def _to_profile(values: np.ndarray) -> dict[str, float]:
    return {
        "static_friction": float(values[0]),
        "dynamic_friction": float(values[1]),
        "restitution": float(values[2]),
        "linear_damping": float(values[3]),
        "angular_damping": float(values[4]),
    }


def main():
    parser = argparse.ArgumentParser(description="Fit soccer-ball profile from drop + rolling curves.")
    parser.add_argument("--drop_csv", type=pathlib.Path, required=True, help="CSV [time,height]")
    parser.add_argument("--rolling_csv", type=pathlib.Path, required=True, help="CSV [time,distance]")
    parser.add_argument("--surface", type=str, default="hard", choices=["hard", "grass"], help="Profile name")
    parser.add_argument("--output_yaml", type=pathlib.Path, required=True)
    parser.add_argument("--max_iters", type=int, default=200)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.2)
    args = parser.parse_args()

    t_drop, y_drop = _load_curve(args.drop_csv)
    t_roll, y_roll = _load_curve(args.rolling_csv)

    init = np.array([0.5, 0.5, 0.5, 1.0, 1.0], dtype=np.float64)
    best, best_loss, optimizer_name = _run_cma(
        init,
        t_drop,
        y_drop,
        t_roll,
        y_roll,
        max_iters=args.max_iters,
        population=args.population,
        sigma=args.sigma,
    )

    output = {
        "version": 1,
        "surface_profiles": {
            args.surface: _to_profile(best),
        },
        "noise_sigma": {
            "static_friction": 0.05,
            "dynamic_friction": 0.05,
            "restitution": 0.03,
            "linear_damping": 0.1,
            "angular_damping": 0.1,
        },
        "fitting": {
            "optimizer": optimizer_name,
            "best_loss": best_loss,
            "max_iters": args.max_iters,
            "population": args.population,
            "sigma": args.sigma,
            "bounds": BOUNDS,
            "drop_csv": str(args.drop_csv),
            "rolling_csv": str(args.rolling_csv),
        },
    }

    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with args.output_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(output, f, sort_keys=False)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
