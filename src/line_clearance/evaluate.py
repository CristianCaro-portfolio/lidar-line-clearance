"""Score a pipeline run against the analytic truth of a simulated survey."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .catenary import Catenary, Conductor
from .config import PipelineConfig
from .pipeline import Analysis

RANK = {"critical": 0, "design": 1, "watch": 2}
CLASS_GROUPS = {
    "ground": ([2], [2]),
    "vegetation": ([3, 5], [3, 4, 5]),
    "conductor": ([14], [14]),
    "tower": ([15], [15]),
    "noise": ([7], [7]),
}


def _prf(tp: int, n_pred: int, n_true: int) -> dict[str, float]:
    precision = tp / n_pred if n_pred else 1.0
    recall = tp / n_true if n_true else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "predicted": n_pred,
        "true": n_true,
    }


def classification_scores(truth: np.ndarray, predicted: np.ndarray) -> dict[str, dict]:
    scores = {}
    for name, (true_codes, pred_codes) in CLASS_GROUPS.items():
        t, p = np.isin(truth, true_codes), np.isin(predicted, pred_codes)
        scores[name] = _prf(int((t & p).sum()), int(p.sum()), int(t.sum()))
    return scores


def _truth_conductors(truth: dict) -> list[tuple[dict, Conductor]]:
    out = []
    for w in truth["wires"]:
        model = Conductor(
            tuple(w["origin_xy"]),
            tuple(w["direction_xy"]),
            w["length"],
            Catenary(w["c"], w["l0"], w["z0"]),
        )
        out.append((w, model))
    return out


def tree_truth(truth: dict, cfg: PipelineConfig) -> list[dict]:
    """True clearance of every tree, from its noise free crown to the true wires."""
    phases = [m for w, m in _truth_conductors(truth) if w["role"] == "phase"]
    step = cfg.clearance.curve_step_m
    flown = cKDTree(np.vstack([m.sample(step) for m in phases]))
    design = cKDTree(
        np.vstack(
            [
                m.envelope(step, cfg.clearance.design_sag_factor, cfg.clearance.design_blowout_deg)
                for m in phases
            ]
        )
    )
    rows = []
    for tree in truth["trees"]:
        crown = tree.crown_surface()
        base = np.array([tree.x, tree.y, tree.base_z])
        rows.append(
            {
                "tree_id": tree.tree_id,
                "x": tree.x,
                "y": tree.y,
                "height": tree.height,
                "radius": tree.crown_radius,
                "flown": float(flown.query(crown)[0].min()),
                "design": float(design.query(crown)[0].min()),
                "fall_in": bool(tree.height >= flown.query(base)[0]),
            }
        )
    return rows


def evaluate(result: Analysis, labels: np.ndarray, truth: dict, cfg: PipelineConfig) -> dict:
    """All accuracy metrics for one run."""
    metrics: dict = {"classification": classification_scores(labels, result.classification)}

    # --- towers
    true_towers = np.asarray(truth["towers"])
    if len(result.towers):
        dist, _ = cKDTree(result.towers[:, :2]).query(true_towers[:, :2])
        found = dist < 5.0
        back, _ = cKDTree(true_towers[:, :2]).query(result.towers[:, :2])
        metrics["towers"] = {
            "true": len(true_towers),
            "detected": len(result.towers),
            "matched": int(found.sum()),
            "false": int((back >= 5.0).sum()),
            "mean_error_m": float(dist[found].mean()) if found.any() else None,
        }
    else:
        metrics["towers"] = {
            "true": len(true_towers),
            "detected": 0,
            "matched": 0,
            "false": 0,
            "mean_error_m": None,
        }

    # --- conductors: distance between fitted and true curves, and sag error
    curve_err, sag_err, matched = [], [], 0
    fitted_curves = [(c, cKDTree(c.model.sample(0.05))) for c in result.conductors]
    for wire, model in _truth_conductors(truth):
        samples = model.sample(1.0)[8:-8]
        best = None
        for conductor, curve in fitted_curves:
            dist = curve.query(samples)[0]
            if best is None or dist.mean() < best[0]:
                best = (float(dist.mean()), conductor)
        if best is not None and best[0] < 1.0:
            matched += 1
            curve_err.append(best[0])
            sag_err.append(abs(best[1].model.sag - wire["sag"]))
    metrics["conductors"] = {
        "true": len(truth["wires"]),
        "fitted": len(result.conductors),
        "matched": matched,
        "mean_curve_error_m": float(np.mean(curve_err)) if curve_err else None,
        "mean_sag_error_m": float(np.mean(sag_err)) if sag_err else None,
        "max_sag_error_m": float(np.max(sag_err)) if sag_err else None,
    }

    # --- clearance findings, scored per tree
    trees = tree_truth(truth, cfg)
    limit = cfg.clearance.violation_m
    tree_xy = np.array([[t["x"], t["y"]] for t in trees]).reshape(-1, 2)
    index = cKDTree(tree_xy) if len(trees) else None
    tree_radius = np.array([t["radius"] for t in trees])
    measured: dict[int, dict] = {}
    unmatched = {"critical": 0, "action": 0}
    fall_pred: set[int] = set()
    fall_unmatched = 0
    for f in result.findings:
        if f.kind == "coverage_gap" or index is None:
            continue
        # nearest crown, not nearest trunk: a wide crown can reach past a small neighbour
        dist, near = index.query([f.x, f.y], k=min(4, len(trees)))
        gap = np.atleast_1d(dist) - tree_radius[np.atleast_1d(near)]
        i = int(np.atleast_1d(near)[np.argmin(gap)])
        hit = gap.min() <= 2.0
        if f.kind == "fall_in":
            if hit:
                fall_pred.add(i)
            else:
                fall_unmatched += 1
            continue
        if not hit:
            unmatched["critical"] += f.severity == "critical"
            unmatched["action"] += f.severity in ("critical", "design")
            continue
        slot = measured.setdefault(i, {"flown": np.inf, "design": np.inf, "rank": 3})
        slot["flown"] = min(slot["flown"], f.clearance_m)
        slot["design"] = min(slot["design"], f.design_clearance_m)
        slot["rank"] = min(slot["rank"], RANK[f.severity])

    def detection(key: str) -> dict:
        """Truth comes from the true distance, prediction from the reported severity."""
        worst = 0 if key == "flown" else 1
        true_set = {i for i, t in enumerate(trees) if t[key] < limit}
        pred_set = {i for i, m in measured.items() if m["rank"] <= worst}
        extra = unmatched["critical" if key == "flown" else "action"]
        return _prf(len(true_set & pred_set), len(pred_set) + extra, len(true_set))

    flown_err = [
        abs(m["flown"] - trees[i]["flown"])
        for i, m in measured.items()
        if trees[i]["flown"] < cfg.clearance.warning_m
    ]
    design_err = [
        abs(m["design"] - trees[i]["design"])
        for i, m in measured.items()
        if trees[i]["flown"] < cfg.clearance.warning_m
    ]
    fall_true = {i for i, t in enumerate(trees) if t["fall_in"]}
    metrics["clearance"] = {
        "critical": detection("flown"),
        "action_list": detection("design"),
        "clearance_mae_m": float(np.mean(flown_err)) if flown_err else None,
        "clearance_max_error_m": float(np.max(flown_err)) if flown_err else None,
        "design_clearance_mae_m": float(np.mean(design_err)) if design_err else None,
        "fall_in": _prf(
            len(fall_true & fall_pred), len(fall_pred) + fall_unmatched, len(fall_true)
        ),
    }
    return metrics


def flatten(metrics: dict, prefix: str = "") -> dict[str, float]:
    """Nested metrics as dotted keys, handy for tables and aggregation."""
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        elif value is not None:
            flat[name] = float(value)
    return flat
