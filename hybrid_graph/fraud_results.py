"""Per-target reports and aggregate certificate curves for fraud experiments."""

import csv
import numpy as np


MASK_FIELDS = {
    "tight": "grid_tight",
    "half": "grid_half",
    "seq_tight": "grid_tight_seq",
    "seq_half": "grid_half_seq",
}
CURVE_COLUMNS = (
    "cert_acc_mode",
    "frac_cert_mode",
    "cert_acc_tight",
    "cert_acc_half",
    "cert_acc_seq_mode",
    "frac_cert_seq_mode",
    "cert_acc_seq_tight",
    "cert_acc_seq_half",
    "frac_cert_seq_tight",
    "frac_cert_seq_half",
)


def certificate_row(node, label, result, budgets, radii):
    """Keep the existing TSV fields and their order."""
    row = dict(
        idx=node,
        label=label,
        predict=int(result["pred_class"][0]),
        second=int(result["second_class"][0]),
    )
    row.update({key: float(result[key][0]) for key in ("pA_lower", "pB_upper")})
    row.update({key: int(result[key][0]) for key in ("m1", "m0")})
    row["correct"] = int(row["predict"] == label)
    for name, field in MASK_FIELDS.items():
        row[f"certified_any_{name}"] = int(result[field].any())
    for prefix, field in (("base_tight", "grid_tight"), ("base_seq_tight", "grid_tight_seq")):
        mask = np.asarray(result[field], dtype=bool)
        d_hits, eps_hits = np.where(mask)
        row[f"{prefix}_area"] = float(mask.mean()) if mask.any() else 0.0
        row[f"{prefix}_max_d"] = int(np.asarray(budgets)[d_hits].max()) if mask.any() else -1
        row[f"{prefix}_max_eps"] = float(np.asarray(radii)[eps_hits].max()) if mask.any() else 0.0
    return row


def certificate_curves(results, correct, mode):
    """Compute all-node accuracy and certified fractions for both criteria."""
    correct = np.asarray(correct, dtype=bool)[:, None, None]
    curves = {}
    for name, field in MASK_FIELDS.items():
        masks = np.stack([result[field] for result in results]).astype(bool, copy=False)
        curves[f"cert_acc_{name}"] = (masks & correct).mean(axis=0)
        curves[f"frac_cert_{name}"] = masks.mean(axis=0)
    for suffix in ("", "seq_"):
        for prefix in ("cert_acc", "frac_cert"):
            curves[f"{prefix}_{suffix}mode"] = curves[f"{prefix}_{suffix}{mode}"]
    return curves


def write_curve_csv(path, budgets, radii, curves):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("d", "eps", *CURVE_COLUMNS))
        for i, budget in enumerate(budgets):
            for j, radius in enumerate(radii):
                writer.writerow(
                    (
                        int(budget),
                        float(radius),
                        *(float(curves[key][i, j]) for key in CURVE_COLUMNS),
                    )
                )
