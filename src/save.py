import os
import json
import pandas as pd
from datetime import datetime
from collections.abc import Mapping


def _feature_rows(feats) -> list[dict]:
    """Normalize the candidate collection while accepting legacy mappings."""
    if isinstance(feats, Mapping):
        return [dict(feats)]
    return [dict(row) for row in feats]


def save_features(feats, target: str, out_path: str, verbose: bool = False):
    """Save all candidate rows for one host star, ordered by orbital period."""
    if not out_path.endswith(".json") and not out_path.endswith(".csv"):
        filename = target + "_" + datetime.now().strftime("%Y%m%d") + ".csv"
        out_path = os.path.join(out_path, filename)

    rows = _feature_rows(feats)
    df = pd.DataFrame(rows)
    if "period_days" in df.columns:
        df["period_days"] = pd.to_numeric(df["period_days"], errors="coerce")
        df = df.sort_values("period_days", kind="stable", na_position="last")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.lower().endswith(".json"):
        with open(out_path, "w") as f:
            json.dump({"star": target, "candidates": df.to_dict("records")}, f, indent=2)
    else:
        df.to_csv(out_path, index=False)

    if verbose:
        print(f"Saved {len(df)} candidate row(s) to: {out_path}")


def save_lightkurve(lc, target: str, out_path: str, verbose: bool = False):
    if not out_path.endswith(".csv"):
        filename = target + "_" + datetime.now().strftime("%Y%m%d") + ".csv"
        out_path = os.path.join(out_path, filename)

    lc.to_csv(out_path)

    if verbose:
        print(f"Saved lightkurve to: {out_path}")
