import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path


def save_features(feats, target: str, out_path: str, verbose: bool = False):
    if not out_path.endswith(".json") and not out_path.endswith(".csv"):
        filename = target + "_" + datetime.now().strftime("%Y%m%d") + ".csv"
        out_path = os.path.join(out_path, filename)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if out_path.lower().endswith(".json"):
        with open(out_path, "w") as f:
            json.dump({"target": target, "features": feats}, f, indent=2)
    else:
        df = pd.DataFrame([feats])
        df.insert(0, "target", target)
        df.to_csv(out_path, index=False)

    if verbose:
        print(f"Saved features to: {out_path}")


def save_lightkurve(lc, target: str, out_path: str, verbose: bool = False):
    if not out_path.endswith(".csv"):
        filename = target + "_" + datetime.now().strftime("%Y%m%d") + ".csv"
        out_path = os.path.join(out_path, filename)

    lc.to_csv(out_path)

    if verbose:
        print(f"Saved lightkurve to: {out_path}")
