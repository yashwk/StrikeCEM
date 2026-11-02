#!/usr/bin/env python3
"""Fail closed until independent NASA Almond reference curves are available."""

import json
import os


def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = [
        "nasa_almond_mom_1_19ghz.json",
        "nasa_almond_mom_3_0ghz.json",
        "nasa_almond_mom_9_92ghz.json",
    ]
    for filename in files:
        path = os.path.join(base_dir, filename)
        with open(path) as f:
            data = json.load(f)
        if (data.get("data_status") != "digitized_or_measured" or
                data.get("validation_eligible") is not True):
            print(f"Cannot validate NASA Almond: {filename} is synthetic illustrative data.")
            print("Add sourced MoM or chamber curves with digitization provenance before comparing.")
            return 2

    print("Independent reference files are present, but no sourced acceptance profile is configured.")
    print("No accuracy comparison was run.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
