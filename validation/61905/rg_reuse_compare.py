# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare fault-injected GPU trials with full recomputation."""

import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
oracle = json.loads((root / "oracle.json").read_text())["output"]
cells = []
for tree in ("baseline", "patched"):
    for policy in ("fail", "recompute"):
        for failure in ("success", "failure"):
            name = f"{tree}-{policy}-{failure}"
            data = json.loads((root / f"{name}.json").read_text())
            rows = data["trials"]
            matches = []
            deltas = []
            for row in rows:
                output = row["unrelated_output"]
                matches.append(
                    output is not None and output["tokens"] == oracle["tokens"]
                )
                if output and matches[-1]:
                    deltas.append(
                        max(
                            abs(actual - expected)
                            for actual, expected in zip(
                                output["chosen_logprobs"], oracle["chosen_logprobs"]
                            )
                        )
                    )
                if tree == "patched" or failure == "success":
                    assert matches[-1], (name, row, oracle)
                    assert len(output["chosen_logprobs"]) == len(
                        oracle["chosen_logprobs"]
                    )
                    assert deltas[-1] <= 0.02, (name, deltas[-1])
                    assert not row["unrelated_affected_by_old_report"]
                else:
                    assert row["unrelated_affected_by_old_report"]
            cells.append(
                {
                    "cell": name,
                    "trials": len(rows),
                    "wrong_attribution": sum(
                        row["unrelated_affected_by_old_report"] for row in rows
                    ),
                    "greedy_matches_oracle": None
                    if all(
                        row.get("stopped_after_wrong_attribution", False)
                        for row in rows
                    )
                    else sum(matches),
                    "stopped_after_wrong_attribution": sum(
                        row.get("stopped_after_wrong_attribution", False)
                        for row in rows
                    ),
                    "max_chosen_logprob_delta": max(deltas) if deltas else None,
                    "median_ttft_s": statistics.median(
                        row["ttft_s"] for row in rows if row["ttft_s"] is not None
                    )
                    if any(row["ttft_s"] is not None for row in rows)
                    else None,
                }
            )
report = {"oracle_tokens": oracle["tokens"], "cells": cells}
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
