import sys as _sys
from pathlib import Path as _Path
for _d in ('src', 'experiments', 'scripts'):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parents[3] / "results" / "triage" / "triage_study_analysis.json"
OUT = HERE / "break_even_data.tex"

MODEL_COMMANDS = {
    "qwen2.5:3b": "beSUQwenThree",
    "llama3.2:3b": "beSULlama",
    "phi3:latest": "beSUPhi",
    "qwen2.5:7b": "beSUQwenSeven",
}


def relevant(records, other):
    rows = [
        row
        for row in records
        if row.get("world") == "attacked" and row.get("a") == "S" and row.get("b") == other
    ]
    return sorted(rows, key=lambda row: row["prevalence"])


data = json.loads(ANALYSIS.read_text())
models = data["loss"]["models"]
lines = ["% Generated from results/triage/triage_study_analysis.json"]

for model, command in MODEL_COMMANDS.items():
    rows = relevant(models[model]["break_even"], "U")
    if len(rows) != 3:
        raise RuntimeError(f"expected three S versus U anchors for {model}")
    coords = []
    for row in rows:
        k_star = row.get("k_star")
        if row.get("a_wins") != "above_k_star" or k_star is None or k_star <= 0:
            raise RuntimeError(f"unexpected S versus U break even record for {model}: {row}")
        coords.append(f"({row['prevalence']:.8g},{k_star:.8g})")
    lines.append(f"\\def\\{command}{{{''.join(coords)}}}")

pitsa_status = {}
for model in MODEL_COMMANDS:
    rows = relevant(models[model]["break_even"], "PITSA")
    statuses = {row.get("a_wins") for row in rows}
    if len(rows) != 3 or len(statuses) != 1:
        raise RuntimeError(f"unexpected S versus PITSA records for {model}: {rows}")
    status = statuses.pop()
    if status == "tie":
        pitsa_status[model] = "tie"
    elif status == "B_always":
        pitsa_status[model] = "pitsa"
    elif status == "below_k_star" and all((row.get("k_star") or 0) < 1 for row in rows):
        pitsa_status[model] = "pitsa"
    else:
        raise RuntimeError(f"unexpected admissible S versus PITSA relation for {model}: {rows}")

pitsa_models = [model for model, status in pitsa_status.items() if status == "pitsa"]
tie_models = [model for model, status in pitsa_status.items() if status == "tie"]
if pitsa_models != ["qwen2.5:3b", "phi3:latest", "qwen2.5:7b"] or tie_models != ["llama3.2:3b"]:
    raise RuntimeError(f"unexpected S versus PITSA summary: {pitsa_status}")

lines.append(r"\def\bePitsaDominanceText{PITSA lower for all\\admissible $k$, three models}")
lines.append(r"\def\bePitsaTieText{llama3.2:3b equal for all $k$}")
OUT.write_text("\n".join(lines) + "\n")
print(OUT)
