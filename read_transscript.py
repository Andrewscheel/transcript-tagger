import json
from pathlib import Path

path = Path("data/sample.json")
with path.open(encoding="utf-8") as f:
    transcript = json.load(f)

print("Call ID:", transcript["call_id"])
print("Number of turns:", len(transcript["turns"]))