"""Export data/final-scoring/results.json as strict JSON (NaN -> null) for the Typst paper to read."""

import json
import math
from pathlib import Path

SOURCE = Path("data/final-scoring/results.json")
TARGET = Path("data/paper/results.json")


def strict(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: strict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strict(item) for item in value]
    return value


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(strict(json.loads(SOURCE.read_text())), indent=1, allow_nan=False) + "\n")
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
