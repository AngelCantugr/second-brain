"""Render a pytest-benchmark JSON report into a human-readable Markdown table.

Usage: python bench_report.py <benchmarks.json> <benchmarks.md>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f}"


def render(data: dict) -> str:
    lines = [
        "# Integration Benchmark Report",
        "",
        f"Generated: {data.get('datetime', 'unknown')}",
        "",
        "| Benchmark | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | StdDev (ms) | Rounds | Ops/sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    benchmarks = sorted(data.get("benchmarks", []), key=lambda b: b["name"])
    for bench in benchmarks:
        stats = bench["stats"]
        lines.append(
            "| {name} | {mean} | {median} | {min} | {max} | {stddev} | {rounds} | {ops:.2f} |".format(
                name=bench["name"],
                mean=_ms(stats["mean"]),
                median=_ms(stats["median"]),
                min=_ms(stats["min"]),
                max=_ms(stats["max"]),
                stddev=_ms(stats["stddev"]),
                rounds=stats["rounds"],
                ops=stats["ops"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <benchmarks.json> <benchmarks.md>", file=sys.stderr)
        raise SystemExit(2)

    json_path, md_path = Path(sys.argv[1]), Path(sys.argv[2])
    data = json.loads(json_path.read_text(encoding="utf-8"))
    md_path.write_text(render(data), encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
