r"""Verify requirement versions and actual application imports with this interpreter.

Run: .\.venv\Scripts\python.exe check_environment.py
Use `python -m pip check` as well to verify transitive dependency constraints.
"""
import importlib
from importlib import metadata
from pathlib import Path
import sys
from packaging.requirements import Requirement

APP_IMPORTS = ["numpy", "pandas", "plotly.graph_objects", "plotly.subplots",
               "polygon", "streamlit", "streamlit.testing.v1"]


def main():
    print(f"Interpreter: {sys.executable}")
    failures = []
    requirements = Path(__file__).with_name("requirements.txt").read_text(encoding="utf-8-sig")
    count = 0
    for line in requirements.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        req = Requirement(line)
        if req.marker and not req.marker.evaluate():
            continue
        count += 1
        try:
            version = metadata.version(req.name)
            if version not in req.specifier:
                failures.append(f"{req}: installed version is {version}")
        except metadata.PackageNotFoundError:
            failures.append(f"Missing requirement: {req}")
    print(f"Checked {count} requirements")
    for name in APP_IMPORTS:
        try:
            importlib.import_module(name)
            print(f"Import OK: {name}")
        except Exception as exc:
            failures.append(f"Import failed: {name}: {exc}")
    for failure in failures:
        print(f"ERROR: {failure}")
    if not failures:
        print("All requirements and application imports passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
