#!/usr/bin/env python3
"""Self-test entry point for the whole repository.

Runs, in order: the executable algebra behind every theorem, the E1 unit
tests, the E2 analytic tests, and the E3 unit tests. The E3 block is skipped
when PyTorch is absent, and the skip is reported rather than hidden.

    python3 tests.py
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run_module_tests(path: Path, extra_sys_path: list[Path]) -> tuple[int, int, list[str]]:
    """Import a pytest-style module and call every test_* function."""
    for p in extra_sys_path:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = sorted(n for n in dir(module) if n.startswith("test_"))
    failures: list[str] = []
    for name in names:
        try:
            getattr(module, name)()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path.stem}.{name}: {type(exc).__name__}: {exc}")
    return len(names) - len(failures), len(names), failures


def main() -> int:
    all_failures: list[str] = []

    print("[1/4] executable algebra for the theorems")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_theory.py")],
        capture_output=True, text=True,
    )
    if "ALL_CHECKS_PASSED" in proc.stdout:
        print("      ALL_CHECKS_PASSED")
    else:
        all_failures.append("scripts/check_theory.py did not report ALL_CHECKS_PASSED")
        print("      FAILED")

    print("[2/4] E1 solved-class unit tests")
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "experiments" / "sampled_phase"), pattern="test_*.py")
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        result = unittest.TextTestRunner(stream=buffer, verbosity=0).run(suite)
    print(f"      {result.testsRun - len(result.failures) - len(result.errors)}"
          f"/{result.testsRun} passed")
    if not result.wasSuccessful():
        all_failures.append("experiments/sampled_phase unit tests failed")

    print("[3/4] E2 analytic tests")
    passed, total, failures = _run_module_tests(
        ROOT / "tests" / "test_analytic.py", [ROOT / "experiments" / "closed_loop"]
    )
    print(f"      {passed}/{total} passed")
    all_failures.extend(failures)

    print("[4/4] E3 unit tests")
    if importlib.util.find_spec("torch") is None:
        print("      SKIPPED (PyTorch not installed; see requirements.txt)")
    else:
        passed, total, failures = _run_module_tests(
            ROOT / "tests" / "test_e3.py", [ROOT / "experiments" / "ppo"]
        )
        print(f"      {passed}/{total} passed")
        all_failures.extend(failures)

    print()
    if all_failures:
        print(f"FAILED ({len(all_failures)})")
        for line in all_failures:
            print("  -", line)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
