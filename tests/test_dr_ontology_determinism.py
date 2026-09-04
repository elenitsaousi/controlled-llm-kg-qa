"""Regression guard for a nondeterminism bug in kg/dr_ontology.py.

`_load_dr_terms` used to iterate `set(graph.subjects())` directly. When two
different DR ontology terms (e.g. a class and an unrelated property) shared
the same normalized alias, whichever term got processed first won that alias
slot -- and that processing order depended on Python's set iteration order,
which is randomized per-process (PYTHONHASHSEED). The practical effect: the
same unchanged ontology file produced a different "how many classes/object
properties/datatype properties does the Digital Reference have" answer every
time the app restarted (observed varying by several dozen entries across
runs). The fix sorts subjects by string before iterating, so alias-collision
winners -- and therefore the reported counts -- are the same on every run.

A single pytest process only ever has one hash seed, so calling
dr_ontology_counts() twice in one process would trivially "agree" even with
the bug present. This test spawns separate child interpreters (each gets its
own random hash seed, like separate app restarts do) and asserts they report
identical counts.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_SNIPPET = (
    "import json\n"
    "from kg.dr_ontology import dr_ontology_counts\n"
    "print(json.dumps(dr_ontology_counts('')))\n"
)


def _counts_from_fresh_process(hash_seed: str) -> dict:
    import os

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    result = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_dr_ontology_counts_are_stable_across_process_restarts():
    counts_a = _counts_from_fresh_process("111")
    counts_b = _counts_from_fresh_process("222")
    counts_c = _counts_from_fresh_process("333")

    assert counts_a == counts_b == counts_c, (
        "DR ontology counts must not depend on the process's hash seed "
        f"(i.e. must be identical across app restarts): {counts_a} vs {counts_b} vs {counts_c}"
    )
    # Sanity: the ontology actually has content to count.
    assert counts_a["class"] > 0
    assert counts_a["object property"] > 0
    assert counts_a["datatype property"] > 0
