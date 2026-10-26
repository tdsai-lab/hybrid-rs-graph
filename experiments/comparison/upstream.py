"""Import verified upstream clones without modifying or mixing their modules."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def verified_repo(name):
    spec = json.loads((ROOT / "external/repos.json").read_text())[name]
    path = ROOT / "external" / name
    if not (path / ".git").exists():
        raise RuntimeError(f"Missing {name}. Run python scripts/fetch_external.py {name}")
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if commit != spec["revision"]:
        raise RuntimeError(f"{name} revision differs from external/repos.json: {commit}")
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError(f"{name} has local changes; use the pinned clean clone.")
    return path, dict(repository=spec["url"], commit=commit)


def load_module(name, relative):
    path, record = verified_repo(name)
    filename = path / relative
    record.update(file=relative, sha256=hashlib.sha256(filename.read_bytes()).hexdigest())
    previous_path = sys.path[:]
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    # AGNNCert and CertifyGNN both use the unqualified name 'utils'.
    previous_utils = sys.modules.pop("utils", None)
    import scipy.stats as stats
    old_binom_test = getattr(stats, "binom_test", None)
    if old_binom_test is None:
        stats.binom_test = lambda x, n=None, p=0.5, alternative="two-sided": stats.binomtest(
            int(x), int(n), p=p, alternative=alternative).pvalue
    try:
        sys.path.insert(0, str(path))
        spec = importlib.util.spec_from_file_location(f"_upstream_{name}_{filename.stem}", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode
        sys.modules.pop("utils", None)
        if previous_utils is not None:
            sys.modules["utils"] = previous_utils
        if old_binom_test is None:
            del stats.binom_test
    return module, record


def topology_thresholds(table_keep):
    """Read the authors' P_Base constants without running their analysis CLI."""
    path, record = verified_repo("CertifyGNN")
    source = path / "calculate_certify_K_efficient.py"
    tree = ast.parse(source.read_text())
    for node in tree.body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        comparator = node.test.comparators[0]
        if isinstance(comparator, ast.Constant) and comparator.value == table_keep:
            for statement in node.body:
                if isinstance(statement, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "P_Base" for t in statement.targets):
                    return ast.literal_eval(statement.value), record | {
                        "file": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    raise ValueError(f"No upstream threshold table for {table_keep}; choose a tabulated value.")
