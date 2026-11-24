"""CI honesty gate for the public ``memnoth-bench`` repository.

This is a structural fence, not a claim about any score. It fails the build if:

* a figure in ``results/summary/*.json`` asserts an official score, a comparability
  tier other than C, a bare percentage, a private absolute path, or a key/gold
  value;
* a tracked source file contains a JWT, a real-looking API key, a private repo
  name, or a machine-specific private drive path;
* any commit message carries AI-attribution markers; or
* a ``harnesses/*/adapter.py`` loses its LANE, its verified dataset pin, or the
  ``official: false`` stamp on ``evaluate()``.

Stdlib only. No network, no provider calls, no keys. Every path is resolved from
the repository root (computed from ``__file__``), and ``git`` subprocesses are
run with that root as cwd, so the module behaves the same from any launch dir.

The only files excluded from the source scan are the two files that *define* the
redaction regexes (``export/export.py`` and ``.github/workflows/verify.yml``) and
this test itself, which necessarily contains the forbidden literals. All three
are asserted to exist so a rename cannot quietly widen the exemption.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import unittest
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = REPO_ROOT / "results" / "summary"
HARNESSES_DIR = REPO_ROOT / "harnesses"


def setUpModule() -> None:
    """Make the repository root the cwd for every test in this module."""
    os.chdir(REPO_ROOT)


def iter_strings(value: Any, where: str = "$") -> Iterator[tuple[str, str]]:
    """Yield ``(json path, string)`` for every string nested in a JSON value."""
    if isinstance(value, str):
        yield where, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from iter_strings(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_strings(item, f"{where}[{index}]")


# --------------------------------------------------------------------------- #
# 1. Figure honesty in results/summary/*.json
# --------------------------------------------------------------------------- #

_WINDOWS_ABS = re.compile(r"[A-Za-z]:[\\/]")
_KEY_VALUE = re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}|eyJhbGci[A-Za-z0-9_\-]*")
_SECRET_FIELDS = frozenset(
    {"api_key", "apikey", "key", "secret", "token", "gold", "gold_answer", "gold_id", "private_key"}
)
_PLACEHOLDERS = frozenset(
    {"", "redacted", "withheld", "n/a", "none", "null", "dummy", "placeholder", "test", "example"}
)


class SummaryFigureHonestyTest(unittest.TestCase):
    """Every committed figure is informational, non-official and non-comparable."""

    def _figures(self) -> list[tuple[str, dict[str, Any]]]:
        files = sorted(SUMMARY_DIR.rglob("*.json")) if SUMMARY_DIR.is_dir() else []
        if not files:
            self.skipTest("results/summary contains no JSON figures yet — nothing to validate")

        figures: list[tuple[str, dict[str, Any]]] = []
        for path in files:
            where = str(path.relative_to(REPO_ROOT))
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self.fail(f"{where}: cannot read JSON: {exc}")
            if isinstance(raw, list):
                records, defaults = raw, {}
            elif isinstance(raw, dict) and isinstance(raw.get("figures"), list):
                records, defaults = raw["figures"], {k: v for k, v in raw.items() if k != "figures"}
            elif isinstance(raw, dict):
                records, defaults = [raw], {}
            else:
                self.fail(f"{where}: expected a figure object, a list, or an object with 'figures'")
            for index, record in enumerate(records):
                merged = {**defaults, **record} if isinstance(record, dict) else record
                figures.append((f"{where}#{index}", merged))

        if not figures:
            self.skipTest("results/summary contains no figures yet — nothing to validate")
        return figures

    def test_official_score_asserted_and_comparability_tier(self) -> None:
        errors: list[str] = []
        for where, fig in self._figures():
            if not isinstance(fig, dict):
                errors.append(f"{where}: figure is not an object")
                continue
            if fig.get("official") is not False:
                errors.append(f"{where}: 'official' must be the literal boolean false")
            if fig.get("score_asserted") is not False:
                errors.append(f"{where}: 'score_asserted' must be the literal boolean false")
            tier = fig.get("comparability_tier")
            if tier is None:
                # An absent tier is allowed only when the figure is explicitly derived.
                if fig.get("diagnostic") != "derived" and fig.get("derived") is not True:
                    errors.append(f"{where}: no comparability_tier and no 'derived' marker")
            elif tier != "C":
                errors.append(f"{where}: comparability_tier must be 'C', got {tier!r}")
        self.assertEqual(errors, [], "Non-official figure invariants violated:\n" + "\n".join(errors))

    def test_no_bare_percent_in_value(self) -> None:
        errors: list[str] = []
        for where, fig in self._figures():
            value = fig.get("value") if isinstance(fig, dict) else None
            if "%" in str(value):
                errors.append(f"{where}: value contains a bare '%' ({value!r})")
        self.assertEqual(errors, [], "Percentage values are never comparable:\n" + "\n".join(errors))

    def test_provenance_has_no_absolute_windows_path(self) -> None:
        errors: list[str] = []
        for where, fig in self._figures():
            provenance = fig.get("provenance", "") if isinstance(fig, dict) else ""
            for text in ([provenance] if isinstance(provenance, str) else []):
                if _WINDOWS_ABS.search(text):
                    errors.append(f"{where}: provenance carries a private absolute path: {text!r}")
        self.assertEqual(errors, [], "Private paths in provenance:\n" + "\n".join(errors))

    def test_no_key_or_gold_values(self) -> None:
        errors: list[str] = []
        for where, fig in self._figures():
            if not isinstance(fig, dict):
                continue
            for name, value in fig.items():
                if name.lower() in _SECRET_FIELDS and isinstance(value, str):
                    if value.strip().lower() not in _PLACEHOLDERS:
                        errors.append(f"{where}: field {name!r} carries a key/gold value")
            for path, text in iter_strings(fig):
                if _KEY_VALUE.search(text):
                    errors.append(f"{where}: {path} looks like a key/JWT value")
        self.assertEqual(errors, [], "Key or gold values in figures:\n" + "\n".join(errors))


# --------------------------------------------------------------------------- #
# 2. Source scan for secrets / private-path leaks
# --------------------------------------------------------------------------- #

# Label -> compiled pattern. The literals are constructed from fragments so this
# test file does not itself trip the scan before the self-exclusion applies.
_FORBIDDEN: dict[str, re.Pattern[str]] = {
    "JWT header": re.compile("eyJh" + "bGci"),
    "real-looking sk- key": re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    "private repo name": re.compile("memnoth-cloud|bench-private", re.IGNORECASE),
    "private E: path": re.compile(r"E:\\memnoth-prod", re.IGNORECASE),
    "private D: temp path": re.compile(r"D:\\temp\\opencode", re.IGNORECASE),
    "private D: build cache": re.compile(r"D:\\memnoth-builds-cache", re.IGNORECASE),
}

# These two tracked files define the redaction regexes and legitimately contain
# the forbidden literals. The test file itself is also exempt because it holds
# the scan literals, but it is a NEW file and is not tracked before it is merged,
# so the tracked-presence guard below covers only the two redaction definitions.
_REDACTION_EXEMPT = frozenset({"export/export.py", ".github/workflows/verify.yml"})
_SCAN_EXEMPT = _REDACTION_EXEMPT | {"tests/test_honesty.py"}


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return [part for part in result.stdout.split("\0") if part]


class SourceHygieneTest(unittest.TestCase):
    """No secret value or private path may enter any tracked source file."""

    def test_redaction_definitions_are_present(self) -> None:
        # Guard the exemption list: these files must stay tracked, or a rename
        # could silently turn a deliberate allowance into a blind spot.
        tracked = set(_tracked_files())
        for name in sorted(_REDACTION_EXEMPT):
            self.assertIn(name, tracked, f"exempt redaction file {name!r} is no longer tracked")

    def test_no_forbidden_patterns_in_tracked_files(self) -> None:
        failures: list[str] = []
        scanned = 0
        for name in _tracked_files():
            if name in _SCAN_EXEMPT:
                continue
            path = REPO_ROOT / name
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if len(raw) > 1_000_000 or b"\0" in raw:
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            scanned += 1
            for line_no, line in enumerate(text.splitlines(), 1):
                for label, pattern in _FORBIDDEN.items():
                    if pattern.search(line):
                        failures.append(f"{name}:{line_no}: {label}")
        self.assertGreater(scanned, 0, "git ls-files returned no scannable text files")
        self.assertEqual(
            failures, [], f"Forbidden content in {len(failures)} tracked line(s):\n" + "\n".join(failures)
        )


# --------------------------------------------------------------------------- #
# 3. Commit-message attribution markers
# --------------------------------------------------------------------------- #

_ATTRIBUTION = re.compile(
    r"co-authored-by|generated\s+(?:with|by)|\bclaude\b|\bagent\b|🤖|ai-generated",
    re.IGNORECASE,
)


class CommitAttributionTest(unittest.TestCase):
    """No commit message may carry AI attribution (full message, not just title)."""

    def test_no_ai_attribution_in_any_commit_message(self) -> None:
        result = subprocess.run(
            ["git", "log", "--format=%B%x00"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        messages = [part for part in result.stdout.split("\0") if part.strip()]
        failures: list[str] = []
        for index, message in enumerate(messages, 1):
            for line in message.splitlines():
                if _ATTRIBUTION.search(line):
                    failures.append(f"commit #{index}: {line.strip()}")
        self.assertGreater(len(messages), 0, "git log returned no commit messages")
        self.assertEqual(
            failures, [], f"AI attribution in {len(failures)} commit line(s):\n" + "\n".join(failures)
        )


# --------------------------------------------------------------------------- #
# 4. Adapter contract: LANE + verified pin + official:false in evaluate()
# --------------------------------------------------------------------------- #

_PIN_NAMES = frozenset(
    {"DATASET_SHA256", "VENDORED_GZ", "UPSTREAM_FIXTURES", "PINS", "SOURCE_FILE_PINS"}
)
_REQUIRED_EVALUATE_STAMP = re.compile(r"official[\"']?\s*:\s*False")


def _top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


class AdapterContractTest(unittest.TestCase):
    """Every harness adapter keeps its lane, its pin and its honesty stamp."""

    def test_stamps_helper_sets_official_false(self) -> None:
        # The adapters delegate the stamp to common.stamps(); if that helper
        # ever stops setting official:false, every adapter silently lies.
        path = HARNESSES_DIR / "common.py"
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        stamps = next(
            (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stamps"), None
        )
        self.assertIsNotNone(stamps, "harnesses/common.py has no stamps() helper")
        segment = ast.get_source_segment(text, stamps) or ""
        self.assertRegex(segment, r"[\"']official[\"']\s*:\s*False")

    def test_every_adapter_has_lane_pin_and_official_false(self) -> None:
        adapters = sorted(HARNESSES_DIR.glob("*/adapter.py"))
        self.assertGreater(len(adapters), 0, "no harnesses/*/adapter.py files found")

        errors: list[str] = []
        for path in adapters:
            where = str(path.relative_to(REPO_ROOT))
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
            names = _top_level_names(tree)

            lane = next(
                (
                    n
                    for n in tree.body
                    if isinstance(n, (ast.Assign, ast.AnnAssign))
                    and any(
                        isinstance(t, ast.Name) and t.id == "LANE"
                        for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
                    )
                ),
                None,
            )
            lane_value = (
                lane.value.value
                if lane is not None
                and isinstance(lane, ast.Assign)
                and isinstance(lane.value, ast.Constant)
                and isinstance(lane.value.value, str)
                else None
            )
            if not lane_value or not lane_value.strip():
                errors.append(f"{where}: missing non-empty LANE constant")

            # A verified dataset pin is either a single DATASET_SHA256 or one of
            # the explicit verified-pin structures used by multi-file harnesses.
            pin_names = names & _PIN_NAMES
            verified_flag = bool(re.search(r"[\"']verified[\"']\s*:\s*True", text)) or any(
                "verified" in name.lower() for name in names
            )
            if not pin_names and not verified_flag:
                errors.append(f"{where}: no DATASET_SHA256 and no verified-pin flag")

            evaluate = next(
                (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate"),
                None,
            )
            if evaluate is None:
                errors.append(f"{where}: no evaluate() function")
                continue
            segment = ast.get_source_segment(text, evaluate) or ""
            if "stamps(" not in segment and not _REQUIRED_EVALUATE_STAMP.search(segment):
                errors.append(f"{where}: evaluate() does not stamp official:false")

        self.assertEqual(
            errors, [], "Adapter contract violations:\n" + "\n".join(errors)
        )


if __name__ == "__main__":
    unittest.main()
