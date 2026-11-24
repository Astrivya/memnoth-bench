"""Thread-safe round-robin pool for Kilo gateway keys.

Security invariants (do not relax):

* Key values are read from a local env file / the process environment and are
  never logged, printed, or returned by :meth:`KeyPool.stats`.
* :func:`load_keys` reports nothing; only its caller may count the returned
  list. ``__main__`` prints counts and cooldown state only.
* ``stats()`` exposes slot indices, counts, and cooldown states -- no values,
  no prefixes, and no fingerprints that could identify a key.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable

DEFAULT_KEYS_PATH = r"D:\temp\opencode\kilo-keys.env"
"""Local env file holding six KILO_KEY_* names. Referenced by path only."""

_ENV_KEY_NAMES = ("KILO_API_KEY", "KILOCODE_API_KEY")
_KEY_NAME = re.compile(r"^KILO_KEY_[A-Za-z0-9_]*$")
_BURST = 6  # cap exponential backoff at 2**6 * base before the absolute cap


def load_keys(path: str | Path = DEFAULT_KEYS_PATH) -> list[str]:
    """Load KILO_KEY_* assignments from ``path`` plus two env vars, de-duped.

    Order is stable: env-file assignments first, then ``KILO_API_KEY`` and
    ``KILOCODE_API_KEY``. Blank lines, comments, non-KILO names and empty
    values are ignored. A missing file is not fatal (env vars may still be
    set). This function never prints or logs anything, least of all a value.
    """
    keys: list[str] = []
    seen: set[str] = set()

    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name = name.strip()
        if name[:6].upper() == "EXPORT ":
            name = name[6:].strip()
        if not _KEY_NAME.fullmatch(name):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1].strip()
        if not value or value in seen:
            continue
        seen.add(value)
        keys.append(value)

    for env_name in _ENV_KEY_NAMES:
        value = (os.environ.get(env_name) or "").strip()
        if value and value not in seen:
            seen.add(value)
            keys.append(value)

    return keys


def _is_rate_limit(status_code: int | None) -> bool:
    """True only for 429 or a 5xx response; other failures are not cooldowns."""
    if status_code is None:
        return False
    return status_code == 429 or 500 <= status_code <= 599


class _Slot:
    """Internal per-key state. ``ordinal`` is the only identity exposed."""

    __slots__ = ("key", "ordinal", "in_use", "cooldown_until", "failures", "successes", "rate_limited")

    def __init__(self, key: str, ordinal: int) -> None:
        self.key = key
        self.ordinal = ordinal
        self.in_use = False
        self.cooldown_until = 0.0
        self.failures = 0
        self.successes = 0
        self.rate_limited = 0


class KeyPool:
    """Round-robin key allocator with per-key cooldown on 429/5xx.

    ``get()`` hands out the next available key and marks it in use; ``release``
    clears that state and, for a 429/5xx failure, cools that one key down for an
    exponentially growing interval. All state is guarded by a single lock, so
    concurrent callers never receive the same key and cooldown bookkeeping is
    atomic. When every key is in use or cooling, ``get()`` returns ``None`` --
    it never fabricates a key and never silently reuses a cooled one.
    """

    def __init__(
        self,
        keys: Iterable[str],
        *,
        cooldown_seconds: float = 60.0,
        max_cooldown_seconds: float = 900.0,
    ) -> None:
        unique: list[str] = []
        seen: set[str] = set()
        for value in keys:
            if not isinstance(value, str) or not value or value in seen:
                continue
            seen.add(value)
            unique.append(value)
        if not unique:
            raise ValueError("KeyPool requires at least one non-empty key")

        self._slots = [_Slot(value, index) for index, value in enumerate(unique)]
        self._by_key = {value: index for index, value in enumerate(unique)}
        self._cursor = 0
        self._lock = threading.RLock()
        self._base_cooldown = max(0.0, float(cooldown_seconds))
        self._max_cooldown = max(0.0, float(max_cooldown_seconds))
        self._unknown_releases = 0

    def get(self) -> str | None:
        """Return the next available key, or ``None`` if all are busy/cooling."""
        with self._lock:
            now = time.monotonic()
            count = len(self._slots)
            for step in range(count):
                index = (self._cursor + step) % count
                slot = self._slots[index]
                if slot.in_use or slot.cooldown_until > now:
                    continue
                self._cursor = (index + 1) % count
                slot.in_use = True
                return slot.key
            return None

    def release(self, key: str, success: bool, status_code: int | None) -> bool:
        """Release ``key``; cool it down on 429/5xx failures.

        Returns ``False`` only when ``key`` is not a member of this pool (the
        message-free failure is counted in :meth:`stats`). A successful release
        resets the failure streak but does not cancel an active time-based
        cooldown: cooldowns expire on their own.
        """
        with self._lock:
            index = self._by_key.get(key)
            if index is None:
                self._unknown_releases += 1
                return False
            slot = self._slots[index]
            slot.in_use = False
            if success:
                slot.failures = 0
                slot.successes += 1
                return True
            slot.failures += 1
            if _is_rate_limit(status_code):
                slot.rate_limited += 1
                delay = self._base_cooldown * (2 ** min(slot.failures - 1, _BURST))
                delay = min(delay, self._max_cooldown) if self._max_cooldown else delay
                slot.cooldown_until = max(
                    slot.cooldown_until, time.monotonic() + delay
                )
            return True

    def stats(self) -> dict[str, Any]:
        """Counts and cooldown states only. This dict contains no key values."""
        with self._lock:
            now = time.monotonic()
            slots: list[dict[str, Any]] = []
            cooling = 0
            in_use = 0
            available = 0
            for slot in self._slots:
                remaining = max(0.0, slot.cooldown_until - now)
                is_cooling = remaining > 0.0
                cooling += int(is_cooling)
                in_use += int(slot.in_use)
                available += int(not is_cooling and not slot.in_use)
                slots.append(
                    {
                        "slot": f"slot-{slot.ordinal}",
                        "in_use": slot.in_use,
                        "cooling_down": is_cooling,
                        "cooldown_remaining_s": round(remaining, 3),
                        "successes": slot.successes,
                        "failures": slot.failures,
                        "rate_limited": slot.rate_limited,
                    }
                )
            return {
                "keys_total": len(self._slots),
                "keys_available": available,
                "keys_in_use": in_use,
                "keys_cooling_down": cooling,
                "unknown_release_count": self._unknown_releases,
                "slots": slots,
            }


def _smoke() -> int:
    """Offline smoke: load, count, and exercise a synthetic 429 cooldown."""
    keys = load_keys()
    print(f"keys_loaded: {len(keys)}")
    if keys:
        print("key_source: env-file KILO_KEY_* + KILO_API_KEY/KILOCODE_API_KEY")
    else:
        print("key_source: none found; using a labelled synthetic placeholder")

    pool_keys = keys or ["synthetic-placeholder-not-a-real-key"]
    pool = KeyPool(pool_keys, cooldown_seconds=0.3, max_cooldown_seconds=0.6)

    first = pool.get()
    assert first is not None, "get() must hand out an available key"
    first_index = pool_keys.index(first)
    assert pool.release(first, success=False, status_code=429)
    state = pool.stats()
    assert state["slots"][first_index]["cooling_down"] is True
    assert state["slots"][first_index]["rate_limited"] == 1
    assert state["keys_cooling_down"] == 1
    print("synthetic_429: cooldown applied to one slot")

    second = pool.get()
    assert second is not None, "a second key must remain available"
    assert second != first, "a cooling key must never be handed out again"
    assert pool.release(second, success=True, status_code=200)
    state = pool.stats()
    assert state["slots"][pool_keys.index(second)]["successes"] == 1
    assert state["slots"][first_index]["in_use"] is False

    rendered = json.dumps(state, sort_keys=True)
    expected_fields = {
        "slot", "in_use", "cooling_down", "cooldown_remaining_s",
        "successes", "failures", "rate_limited",
    }
    assert all(set(slot) == expected_fields for slot in state["slots"])
    assert "value" not in rendered and "key" not in json.dumps(state["slots"])
    print("stats_contains_no_key_values: true")
    print("stats_final: " + rendered)
    if len(pool_keys) == 1:
        print("limitations: one key only; round-robin skip could not be fully exercised")
    print("smoke_ok: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
