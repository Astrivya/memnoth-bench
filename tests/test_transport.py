"""Regression tests for the two transport fixes in 053dd34 and 1995928.

These lock in behaviour that regressed silently before the fixes:

* ``KeyPool`` cooling defaults are gentle (15s / 120s / 2s), not the old
  60s / 900s schedule that cooled every key under sustained load.
* ``get(wait_seconds=0)`` returns ``None`` immediately when every slot is
  cooling, while a positive wait lets a cooling key come back around.
* ``release`` resets the failure streak on success and cools a slot on a
  429; ``stats()`` exposes counts and states only, never key values.
* ``_post_json`` treats an empty answer as a transient failure and retries
  it inside the same bounded budget, returning ``'empty_answer'`` only on
  exhaustion, and short-circuits to ``'no_key'`` with zero network calls.

Stdlib only. No real keys, no network: every HTTP response is a fake and the
only key value used is the literal dummy ``test-key-not-real``.
"""

from __future__ import annotations

import json
import os
import time
import unittest
from unittest.mock import patch

from runner import kilo_pool, locomo_reader

# Tuning env vars must not leak into the defaults test.
_TUNING_ENV = (
    kilo_pool.COOLDOWN_ENV,
    kilo_pool.MAX_COOLDOWN_ENV,
    kilo_pool.WAIT_ENV,
)

# The ONLY key string handed to _post_json. It is a labelled dummy, never real.
DUMMY_KEY = "test-key-not-real"


class _FakeResponse:
    """Minimal context-manager response for ``urlopen`` returning fixed bytes."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _chat_body(content: str) -> bytes:
    """Build a chat-completions JSON body with a single ``content`` value."""
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")


class KeyPoolDefaultsTest(unittest.TestCase):
    """The gentle 15/120/2 schedule must not drift back to 60/900."""

    def test_defaults_are_gentle(self) -> None:
        # Module constants are the pin, independent of any ambient env var.
        self.assertEqual(kilo_pool.DEFAULT_COOLDOWN_SECONDS, 15.0)
        self.assertEqual(kilo_pool.DEFAULT_MAX_COOLDOWN_SECONDS, 120.0)
        self.assertEqual(kilo_pool.DEFAULT_WAIT_SECONDS, 2.0)

        # A default-constructed pool (with tuning env cleared) resolves to the
        # same gentle values, not the old 60s/900s schedule.
        with patch.dict(os.environ, {name: "" for name in _TUNING_ENV}):
            pool = kilo_pool.KeyPool([DUMMY_KEY])
        self.assertEqual(pool._base_cooldown, 15.0)
        self.assertEqual(pool._max_cooldown, 120.0)
        self.assertEqual(pool._default_wait, 2.0)
        self.assertNotIn(pool._base_cooldown, (60.0, 900.0))
        self.assertNotIn(pool._max_cooldown, (900.0,))


class KeyPoolWaitTest(unittest.TestCase):
    """``get()`` honours its wait budget without ever reusing a cooling key."""

    def test_get_zero_wait_returns_none_immediately_when_all_cooling(self) -> None:
        pool = kilo_pool.KeyPool(
            ["synthetic-key-cool"],
            cooldown_seconds=0.5,
            max_cooldown_seconds=1.0,
            wait_seconds=0.0,
        )
        key = pool.get()
        self.assertEqual(key, "synthetic-key-cool")
        self.assertTrue(pool.release(key, success=False, status_code=429))

        # Every slot is now cooling; wait=0 must return None right away rather
        # than blocking until the 0.5s cooldown expires.
        started = time.monotonic()
        self.assertIsNone(pool.get(wait_seconds=0))
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.2)

    def test_get_positive_wait_waits_for_cooldown_and_returns_key(self) -> None:
        pool = kilo_pool.KeyPool(
            ["synthetic-key-wait"],
            cooldown_seconds=0.2,
            max_cooldown_seconds=0.4,
            wait_seconds=0.0,
        )
        key = pool.get()
        self.assertTrue(pool.release(key, success=False, status_code=429))

        # With a positive budget, get() polls until the short cooldown expires
        # and then hands the key back out.
        started = time.monotonic()
        again = pool.get(wait_seconds=1.0)
        elapsed = time.monotonic() - started
        self.assertEqual(again, "synthetic-key-wait")
        self.assertGreaterEqual(elapsed, 0.1)
        self.assertLess(elapsed, 1.0)


class KeyPoolReleaseTest(unittest.TestCase):
    """Failure bookkeeping and the no-key-values invariant in ``stats()``."""

    def test_release_429_sets_cooldown(self) -> None:
        pool = kilo_pool.KeyPool(
            ["synthetic-key-429"],
            cooldown_seconds=0.5,
            max_cooldown_seconds=1.0,
            wait_seconds=0.0,
        )
        key = pool.get()
        self.assertTrue(pool.release(key, success=False, status_code=429))
        slot = pool.stats()["slots"][0]
        self.assertTrue(slot["cooling_down"])
        self.assertEqual(slot["failures"], 1)
        self.assertEqual(slot["rate_limited"], 1)
        # A cooling key is never handed out again in the meantime.
        self.assertIsNone(pool.get(wait_seconds=0))

    def test_release_success_resets_failure_streak(self) -> None:
        pool = kilo_pool.KeyPool(
            ["synthetic-key-success"],
            cooldown_seconds=0.5,
            max_cooldown_seconds=1.0,
            wait_seconds=0.0,
        )
        key = pool.get()
        # A non-rate-limit failure increments the streak without a cooldown, so
        # the same key can be reacquired immediately for the success release.
        self.assertTrue(pool.release(key, success=False, status_code=400))
        self.assertEqual(pool.stats()["slots"][0]["failures"], 1)

        again = pool.get()
        self.assertEqual(again, "synthetic-key-success")
        self.assertTrue(pool.release(again, success=True, status_code=200))
        slot = pool.stats()["slots"][0]
        self.assertEqual(slot["failures"], 0)
        self.assertEqual(slot["successes"], 1)

    def test_stats_exposes_no_key_values(self) -> None:
        sentinel = "SENTINEL_VALUE_MUST_NOT_APPEAR"
        pool = kilo_pool.KeyPool([sentinel], wait_seconds=0.0)
        key = pool.get()
        pool.release(key, success=False, status_code=429)
        state = pool.stats()

        rendered = json.dumps(state, sort_keys=True)
        self.assertNotIn(sentinel, rendered)
        for slot in state["slots"]:
            self.assertIn("slot", slot)
            self.assertNotIn("value", slot)
            self.assertNotIn("key", slot)


class PostJsonRetryTest(unittest.TestCase):
    """Empty content is a transient failure retried inside one bounded budget."""

    def _call(self) -> tuple[str | None, str | None]:
        return locomo_reader._post_json(
            [{"role": "user", "content": "question"}],
            "dummy/model",
            DUMMY_KEY,
            "https://example.invalid/v1",
            16,
        )

    def test_post_json_retries_empty_content_then_returns_valid_content(self) -> None:
        responses = [
            _FakeResponse(_chat_body("")),       # attempt 1: blank content
            _FakeResponse(_chat_body("   ")),    # attempt 2: whitespace content
            _FakeResponse(_chat_body("7 May 2023")),  # attempt 3: valid answer
        ]
        with patch.object(
            locomo_reader.urllib.request, "urlopen", side_effect=responses
        ) as urlopen, patch.object(locomo_reader.time, "sleep") as sleep:
            answer, error = self._call()

        self.assertEqual((answer, error), ("7 May 2023", None))
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_post_json_returns_empty_answer_after_exhausting_retries(self) -> None:
        attempts = locomo_reader._RETRIES + 1
        responses = [_FakeResponse(_chat_body("")) for _ in range(attempts)]
        with patch.object(
            locomo_reader.urllib.request, "urlopen", side_effect=responses
        ) as urlopen, patch.object(locomo_reader.time, "sleep") as sleep:
            answer, error = self._call()

        self.assertIsNone(answer)
        self.assertEqual(error, "empty_answer")
        self.assertEqual(urlopen.call_count, attempts)
        self.assertEqual(sleep.call_count, locomo_reader._RETRIES)

    def test_post_json_no_key_makes_no_network_call(self) -> None:
        with patch.object(
            locomo_reader.urllib.request,
            "urlopen",
            side_effect=AssertionError("network must not be touched"),
        ) as urlopen:
            answer, error = locomo_reader._post_json(
                [{"role": "user", "content": "question"}],
                "dummy/model",
                "",
                "https://example.invalid/v1",
                16,
            )

        self.assertEqual((answer, error), (None, "no_key"))
        urlopen.assert_not_called()

    def test_post_json_no_key_also_covers_none(self) -> None:
        # Guard the ``str(api_key or '')`` path with None as well as "".
        with patch.object(
            locomo_reader.urllib.request, "urlopen"
        ) as urlopen:
            answer, error = locomo_reader._post_json(
                [{"role": "user", "content": "question"}],
                "dummy/model",
                None,  # type: ignore[arg-type]
                "https://example.invalid/v1",
                16,
            )
        self.assertEqual((answer, error), (None, "no_key"))
        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
