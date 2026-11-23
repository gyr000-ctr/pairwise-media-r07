import math

import pytest

import limiter

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _l():
    return limiter.Limiter(0.9, 8, 0.1)


def _check_len():
    return len(_l().process([0.1] * 10)) == 10


def _check_peak():
    out = _l().process([1.0] * 100)
    return max(abs(v) for v in out) <= 0.9 + 1e-3


def _check_bad_ceiling():
    return _raises(lambda: limiter.Limiter(0.0, 8, 0.1))


def _check_bad_lookahead():
    return _raises(lambda: limiter.Limiter(0.9, 0, 0.1))


def _check_empty():
    return _l().process([]) == []


def _check_type():
    return isinstance(_l().process([0.1]), list)


def _check_det():
    return _l().process([0.5] * 5) == _l().process([0.5] * 5)


def _check_no_jump():
    out = _l().process([0.0, 1.0, 0.0])
    return max(abs(out[i + 1] - out[i]) for i in range(len(out) - 1)) < 1.0

CASES = [("len", "_check_len"), ("peak", "_check_peak"), ("bad_ceiling", "_check_bad_ceiling"), ("bad_lookahead", "_check_bad_lookahead"), ("empty", "_check_empty"), ("type", "_check_type"), ("det", "_check_det"), ("no_jump", "_check_no_jump"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Limiter() is not None
        return
    assert fn() is True
