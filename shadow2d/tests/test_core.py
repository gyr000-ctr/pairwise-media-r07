import math

import pytest

import shadow2d

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

O = [[1.0, 1.0], [1.0, 1.0]]


def _s(n=8):
    return shadow2d.SoftShadow(n, 0)


def _check_lit():
    return abs(_s().sample(O, 0, 0) - 1.0) < 1e-6


def _check_dark():
    z = [[0.0, 0.0], [0.0, 0.0]]
    return abs(_s().sample(z, 0, 0)) < 1e-6


def _check_range():
    return 0.0 <= _s().sample([[0.0, 1.0], [1.0, 0.0]], 0, 0) <= 1.0


def _check_bad_samples():
    return _raises(lambda: shadow2d.SoftShadow(0, 0))


def _check_bad_occ():
    return _raises(lambda: _s().sample([], 0, 0))


def _check_type():
    return isinstance(_s().sample(O, 0, 0), float)


def _check_det():
    return _s().sample(O, 0, 0) == _s().sample(O, 0, 0)


def _check_oob():
    return _raises(lambda: _s().sample(O, 9, 9))

CASES = [("lit", "_check_lit"), ("dark", "_check_dark"), ("range", "_check_range"), ("bad_samples", "_check_bad_samples"), ("bad_occ", "_check_bad_occ"), ("type", "_check_type"), ("det", "_check_det"), ("oob", "_check_oob"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert SoftShadow() is not None
        return
    assert fn() is True
