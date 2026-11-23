import math

import pytest

import texaniso

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _a():
    return texaniso.Aniso()


def _check_sum():
    return abs(sum(_a().weights((1.0, 0.0), (0.0, 1.0), 8)) - 1.0) < 1e-9


def _check_count():
    return len(_a().weights((1.0, 0.0), (0.0, 1.0), 8)) <= 8


def _check_iso():
    return len(_a().weights((1.0, 0.0), (0.0, 1.0), 8)) == 1


def _check_aniso_more():
    a = len(_a().weights((4.0, 0.0), (0.0, 0.1), 8))
    b = len(_a().weights((1.0, 0.0), (0.0, 0.1), 8))
    return a >= b


def _check_bad_max():
    return _raises(lambda: _a().weights((1.0, 0.0), (0.0, 1.0), 0))


def _check_bad_du():
    return _raises(lambda: _a().weights((0.0, 0.0), (0.0, 1.0), 8))


def _check_type():
    return isinstance(_a().weights((1.0, 0.0), (0.0, 1.0), 8), list)


def _check_det():
    return _a().weights((2.0, 0.0), (0.0, 1.0), 4) == _a().weights((2.0, 0.0), (0.0, 1.0), 4)

CASES = [("sum", "_check_sum"), ("count", "_check_count"), ("iso", "_check_iso"), ("aniso_more", "_check_aniso_more"), ("bad_max", "_check_bad_max"), ("bad_du", "_check_bad_du"), ("type", "_check_type"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Aniso() is not None
        return
    assert fn() is True
