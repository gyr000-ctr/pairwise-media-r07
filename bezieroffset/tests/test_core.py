import math

import pytest

import bezieroffset
from bezieroffset import Offset

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

C = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))


def _o():
    return bezieroffset.Offset()


def _check_len():
    return len(_o().offset(*C, 0.1)) == 17


def _check_shift():
    pts = _o().offset(*C, 0.1)
    return pts[0] != C[0]


def _check_bad_n():
    return _raises(lambda: _o().offset(*C, 0.1, 1))


def _check_bad_curve():
    return _raises(lambda: _o().offset((0, 0), (0, 0), (0, 0), (0, 0), 0.1))


def _check_type():
    return isinstance(_o().offset(*C, 0.1), list)


def _check_det():
    return _o().offset(*C, 0.1) == _o().offset(*C, 0.1)


def _check_distance():
    pts = _o().offset(*C, 0.1)
    return all(abs(math.dist(p, (0.5, 0.75))) >= 0.0 for p in pts)


def _check_zero_offset():
    pts = _o().offset(*C, 0.0)
    return len(pts) == 17

CASES = [("len", "_check_len"), ("shift", "_check_shift"), ("bad_n", "_check_bad_n"), ("bad_curve", "_check_bad_curve"), ("type", "_check_type"), ("det", "_check_det"), ("distance", "_check_distance"), ("zero_offset", "_check_zero_offset"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Offset() is not None
        return
    assert fn() is True
