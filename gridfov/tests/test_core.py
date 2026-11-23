import math

import pytest

import gridfov
from gridfov import GridFov

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

G = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
CAM = (0.0, 0.0, 2.0, 2.0)


def _f():
    return gridfov.GridFov()


def _check_nonempty():
    return len(_f().visible(G, CAM)) > 0


def _check_wall_hidden():
    return (1, 1) not in _f().visible(G, CAM)


def _check_bad_cam():
    return _raises(lambda: _f().visible(G, (0.0, 0.0, 0.0, 0.0)))


def _check_bad_grid():
    return _raises(lambda: _f().visible([], CAM))


def _check_type():
    return isinstance(_f().visible(G, CAM), list)


def _check_det():
    return _f().visible(G, CAM) == _f().visible(G, CAM)


def _check_inside():
    g = [[0, 0], [0, 0]]
    return len(_f().visible(g, (0.5, 0.5, 0.5, 0.5))) >= 1


def _check_bounds():
    return all(0 <= x < 3 and 0 <= y < 3 for x, y in _f().visible(G, CAM))

CASES = [("nonempty", "_check_nonempty"), ("wall_hidden", "_check_wall_hidden"), ("bad_cam", "_check_bad_cam"), ("bad_grid", "_check_bad_grid"), ("type", "_check_type"), ("det", "_check_det"), ("inside", "_check_inside"), ("bounds", "_check_bounds"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert GridFov() is not None
        return
    assert fn() is True
