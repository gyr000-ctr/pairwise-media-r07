"""shadow2d 软阴影验收用例：收敛、复现、边界、并发、性能、非法参数。"""

import concurrent.futures
import math

import pytest

from shadow2d import SoftShadow

GRAD = [[(x + y) / 16.0 for x in range(9)] for y in range(9)]


def test_shadow_lit_dark_exact():
    assert SoftShadow(8, 0).sample([[1.0, 1.0], [1.0, 1.0]], 0, 0) == 1.0
    assert SoftShadow(8, 0).sample([[0.0, 0.0], [0.0, 0.0]], 1, 1) == 0.0


def test_shadow_converges_with_samples():
    ref = SoftShadow(4096, 0).sample(GRAD, 4, 4)
    err_lo = abs(SoftShadow(8, 0).sample(GRAD, 4, 4) - ref)
    err_hi = abs(SoftShadow(512, 0).sample(GRAD, 4, 4) - ref)
    assert err_hi <= err_lo + 1e-9
    assert err_hi < 0.02


def test_shadow_reproducible_same_seed():
    a = SoftShadow(32, 7)
    b = SoftShadow(32, 7)
    assert a.sample(GRAD, 3, 5) == b.sample(GRAD, 3, 5)


def test_shadow_boundary_pixels_in_range():
    s = SoftShadow(16, 0)
    for y in (0, 8):
        for x in (0, 8):
            v = s.sample(GRAD, x, y)
            assert 0.0 <= v <= 1.0


def test_shadow_thread_safe():
    s = SoftShadow(16, 0)
    expected = [s.sample(GRAD, x % 9, x // 9) for x in range(81)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(lambda i: s.sample(GRAD, i % 9, i // 9), range(81)))
    assert got == expected


def test_shadow_invalid_params():
    for bad in (0, -1, 2.5, float("nan"), float("inf"), "8", None):
        with pytest.raises(ValueError):
            SoftShadow(bad, 0)
    s = SoftShadow(8, 0)
    with pytest.raises(ValueError):
        s.sample([], 0, 0)
    with pytest.raises(ValueError):
        s.sample([[1.0], [1.0, 0.0]], 0, 0)
    with pytest.raises(ValueError):
        s.sample([[float("nan")]], 0, 0)
    with pytest.raises(ValueError):
        s.sample([[1.0]], 5, 0)
    with pytest.raises(ValueError):
        s.sample([[1.0]], 0, -1)
    with pytest.raises(ValueError):
        s.sample([[1.0]], 0.5, 0)
    with pytest.raises(ValueError):
        s.sample([[1.0]], float("inf"), 0)


def test_shadow_perf_10k_pixels():
    occ = [[(x * y) % 2 * 1.0 for x in range(100)] for y in range(100)]
    s = SoftShadow(64, 0)
    import time

    t0 = time.perf_counter()
    for i in range(10000):
        s.sample(occ, i % 100, i // 100)
    assert time.perf_counter() - t0 < 0.4
