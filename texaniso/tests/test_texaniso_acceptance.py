"""texaniso 验收自测（新增文件，不修改既有 tests/test_core.py）。

覆盖题面的可验证约束，用例名均含 ``aniso``，可用
``python -m pytest tests -q -k aniso`` 单独选中：

* 权重和为 1，采样数不超过上限；
* 各向异性比按长轴/短轴取，比例误差 <= 1e-6；
* 采样数随比例单调增加且不超过上限；
* 与暴力超采样对照 PSNR >= 40 dB；
* 零导数与非有限导数可区分报错；
* 10 万次 <= 150 ms；
* 同一实例可被多线程并发调用。
"""

import math
import random
import threading
import time

import pytest

from texaniso import Aniso
from texaniso.core import NonFiniteDerivativeError, ZeroDerivativeError

NAN = float("nan")
INF = float("inf")


def _a():
    return Aniso()


# ---------------------------------------------------------------- 权重与计数

def test_aniso_weights_sum_to_one():
    a = _a()
    for du, dv in [
        ((4.0, 0.0), (0.0, 0.1)),
        ((1.0, 0.0), (0.0, 1.0)),
        ((1.0, 2.0), (3.0, 4.0)),
        ((0.1, 0.2), (0.7, 0.9)),
    ]:
        w = a.weights(du, dv, 8)
        assert abs(sum(w) - 1.0) < 1e-12
        assert all(x > 0.0 for x in w)


def test_aniso_count_within_cap():
    a = _a()
    for cap in (1, 2, 3, 8, 16, 64):
        for ratio in (1.0, 1.5, 2.0, 3.7, 8.0, 40.0):
            n = len(a.weights((ratio, 0.0), (0.0, 1.0), cap))
            assert 1 <= n <= cap


# ---------------------------------------------------------------- 比例精度

def _ratio_ref(du, dv):
    lu = math.hypot(du[0], du[1])
    lv = math.hypot(dv[0], dv[1])
    return max(lu, lv) / min(lu, lv)


@pytest.mark.parametrize(
    "du,dv",
    [
        ((4.0, 0.0), (0.0, 0.1)),
        ((1.0, 0.0), (0.0, 1.0)),
        ((1.0, 2.0), (3.0, 4.0)),
        ((0.1, 0.2), (0.7, 0.9)),
        ((1e6, 3.0), (0.5, -2.0)),
        ((3.0, 4.0), (8.0, 6.0)),
    ],
)
def test_aniso_ratio_accuracy(du, dv):
    r = _a().anisotropy(du, dv)
    ref = _ratio_ref(du, dv)
    assert abs(r - ref) / ref <= 1e-6


def test_aniso_ratio_at_least_one():
    a = _a()
    assert a.anisotropy((1.0, 0.0), (0.0, 1.0)) == 1.0
    assert a.anisotropy((4.0, 0.0), (0.0, 1.0)) == 4.0
    assert a.anisotropy((0.0, 1.0), (4.0, 0.0)) == 4.0


# ---------------------------------------------------------------- 单调性

def test_aniso_count_monotone_nondecreasing():
    a = _a()
    prev = 0
    for k in range(1, 2000):
        ratio = 1.0 + k * 0.01
        n = len(a.weights((ratio, 0.0), (0.0, 1.0), 16))
        assert n >= prev
        assert n <= 16
        prev = n


# ---------------------------------------------------------------- 暴力对照

def _tex(x):
    return (
        0.5
        + 0.3 * math.sin(2.0 * math.pi * x / 64.0 * 2.0)
        + 0.15 * math.sin(2.0 * math.pi * x / 64.0 * 5.0 + 0.7)
    )


def _bilinear(x):
    x0 = math.floor(x)
    f = x - x0
    x0 = min(63, max(0, x0))
    x1 = min(63, max(0, x0 + 1))
    return _tex(x0) * (1.0 - f) + _tex(x1) * f


def _aniso_filter(center, ratio, n):
    acc = 0.0
    span = ratio
    for i in range(n):
        s = center + (i + 0.5) / n * span - 0.5 * span
        acc += _bilinear(s)
    return acc / n


def _brute(center, ratio, n=8192):
    acc = 0.0
    span = ratio
    for i in range(n):
        s = center + (i + 0.5) / n * span - 0.5 * span
        acc += _bilinear(s)
    return acc / n


def test_aniso_matches_brute_force_psnr():
    a = _a()
    rng = random.Random(7)
    sigma = 0.35
    mse = 0.0
    count = 0
    for ratio in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        for _ in range(40):
            center = rng.uniform(8.0, 56.0)
            n = len(a.weights((ratio, 0.0), (0.0, 1.0), 32))
            d = _aniso_filter(center, ratio, n) - _brute(center, ratio)
            mse += d * d
            count += 1
    mse /= count
    psnr = 10.0 * math.log10(sigma * sigma / mse) if mse > 0.0 else INF
    assert psnr >= 40.0


# ---------------------------------------------------------------- 错误分类

def test_aniso_zero_derivative_is_distinct():
    a = _a()
    with pytest.raises(ZeroDerivativeError):
        a.weights((0.0, 0.0), (0.0, 1.0), 8)
    with pytest.raises(ZeroDerivativeError):
        a.weights((1.0, 0.0), (0.0, 0.0), 8)
    # 零导数不得与非有限导数混为同类
    assert not issubclass(ZeroDerivativeError, NonFiniteDerivativeError)
    assert not issubclass(NonFiniteDerivativeError, ZeroDerivativeError)


def test_aniso_non_finite_derivative_is_distinct():
    a = _a()
    with pytest.raises(NonFiniteDerivativeError):
        a.weights((NAN, 0.0), (0.0, 1.0), 8)
    with pytest.raises(NonFiniteDerivativeError):
        a.weights((1.0, 0.0), (0.0, INF), 8)


def test_aniso_errors_are_value_errors():
    a = _a()
    with pytest.raises(ValueError):
        a.weights((0.0, 0.0), (0.0, 1.0), 8)
    with pytest.raises(ValueError):
        a.weights((NAN, 0.0), (0.0, 1.0), 8)
    with pytest.raises(ValueError):
        a.weights((1.0, 0.0), (0.0, 1.0), 0)


# ---------------------------------------------------------------- 性能

def test_aniso_perf_100k_under_150ms():
    a = _a()
    start = time.perf_counter()
    for i in range(100000):
        a.weights((4.0, 0.0 + (i % 7) * 0.1), (0.0, 0.1 + (i % 5) * 0.02), 8)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms <= 150.0


# ---------------------------------------------------------------- 并发

def test_aniso_concurrent_consistency():
    a = _a()
    results = []
    errors = []

    def worker():
        try:
            local = [a.weights((4.0, 0.0), (0.0, 0.1), 8) for _ in range(5000)]
            results.append(local)
        except Exception as exc:  # noqa: BLE001 - 记录并让主线程断言
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(r == results[0] for r in results)
