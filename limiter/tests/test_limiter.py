"""题面约束的验收用例：真峰值、前视预衰减、release 单调恢复、
跨缓冲连续、性能上界、非法参数/非有限输入分类报错、并发安全。"""

import threading
import time

import numpy as np
import pytest

import limiter
from limiter import Limiter


def _oversampled_peak(x, factor=4):
    """4 倍过采样（线性插值）近似真峰值。"""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    t_old = np.arange(x.size)
    t_new = np.linspace(0.0, x.size - 1, x.size * factor - (factor - 1))
    return float(np.max(np.abs(np.interp(t_new, t_old, x))))


def test_limit_true_peak_under_ceiling():
    lim = Limiter(0.9, 16, 0.05)
    rng = np.random.default_rng(0)
    t = np.arange(4096)
    sig = 1.4 * np.sin(2 * np.pi * t / 37.0) + 0.3 * rng.standard_normal(t.size)
    out = lim.process(sig.tolist())
    assert _oversampled_peak(out) <= 0.9 + 1e-3
    assert max(abs(v) for v in out) <= 0.9 + 1e-3


def test_limit_lookahead_preduck_no_overshoot():
    lim = Limiter(0.5, 32, 0.05)
    sig = [0.2] * 64 + [2.0] * 64 + [0.2] * 64
    out = lim.process(sig)
    # 峰值到达前增益已开始下降（预衰减）
    assert abs(out[63]) < 0.2
    # 全程无过冲
    assert max(abs(v) for v in out) <= 0.5 + 1e-3


def test_limit_release_monotonic_recovery():
    lim = Limiter(0.5, 8, 0.05)
    lim.process([1.0] * 64)
    out = lim.process([0.25] * 64)
    gains = [v / 0.25 for v in out]
    # 恢复阶段单调不减，并最终回到 1
    assert all(g2 >= g1 - 1e-12 for g1, g2 in zip(gains, gains[1:]))
    assert gains[-1] == pytest.approx(1.0)


def test_limit_cross_buffer_continuity():
    a = Limiter(0.9, 8, 0.1)
    b = Limiter(0.9, 8, 0.1)
    sig = [1.0] * 200
    one_shot = a.process(sig)
    chunked = b.process(sig[:100]) + b.process(sig[100:])
    # 缓冲边界处增益不跳变，持续超限信号持续被限幅
    assert abs(chunked[100] - chunked[99]) < 0.05
    assert all(abs(v) <= 0.9 + 1e-3 for v in chunked)
    assert chunked[-1] == pytest.approx(one_shot[-1])


def test_limit_performance_1m_samples():
    lim = Limiter(0.9, 64, 0.001)
    rng = np.random.default_rng(1)
    sig = (1.5 * rng.standard_normal(1_000_000)).tolist()
    start = time.perf_counter()
    out = lim.process(sig)
    elapsed = time.perf_counter() - start
    assert len(out) == 1_000_000
    assert elapsed < 0.4


def test_limit_param_errors_distinguishable():
    with pytest.raises(TypeError):
        Limiter("0.9", 8, 0.1)
    with pytest.raises(ValueError):
        Limiter(float("nan"), 8, 0.1)
    with pytest.raises(ValueError):
        Limiter(-1.0, 8, 0.1)
    with pytest.raises(TypeError):
        Limiter(0.9, 8.5, 0.1)
    with pytest.raises(ValueError):
        Limiter(0.9, -3, 0.1)
    with pytest.raises(ValueError):
        Limiter(0.9, 8, 0.0)
    with pytest.raises(ValueError):
        Limiter(0.9, 8, float("inf"))


def test_limit_nonfinite_input_classified():
    lim = Limiter(0.9, 8, 0.1)
    with pytest.raises(ValueError):
        lim.process([0.1, float("nan"), 0.2])
    with pytest.raises(ValueError):
        lim.process([float("inf")])
    with pytest.raises(TypeError):
        lim.process(["a", "b"])
    with pytest.raises(TypeError):
        lim.process(0.5)


def test_limit_concurrent_process_safe():
    lim = Limiter(0.9, 8, 0.1)
    errors = []

    def worker():
        try:
            for _ in range(50):
                out = lim.process([1.0] * 256)
                assert len(out) == 256
                assert max(abs(v) for v in out) <= 0.9 + 1e-3
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
