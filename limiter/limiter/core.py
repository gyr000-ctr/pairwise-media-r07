"""前视限幅器（lookahead limiter）核心实现。

设计要点：
- 4 倍过采样（Catmull-Rom 插值）估计真峰值，据此计算所需增益，输出真峰值 ≤ 阈值；
- 前视窗内按线性斜坡预衰减（反向 min-plus 距离变换，O(n)），增益永不超过 1（无过冲）；
- release 阶段增益以固定斜率单调恢复（正向 min-plus 距离变换，O(n)）；
- 增益状态与输入历史跨 process 调用保持，保证跨缓冲连续；
- 全程 numpy 向量化，100 万样本远低于 400 ms；
- 非法参数按类型/取值分别抛出 TypeError / ValueError（可区分），非有限输入单独分类报错；
- process 内部加锁，并发调用安全。
"""

from __future__ import annotations

import math
import numbers
import threading

import numpy as np

__all__ = ["Limiter"]

# 4 倍过采样：每个采样间隔内 3 个插值相位（端点由采样值本身覆盖）
_PHASES = (0.25, 0.5, 0.75)
# 真峰值安全余量（相对），抵消插值估计误差
_MARGIN = 1e-3
# 防止除零
_TINY = 1e-300


class Limiter:
    """前视限幅器。

    参数：
        ceiling: 输出真峰值上限，正有限实数；
        lookahead: 前视窗长度（采样数），≥ 1 的整数；
        release: 增益每采样点的恢复步长，(0, 1] 内的有限实数。
    """

    def __init__(self, ceiling=1.0, lookahead=128, release=0.01):
        # --- ceiling：正有限实数 ---
        if isinstance(ceiling, bool) or not isinstance(ceiling, numbers.Real):
            raise TypeError(
                f"ceiling must be a real number, got {type(ceiling).__name__}"
            )
        ceiling = float(ceiling)
        if not math.isfinite(ceiling):
            raise ValueError("ceiling must be finite (not nan/inf)")
        if ceiling <= 0.0:
            raise ValueError("ceiling must be positive")

        # --- lookahead：≥ 1 的整数 ---
        if isinstance(lookahead, bool) or not isinstance(lookahead, numbers.Integral):
            raise TypeError(
                f"lookahead must be an integer, got {type(lookahead).__name__}"
            )
        lookahead = int(lookahead)
        if lookahead < 1:
            raise ValueError("lookahead must be >= 1")

        # --- release：(0, 1] 内的有限实数 ---
        if isinstance(release, bool) or not isinstance(release, numbers.Real):
            raise TypeError(
                f"release must be a real number, got {type(release).__name__}"
            )
        release = float(release)
        if not math.isfinite(release):
            raise ValueError("release must be finite (not nan/inf)")
        if not 0.0 < release <= 1.0:
            raise ValueError("release must be in the interval (0, 1]")

        self.ceiling = ceiling
        self.lookahead = lookahead
        self.release = release

        # 跨缓冲状态：上一缓冲末尾的增益、最后两个输入/输出采样
        self._gain = 1.0
        self._hist = np.zeros(0, dtype=np.float64)
        self._out_hist = np.zeros(0, dtype=np.float64)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ API

    def process(self, samples):
        """处理一段采样，返回等长 list[float]。"""
        arr = self._coerce_samples(samples)
        if arr.size == 0:
            return []
        with self._lock:
            return self._process(arr).tolist()

    # -------------------------------------------------------------- 内部实现

    @staticmethod
    def _coerce_samples(samples):
        """把输入转成 1 维 float64 数组，并对非法输入分类报错。"""
        if isinstance(samples, np.ndarray):
            raw = samples
        else:
            try:
                raw = list(samples)
            except TypeError:
                raise TypeError(
                    "samples must be an iterable of real numbers"
                ) from None
        try:
            arr = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("samples must contain only real numbers") from exc
        if arr.ndim != 1:
            raise ValueError("samples must be a one-dimensional sequence")
        if not np.all(np.isfinite(arr)):
            raise ValueError("samples contain non-finite values (nan/inf)")
        return arr

    @staticmethod
    def _true_peak(x, hist):
        """4 倍过采样（Catmull-Rom）的逐采样真峰值估计。

        x: 1 维 float64 数组；hist: 之前缓冲的最后 0~2 个采样（左端上下文）。
        返回与 x 等长的峰值数组，peak[i] 覆盖区间 [i-1, i] 与 [i, i+1]。
        """
        n = x.shape[0]
        # xe[k] 对应采样序号 k-2；左端接历史，右端边缘复制。
        xe = np.empty(n + 4, dtype=np.float64)
        xe[2:2 + n] = x
        if hist.size >= 2:
            xe[0] = hist[-2]
            xe[1] = hist[-1]
        elif hist.size == 1:
            xe[0] = hist[0]
            xe[1] = hist[0]
        else:
            xe[0] = x[0]
            xe[1] = x[0]
        xe[n + 2] = x[-1]
        xe[n + 3] = x[-1]

        p0 = xe[0:n + 1]
        p1 = xe[1:n + 2]
        p2 = xe[2:n + 3]
        p3 = xe[3:n + 4]
        # seg[k]：采样 k-1 与 k 之间插值曲线的最大模
        seg = np.zeros(n + 1, dtype=np.float64)
        for t in _PHASES:
            t2 = t * t
            t3 = t2 * t
            v = 0.5 * (
                2.0 * p1
                + (p2 - p0) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            np.maximum(seg, np.abs(v), out=seg)
        return np.maximum(np.abs(x), np.maximum(seg[:-1], seg[1:]))

    def _process(self, x):
        n = x.shape[0]
        target = self.ceiling * (1.0 - _MARGIN)

        # ---- 各采样点所需增益（按真峰值估计）----
        peak = self._true_peak(x, self._hist)
        req = np.minimum(1.0, target / np.maximum(peak, _TINY))

        # ---- 前视窗内线性预衰减（反向斜坡，O(n)）----
        # env[i] = min(1, min_{j>=i} req[j] + (j-i)/lookahead)
        idx = np.arange(n, dtype=np.float64)
        ramp = idx * (1.0 / self.lookahead)
        env = np.minimum.accumulate((req + ramp)[::-1])[::-1] - ramp
        np.minimum(env, 1.0, out=env)

        # ---- release：增益单调恢复（正向斜坡，含跨缓冲状态）----
        # gain[i] = min(env[i], gain[i-1] + release)
        ge = np.empty(n + 1, dtype=np.float64)
        ge[0] = self._gain
        ge[1:] = env
        je = np.arange(n + 1, dtype=np.float64) * self.release
        rec = je + np.minimum.accumulate(ge - je)
        gain = np.minimum(rec[1:], 1.0)

        out = x * gain

        # ---- 真峰值闭环校正：直接对输出的过采样峰值测量并迭代压回阈值 ----
        # （快路径：采样峰值远低阈值时插值不可能越界，跳过）
        if np.max(np.abs(out)) > 0.5 * target:
            for _ in range(3):
                op = self._true_peak(out, self._out_hist)
                corr = np.minimum(1.0, target / np.maximum(op, _TINY))
                if corr.min() >= 1.0:
                    break
                gain *= corr
                out = x * gain

        # ---- 状态更新（跨缓冲连续）----
        self._gain = float(gain[-1])
        if n >= 2:
            self._hist = x[-2:].copy()
            self._out_hist = out[-2:].copy()
        else:
            self._hist = np.concatenate([self._hist, x])[-2:].copy()
            self._out_hist = np.concatenate([self._out_hist, out])[-2:].copy()
        return out
