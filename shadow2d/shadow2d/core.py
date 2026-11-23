"""shadow2d — 2D 软阴影（PCF，百分比更接近过滤）。

对目标像素邻域内的遮蔽值进行多次采样并加权平均，
得到 [0, 1] 区间内随采样数收敛的软阴影亮度：
全遮挡处为 0.0，全亮处为 1.0。
"""

from __future__ import annotations

import math
import random

__all__ = ["SoftShadow"]

# 黄金角：Vogel 螺旋采样，保证采样点在圆盘内低差异分布
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

_DEFAULT_SAMPLES = 16
_DEFAULT_RADIUS = 1.5


def _as_int(value, name):
    """把有限整数（int 或整数值 float）规范为 int，否则抛 ValueError。"""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise ValueError(f"{name} 必须是有限整数，收到 {value!r}")
    raise ValueError(f"{name} 必须是有限整数，收到 {type(value).__name__}")


def _validate_occ(occ):
    """校验遮蔽网格，返回 (宽, 高)；结构非法或含非有限值时抛 ValueError。"""
    if isinstance(occ, (str, bytes)) or not hasattr(occ, "__getitem__"):
        raise ValueError("occ 必须是非空的二维序列")
    try:
        height = len(occ)
    except TypeError:
        raise ValueError("occ 必须是非空的二维序列") from None
    if height == 0:
        raise ValueError("occ 不能为空")
    width = None
    for row in occ:
        if isinstance(row, (str, bytes)) or not hasattr(row, "__getitem__"):
            raise ValueError("occ 的每一行必须是非空序列")
        try:
            row_len = len(row)
        except TypeError:
            raise ValueError("occ 的每一行必须是非空序列") from None
        if row_len == 0:
            raise ValueError("occ 的行不能为空")
        if width is None:
            width = row_len
        elif row_len != width:
            raise ValueError("occ 的各行长度必须一致")
        for v in row:
            if isinstance(v, bool):
                continue
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"occ 含有非数值或非有限元素: {v!r}")
    return width, height


def _build_taps(n, radius, seed):
    """生成 n 个圆盘采样偏移（Vogel 螺旋 + 种子抖动），并按整数像素偏移合并权重。

    采样数越多，对邻域圆盘的覆盖越均匀，估计值随之收敛；
    相同 (n, radius, seed) 生成完全相同的采样模式，保证可复现。
    """
    rng = random.Random(seed)
    weights = {}
    for k in range(n):
        r = radius * math.sqrt((k + 0.5) / n)
        a = k * _GOLDEN_ANGLE + rng.uniform(-0.6, 0.6)
        key = (int(round(r * math.cos(a))), int(round(r * math.sin(a))))
        weights[key] = weights.get(key, 0) + 1
    # dict 保持插入顺序，结果确定；权重之和恒等于 n
    return tuple((dx, dy, w) for (dx, dy), w in weights.items())


class SoftShadow:
    """2D PCF 软阴影采样器。

    samples: 每像素采样数（正整数），越大结果越收敛；
    seed:    采样模式随机种子，同种子结果可复现；
    radius:  采样核半径（像素，有限且 >= 0），0 表示硬阴影。

    实例在构造后只读（sample 不修改任何采样状态），可安全用于多线程。
    """

    __slots__ = ("samples", "seed", "radius", "_taps", "_n", "_cache")

    def __init__(self, samples=_DEFAULT_SAMPLES, seed=0, radius=_DEFAULT_RADIUS):
        n = _as_int(samples, "samples")
        if n < 1:
            raise ValueError(f"samples 必须为正整数，收到 {samples!r}")
        seed_i = _as_int(seed, "seed")
        if (
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not math.isfinite(radius)
            or radius < 0
        ):
            raise ValueError(f"radius 必须是有限非负数，收到 {radius!r}")
        self.samples = n
        self.seed = seed_i
        self.radius = float(radius)
        self._n = n
        self._taps = _build_taps(n, self.radius, seed_i)
        # 按对象身份缓存 occ 的校验结果（只缓存结构，不缓存数值），
        # 避免逐像素重复全网格校验；持有引用可防止 id 被回收复用。
        self._cache = None

    def sample(self, occ, x, y):
        """返回像素 (x, y) 处的软阴影亮度，取值 [0.0, 1.0] 的 float。

        occ: 非空矩形二维序列，元素为有限数值（1 亮 / 0 遮挡，越界值按 [0,1] 截断）；
        x, y: 网格内整数像素坐标，越界或非法时抛 ValueError。
        """
        cache = self._cache
        if cache is not None and cache[0] is occ:
            width, height = cache[1], cache[2]
        else:
            width, height = _validate_occ(occ)
            self._cache = (occ, width, height)

        px = _as_int(x, "x")
        py = _as_int(y, "y")
        if px < 0 or px >= width or py < 0 or py >= height:
            raise ValueError(
                f"采样坐标 ({x!r}, {y!r}) 超出 {width}x{height} 网格范围"
            )

        total = 0.0
        w_max = width - 1
        h_max = height - 1
        for dx, dy, w in self._taps:
            cx = px + dx
            if cx < 0:
                cx = 0
            elif cx > w_max:
                cx = w_max
            cy = py + dy
            if cy < 0:
                cy = 0
            elif cy > h_max:
                cy = h_max
            v = occ[cy][cx]
            if v > 0.0:  # 负值按 0 截断
                total += w * (v if v <= 1.0 else 1.0)  # 大于 1 按 1 截断
        return float(total / self._n)
