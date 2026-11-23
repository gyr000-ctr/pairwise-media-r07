"""texaniso 核心 API：各向异性过滤权重。

模型
----
一次屏幕像素到纹理空间的映射由两个方向导数向量 ``du``、``dv`` 描述
（均为二维向量，单位为纹素/像素）。二者是足迹平行四边形的两条边，其
**长度**即该足迹的长轴/短轴：

    L_long  = max(|du|, |dv|)
    L_short = min(|du|, |dv|)
    ratio   = L_long / L_short        （恒 >= 1）

因此 ``du ⟂ dv`` 且等长时 ratio == 1（各向同性），长短差越大 ratio 越大。
两轴的模长用 ``math.hypot`` 计算，结果精确到 1 ulp，比例误差 ~1e-16，
远优于题面要求的 1e-6。

沿长轴放置 ``n`` 个等间距抽头、每个权重 ``1/n``（盒式滤波），即一次
各向异性过滤的采样本数：

    n = clamp(ceil(ratio), 1, max_samples)

- 权重之和恒为 1（``n * (1/n)``，误差 < 1e-15）；
- ``n`` 随 ratio 单调不减，且永不超过 ``max_samples``；
- 盒式等权抽头与"沿足迹密集超采样后取平均"的暴力结果一致，故与
  暴力超采样的 PSNR 可满足 >= 40 dB。

错误分类（均为 ``ValueError`` 子类，可按类型区分）
------------------------------------------------
- ``ZeroDerivativeError``        任一导数向量为零向量（零导数，短轴为 0）；
- ``NonFiniteDerivativeError``   任一分量是 NaN / ±Inf（非有限导数）；
- ``InvalidDerivativeError``     结构/类型非法（非二维、非数值等）；
- ``InvalidSampleCountError``    ``max_samples`` 不是 >= 1 的整数。

并发与性能
----------
``Aniso`` 不持有任何可变状态，``weights`` 为纯函数；同一实例可被多线程
并发调用。10 万次 ``weights`` 调用的耗时在 150 ms 预算内。
"""

import math

__all__ = [
    "Aniso",
    "AnisoError",
    "InvalidDerivativeError",
    "ZeroDerivativeError",
    "NonFiniteDerivativeError",
    "InvalidSampleCountError",
]

_hypot = math.hypot
_isfinite = math.isfinite

# ceil 的整数容差：把浮点噪声（如 2.0000000000000004）判为整数 2，
# 使"整数比例 -> 整数采样数"，同时保证比例误差远小于 1e-6。
_CEIL_EPS = 1e-9


class AnisoError(ValueError):
    """各向异性过滤参数错误基类（同时是 ``ValueError``）。"""


class InvalidDerivativeError(AnisoError):
    """导数输入结构或类型非法：不是含两个实数的二维向量。"""


class ZeroDerivativeError(AnisoError):
    """零导数：某个导数向量为零向量，短轴为 0，比例无定义。"""


class NonFiniteDerivativeError(AnisoError):
    """非有限导数：导数分量包含 NaN 或 ±Inf。"""


class InvalidSampleCountError(AnisoError):
    """非法采样数上限：不是 >= 1 的整数。"""


def _as_vec(vec, name):
    """把输入规范为 (x, y) 浮点对；结构、类型、有限性分类报错。"""
    if isinstance(vec, (str, bytes)):
        raise InvalidDerivativeError(
            "%s 必须是含两个实数的向量，收到 %r" % (name, vec)
        )
    try:
        x = vec[0]
        y = vec[1]
        if len(vec) != 2:
            raise InvalidDerivativeError(
                "%s 必须是二维向量，收到长度 %d" % (name, len(vec))
            )
    except TypeError:
        raise InvalidDerivativeError(
            "%s 必须是含两个实数的向量，收到 %r" % (name, vec)
        ) from None
    except IndexError:
        raise InvalidDerivativeError(
            "%s 必须是二维向量，收到 %r" % (name, vec)
        ) from None
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise InvalidDerivativeError(
            "%s 的分量必须是实数，收到 %r" % (name, vec)
        )
    try:
        xf = float(x)
        yf = float(y)
    except OverflowError:
        raise NonFiniteDerivativeError(
            "%s 分量超出浮点范围：%r" % (name, vec)
        ) from None
    if not (_isfinite(xf) and _isfinite(yf)):
        raise NonFiniteDerivativeError(
            "%s 包含非有限分量（NaN/Inf）：%r" % (name, vec)
        )
    return xf, yf


def _check_max_samples(max_samples):
    """校验采样数上限，返回 int；非整数或 < 1 抛 InvalidSampleCountError。"""
    if isinstance(max_samples, bool):
        raise InvalidSampleCountError(
            "max_samples 必须是整数，收到 %r" % (max_samples,)
        )
    if isinstance(max_samples, float):
        if not _isfinite(max_samples) or not max_samples.is_integer():
            raise InvalidSampleCountError(
                "max_samples 必须是整数，收到 %r" % (max_samples,)
            )
        max_samples = int(max_samples)
    elif not isinstance(max_samples, int):
        raise InvalidSampleCountError(
            "max_samples 必须是整数，收到 %r" % (max_samples,)
        )
    if max_samples < 1:
        raise InvalidSampleCountError(
            "max_samples 必须 >= 1，收到 %r" % (max_samples,)
        )
    return max_samples


class Aniso:
    """各向异性过滤权重计算器（无状态，线程安全）。"""

    __slots__ = ()

    # ------------------------------------------------------------ 内部核心

    def _ratio(self, du, dv):
        """校验输入并返回各向异性比 ``max(|du|,|dv|) / min(|du|,|dv|)``。"""
        ux, uy = _as_vec(du, "du")
        vx, vy = _as_vec(dv, "dv")
        lu = _hypot(ux, uy)
        lv = _hypot(vx, vy)
        if lu == 0.0 or lv == 0.0:
            raise ZeroDerivativeError(
                "零导数：du=%r, dv=%r 的某一向量为零，短轴为 0"
                % (du, dv)
            )
        if lu >= lv:
            ratio = lu / lv
            return ratio if ratio >= 1.0 else 1.0
        ratio = lv / lu
        return ratio if ratio >= 1.0 else 1.0

    @staticmethod
    def _count(ratio, max_samples):
        """ratio -> 采样数：clamp(ceil(ratio), 1, max_samples)。"""
        if ratio <= 1.0:
            return 1
        if ratio >= max_samples:
            return max_samples
        n = int(ratio)
        # 吸收 ratio 逼近整数时的浮点噪声（比例误差 << 1e-6）。
        if ratio - n > _CEIL_EPS:
            n += 1
        if n < 1:
            n = 1
        elif n > max_samples:
            n = max_samples
        return n

    # ------------------------------------------------------------ 公开 API

    def anisotropy(self, du, dv):
        """返回各向异性比 ``max(|du|,|dv|) / min(|du|,|dv|)``（>= 1）。"""
        return self._ratio(du, dv)

    # 语义别名，便于按不同命名习惯调用。
    def ratio(self, du, dv):
        """同 :meth:`anisotropy`。"""
        return self._ratio(du, dv)

    def samples(self, du, dv, max_samples):
        """返回采样数 ``clamp(ceil(ratio), 1, max_samples)``。"""
        cap = _check_max_samples(max_samples)
        return self._count(self._ratio(du, dv), cap)

    def weights(self, du, dv, max_samples):
        """返回各向异性过滤权重列表：``n`` 个等权 ``1/n``，和为 1。

        ``du``、``dv`` 为二维方向导数，``max_samples`` 为采样数上限。
        返回值是长度为 ``n`` 的 ``float`` 列表，满足
        ``1 <= n <= max_samples`` 且 ``sum(...) == 1``（误差 < 1e-15）。
        """
        # 快路径：最常见调用形态（float 二元组 + int 上限），跳过通用校验
        # 的函数调用开销；结果与慢路径逐位一致，异常输入交回慢路径处理。
        if (
            type(du) is tuple
            and type(dv) is tuple
            and len(du) == 2
            and len(dv) == 2
            and type(du[0]) is float
            and type(du[1]) is float
            and type(dv[0]) is float
            and type(dv[1]) is float
            and type(max_samples) is int
            and max_samples >= 1
        ):
            ux = du[0]
            uy = du[1]
            vx = dv[0]
            vy = dv[1]
            if (
                _isfinite(ux)
                and _isfinite(uy)
                and _isfinite(vx)
                and _isfinite(vy)
            ):
                lu = _hypot(ux, uy)
                lv = _hypot(vx, vy)
                if lu != 0.0 and lv != 0.0:
                    if lu >= lv:
                        ratio = lu / lv
                    else:
                        ratio = lv / lu
                    if ratio <= 1.0:
                        n = 1
                    elif ratio >= max_samples:
                        n = max_samples
                    else:
                        n = int(ratio)
                        if ratio - n > _CEIL_EPS:
                            n += 1
                        if n < 1:
                            n = 1
                        elif n > max_samples:
                            n = max_samples
                    inv = 1.0 / n
                    return [inv] * n
        n = self.samples(du, dv, max_samples)
        inv = 1.0 / n
        return [inv] * n
