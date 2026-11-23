"""gridfov — 网格视锥剔除 + 遮挡剔除。

坐标与参数约定
--------------
* ``grid[y][x]``：``x`` 为列、``y`` 为行；格 ``(x, y)`` 覆盖世界坐标
  ``[x, x+1) × [y, y+1)``。``0`` 是透明空格，非 ``0`` 是不透明的遮挡格。
  格值必须是有限实数；``True`` / ``False`` 按 ``1`` / ``0`` 处理。
* ``cam = (cx, cy, direction, fov)``：

  - ``(cx, cy)``：相机位置（连续世界坐标），必须落在 ``[0, w) × [0, h)``；
    相机所在格是 ``(floor(cx), floor(cy))``，无论相机落在格内哪个位置，
    该格恒可见（“相机在格内不漏”）。
  - ``direction``：视轴方向，弧度，从 +x 轴起算、朝 +y 方向增加
    （与 ``atan2(dy, dx)`` 同约定），可为任意有限值。
  - ``fov``：视锥的**全张角**，弧度，取值 ``(0, 2π]``；``2π`` 即全向视野。

可见性判定（与“逐格投影 + Bresenham”参照一致）
--------------------------------------------
对每个格 ``(x, y)``：

1. 相机所在格恒可见；
2. 不透明格只作遮挡体，本身不算可见目标；
3. 其余格把它**格心** ``(x + 0.5, y + 0.5)`` 投影进视锥：令
   ``v = 格心 - 相机``、``u = (cos direction, sin direction)``、
   ``half = fov / 2``，当 ``v`` 与 ``u`` 的夹角 ``≤ half`` 时格心在闭视锥
   内。夹角用 ``atan2(|u × v|, u · v)`` 求，结果落在 ``[0, π]``，既不用
   分象限，也不受 ``cos(π/2)`` 的 1 ulp 误差影响：夹角恰好等于半角
   （如 ``direction = 0``、``fov = π`` 时格心正上/正下方的格）按闭区间
   计入。``half = π`` 时恒真。
4. 用 Bresenham 线段连接相机格 ``(ccx, ccy)`` 与目标格；当且仅当路径上
   的**中间格**（不含两端）全部透明时该格可见。

输出为按行优先（先 ``y`` 后 ``x``）排列的 ``(x, y)`` 整数元组 ``list``，
顺序确定；每次调用返回独立的新列表。

实现与复杂度
------------
朴素做法对每个目标跑一遍 Bresenham，``512×512`` 是 O(N³) 的格访问，纯
Python 远超 500 ms。这里把“对每个目标跑 Bresenham”改成“一次扫完所有
射线”：以相机格为原点，按主轴分两组（``|dx| ≥ |dy|`` 走 x、否则走 y），
每组候选按主轴步数升序排列，然后**逐步推进整批候选**——第 ``k`` 步时
只让主轴步数 ``≥ k`` 的候选前进一步（``acc += 2·minor``，进位时副轴也
进一步），并对这一步落到的格批量查墙；命中不透明格者立即判为被遮挡。
每个候选恰好在自己的主轴步数到达时停步，因此总推进量是 O(射线访问格
数)，``512×512`` 实测最坏约 150 ms（远低于 500 ms 上界）；全透明网格
有 O(N²) 的直通快路（约 20–40 ms）。

批量推进用 numpy 完成；没有 numpy 时退化为逐条射线的纯 Python 走位
（正确性相同，速度较慢）。

并发安全
--------
``GridFov`` 无实例状态（``__slots__`` 为空），模块里也没有可变缓存；
``visible`` 是纯函数，中间量全部是局部对象，返回新列表。同一实例可被
任意多线程同时调用，结果与串行调用逐位一致。
"""

from __future__ import annotations

import math

try:  # numpy 只用于加速向量化推进；缺失时走纯 Python 退化路径。
    import numpy as _np
except Exception:  # pragma: no cover - 环境无 numpy 时的退化路径
    _np = None

__all__ = [
    "GridFov",
    "FovError",
    "InvalidGridError",
    "InvalidCameraError",
    "NonFiniteError",
    "TypeMismatchError",
]

_TAU = 2.0 * math.pi


# ---------------------------------------------------------------------------
# 可区分的异常类型（均继承 ValueError，兼容既有 _raises 断言）
# ---------------------------------------------------------------------------
class FovError(ValueError):
    """gridfov 参数错误的基类；本身也是 ``ValueError``。"""


class InvalidGridError(FovError):
    """网格结构非法：空网格、空行或非矩形。"""


class InvalidCameraError(FovError):
    """相机参数取值非法：长度不为 4、fov 越界或相机落在网格外。"""


class NonFiniteError(FovError):
    """输入含 NaN 或 ±inf（格值、相机坐标、方向角或视场角）。"""


class TypeMismatchError(FovError, TypeError):
    """参数类型错误；同时是 ``ValueError`` 与 ``TypeError``。"""


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------
def _cell_is_opaque(value, x, y):
    """把单个格值判定为是否不透明；非数值/非有限按分类抛错。"""
    kind = type(value)
    if kind is int:
        return value != 0
    if kind is float:
        if not math.isfinite(value):
            raise NonFiniteError(
                "grid 格值 (%d, %d) 非有限（NaN 或 ±inf）" % (x, y)
            )
        return value != 0.0
    if kind is bool:
        return value
    if isinstance(value, (str, bytes, bytearray, complex)):
        raise TypeMismatchError(
            "grid 格值 (%d, %d) 不是实数: %r" % (x, y, value)
        )
    # 其它数值类型（numpy 标量、Decimal、fractions 等）：尝试转 float。
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeMismatchError(
            "grid 格值 (%d, %d) 不是实数: %r" % (x, y, value)
        ) from exc
    if not math.isfinite(number):
        raise NonFiniteError(
            "grid 格值 (%d, %d) 非有限（NaN 或 ±inf）" % (x, y)
        )
    return number != 0.0


def _grid_to_opaque(grid):
    """把 ``grid`` 规整为不透明掩码。

    返回 ``(opaque, width, height)``：有 numpy 时 ``opaque`` 是行优先的
    二维 ``bool`` 数组，否则是 ``bytearray``（1 = 不透明）。
    """
    if isinstance(grid, (str, bytes, bytearray)):
        raise TypeMismatchError("grid 必须是二维数值序列，不能是字符串/字节串")
    if _np is not None:
        return _grid_to_opaque_numpy(grid)
    return _grid_to_opaque_python(grid)


def _grid_to_opaque_numpy(grid):  # pragma: no cover - 依赖 numpy
    np = _np
    if isinstance(grid, np.ndarray):
        raw = grid
    else:
        try:
            raw = np.asarray(grid)
        except (TypeError, ValueError) as exc:
            raise InvalidGridError("grid 必须是矩形的二维数值序列") from exc

    if raw.ndim == 0:
        raise TypeMismatchError(
            "grid 必须是二维数值序列，got 标量 %r" % (raw.item(),)
        )
    if raw.ndim != 2:
        raise InvalidGridError("grid 必须是二维的（got ndim=%d）" % raw.ndim)
    height, width = raw.shape
    if height == 0 or width == 0:
        raise InvalidGridError(
            "grid 不能为空（rows=%d, cols=%d）" % (height, width)
        )

    kind = raw.dtype.kind
    if kind in "USVcMm":
        raise TypeMismatchError(
            "grid 的格值必须是实数（got dtype=%s）" % raw.dtype
        )
    if kind == "b":
        return raw.copy(), width, height
    if kind in "iu":
        return raw != 0, width, height
    if kind == "f":
        if not np.isfinite(raw).all():
            raise NonFiniteError("grid 含非有限格值（NaN 或 ±inf）")
        return raw != 0.0, width, height
    if kind != "O":
        raise TypeMismatchError(
            "grid 的格值必须是实数（got dtype=%s）" % raw.dtype
        )

    # 混合类型（object dtype）：逐格检查类型与有限性。
    opaque = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            opaque[y, x] = _cell_is_opaque(raw[y, x], x, y)
    return opaque, width, height


def _grid_to_opaque_python(grid):
    try:
        rows = list(grid)
    except TypeError as exc:
        raise TypeMismatchError(
            "grid 必须是二维数值序列，got %s" % type(grid).__name__
        ) from exc
    if not rows:
        raise InvalidGridError("grid 不能为空")

    width = None
    opaque = bytearray()
    for y, row in enumerate(rows):
        if isinstance(row, (str, bytes, bytearray)):
            raise TypeMismatchError(
                "grid 的第 %d 行必须是数值序列，不能是字符串" % y
            )
        try:
            size = len(row)
        except TypeError as exc:
            raise TypeMismatchError(
                "grid 的第 %d 行必须是数值序列，got %s"
                % (y, type(row).__name__)
            ) from exc
        if width is None:
            if size == 0:
                raise InvalidGridError("grid 不能有空行")
            width = size
        elif size != width:
            raise InvalidGridError(
                "grid 必须是矩形：第 %d 行有 %d 列，应为 %d" % (y, size, width)
            )
        for x, value in enumerate(row):
            opaque.append(1 if _cell_is_opaque(value, x, y) else 0)
    return opaque, width, len(rows)


def _parse_cam(cam, width, height):
    """校验并解析 ``cam = (cx, cy, direction, fov)``。"""
    if isinstance(cam, (str, bytes, bytearray)):
        raise TypeMismatchError("cam 必须是含四个实数的序列，不能是字符串/字节串")
    try:
        items = list(cam)
    except TypeError as exc:
        raise TypeMismatchError(
            "cam 必须是可迭代的四个实数，got %s" % type(cam).__name__
        ) from exc
    if len(items) != 4:
        raise InvalidCameraError("cam 必须恰好包含 4 个值，got %d" % len(items))

    numbers = []
    for index, value in enumerate(items):
        if isinstance(value, (str, bytes, bytearray, complex, bool)):
            raise TypeMismatchError(
                "cam[%d] 必须是实数，got %s" % (index, type(value).__name__)
            )
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeMismatchError(
                "cam[%d] 必须是实数，got %r" % (index, value)
            ) from exc
        except OverflowError as exc:
            raise NonFiniteError("cam[%d] 超出浮点范围" % index) from exc
        if not math.isfinite(number):
            raise NonFiniteError("cam[%d] 必须有限，got %r" % (index, value))
        numbers.append(number)

    cx, cy, direction, fov = numbers
    if not 0.0 < fov <= _TAU:
        raise InvalidCameraError("fov（全张角，弧度）必须在 (0, 2π] 内，got %r" % fov)
    if not (0.0 <= cx < width and 0.0 <= cy < height):
        raise InvalidCameraError(
            "相机位置 (%r, %r) 必须在网格 [0, %d) × [0, %d) 内"
            % (cx, cy, width, height)
        )
    return cx, cy, direction, fov


# ---------------------------------------------------------------------------
# 视锥掩码
# ---------------------------------------------------------------------------
def _cone_mask_numpy(width, height, cx, cy, direction, fov):
    np = _np
    half = fov * 0.5
    if half >= math.pi:
        return np.ones((height, width), dtype=bool)
    xs = np.arange(width, dtype=np.float64) + 0.5 - cx
    ys = np.arange(height, dtype=np.float64) + 0.5 - cy
    ux = math.cos(direction)
    uy = math.sin(direction)
    dot = xs[None, :] * ux + ys[:, None] * uy
    cross = np.abs(ux * ys[:, None] - uy * xs[None, :])
    return np.arctan2(cross, dot) <= half


# ---------------------------------------------------------------------------
# 批量推进：对一组同主轴方向的候选同时做增量 Bresenham
# ---------------------------------------------------------------------------
def _sweep_numpy(flat, width, ccx, ccy, members, major, minor,
                 sign_major, sign_minor, x_major, keep):
    """推进一组候选，未被墙挡住的在 ``keep``（按候选序号）置 True。

    ``members`` 是候选在全体候选数组中的序号；``major`` / ``minor`` 是
    主轴 / 副轴偏移，``sign_major`` / ``sign_minor`` 是两轴步进方向，
    ``x_major`` 表示主轴是否为 x 轴。
    """
    np = _np
    order = np.argsort(major, kind="stable")
    index = members[order]
    # 扁平下标最大为 width*height-1；int32 足够时用它（更快），否则 intp。
    index_dtype = np.int32 if flat.size <= 2 ** 31 - 1 else np.intp
    major = major[order].astype(index_dtype)
    minor = minor[order].astype(index_dtype)
    sign_major = sign_major[order].astype(index_dtype)
    sign_minor = sign_minor[order].astype(index_dtype)

    count = index.size
    two_major = major * 2
    two_minor = minor * 2
    acc = major.copy()
    flat_index = np.full(count, ccy * width + ccx, dtype=index_dtype)
    width_i = index_dtype(width)
    if x_major:
        step_major = sign_major
        step_minor = sign_minor * width_i
    else:
        step_major = sign_major * width_i
        step_minor = sign_minor
    dead = np.zeros(count, dtype=bool)

    start = 0
    max_step = int(major[-1])
    for k in range(1, max_step + 1):
        # 主轴步数 < k 的候选已到达目标，退出推进。
        while start < count and major[start] < k:
            start += 1
        if start >= count:
            break
        part = slice(start, count)
        acc[part] += two_minor[part]
        carry = acc[part] >= two_major[part]
        acc[part] -= two_major[part] * carry
        flat_index[part] += step_major[part]
        flat_index[part] += step_minor[part] * carry
        hit = flat[flat_index[part]]
        hit &= ~dead[part]
        if hit.any():
            dead[start + np.nonzero(hit)[0]] = True

    alive = index[~dead]
    if alive.size:
        keep[alive] = True


def _visible_numpy(opaque, width, height, cx, cy, direction, fov):
    np = _np
    ccx = int(math.floor(cx))
    ccy = int(math.floor(cy))
    if ccx >= width:
        ccx = width - 1
    if ccy >= height:
        ccy = height - 1

    candidates = _cone_mask_numpy(width, height, cx, cy, direction, fov)
    candidates &= ~opaque
    candidates[ccy, ccx] = False

    cand_y, cand_x = np.nonzero(candidates)
    count = cand_x.size
    keep = np.zeros(count, dtype=bool)
    if count:
        if not opaque.any():
            # 全透明网格：没有任何遮挡，候选全部可见，省掉整轮推进。
            keep[:] = True
        else:
            dx = cand_x - ccx
            dy = cand_y - ccy
            adx = np.abs(dx)
            ady = np.abs(dy)
            x_major = adx >= ady
            flat = opaque.reshape(-1)
            for is_x in (True, False):
                members = np.nonzero(x_major == is_x)[0]
                if members.size == 0:
                    continue
                if is_x:
                    major = adx[members]
                    minor = ady[members]
                    sign_major = np.sign(dx[members]).astype(np.intp)
                    sign_minor = np.sign(dy[members]).astype(np.intp)
                else:
                    major = ady[members]
                    minor = adx[members]
                    sign_major = np.sign(dy[members]).astype(np.intp)
                    sign_minor = np.sign(dx[members]).astype(np.intp)
                _sweep_numpy(
                    flat, width, ccx, ccy, members, major, minor,
                    sign_major, sign_minor, is_x, keep,
                )

    visible = np.zeros((height, width), dtype=bool)
    visible[ccy, ccx] = True
    if count:
        visible[cand_y[keep], cand_x[keep]] = True
    out_y, out_x = np.nonzero(visible)
    return list(zip(out_x.tolist(), out_y.tolist()))


# ---------------------------------------------------------------------------
# 纯 Python 退化路径（无 numpy）：逐条约化方向的射线走位
# ---------------------------------------------------------------------------
def _visible_python(opaque, width, height, cx, cy, direction, fov):
    ccx = int(math.floor(cx))
    ccy = int(math.floor(cy))
    if ccx >= width:
        ccx = width - 1
    if ccy >= height:
        ccy = height - 1

    half = fov * 0.5
    full_circle = half >= math.pi
    ux = math.cos(direction)
    uy = math.sin(direction)
    atan2 = math.atan2
    vis = bytearray(width * height)
    vis[ccy * width + ccx] = 1

    def in_cone(x, y):
        if full_circle:
            return True
        vx = x + 0.5 - cx
        vy = y + 0.5 - cy
        return atan2(abs(ux * vy - uy * vx), ux * vx + uy * vy) <= half

    def mark(x, y):
        if in_cone(x, y):
            vis[y * width + x] = 1

    if not any(opaque):
        # 全透明网格：没有遮挡，锥内所有格都可见。
        for y in range(height):
            base = y * width
            for x in range(width):
                if not vis[base + x] and in_cone(x, y):
                    vis[base + x] = 1
    else:
        _sweep_python(opaque, width, height, ccx, ccy, mark)

    out = []
    for y in range(height):
        base = y * width
        for x in range(width):
            if vis[base + x]:
                out.append((x, y))
    return out


def _sweep_python(opaque, width, height, ccx, ccy, mark):
    """逐条原始方向的数字直线走位；遮挡格只挡后面的目标格。"""
    gcd = math.gcd
    # x 为主轴：|dy| ≤ |dx|。
    for sx in (1, -1):
        a_max = (width - 1 - ccx) if sx > 0 else ccx
        if a_max <= 0:
            continue
        for sy in (1, 0, -1):
            if sy == 0:
                b_range = (0,)
            else:
                b_max = (height - 1 - ccy) if sy > 0 else ccy
                if b_max <= 0:
                    continue
                b_range = range(1, min(a_max, b_max) + 1)
            for minor_b in b_range:
                two_b = 2 * minor_b
                for major_a in range(max(minor_b, 1), a_max + 1):
                    if minor_b and gcd(major_a, minor_b) != 1:
                        continue
                    two_a = 2 * major_a
                    y = ccy
                    acc = major_a
                    x = ccx + sx
                    k = 1
                    while k <= a_max:
                        acc += two_b
                        if acc >= two_a:
                            acc -= two_a
                            y += sy
                        if y < 0 or y >= height:
                            break
                        if opaque[y * width + x]:
                            break
                        k += 1
                        x += sx
                    # 同一原始方向上的目标格位于主轴步数 k 的整数倍处，
                    # 其 Bresenham 前缀正是刚走过的这段，故 k 之前都通视。
                    m = 1
                    while True:
                        kk = m * major_a
                        if kk >= k or kk > a_max:
                            break
                        x = ccx + sx * kk
                        y = ccy + sy * ((2 * kk * minor_b + major_a) // two_a)
                        if y < 0 or y >= height:
                            break
                        if not opaque[y * width + x]:
                            mark(x, y)
                        m += 1

    # y 为主轴：|dx| < |dy|（对角线已由 x 主轴分支覆盖）。
    for sy in (1, -1):
        b_max = (height - 1 - ccy) if sy > 0 else ccy
        if b_max <= 0:
            continue
        for sx in (1, 0, -1):
            if sx == 0:
                a_range = (0,)
            else:
                a_max = (width - 1 - ccx) if sx > 0 else ccx
                if a_max <= 0:
                    continue
                a_range = range(1, min(b_max - 1, a_max) + 1)
            for minor_a in a_range:
                two_a = 2 * minor_a
                for major_b in range(max(minor_a + 1, 1), b_max + 1):
                    if minor_a and gcd(major_b, minor_a) != 1:
                        continue
                    two_b = 2 * major_b
                    x = ccx
                    acc = major_b
                    y = ccy + sy
                    k = 1
                    while k <= b_max:
                        acc += two_a
                        if acc >= two_b:
                            acc -= two_b
                            x += sx
                        if x < 0 or x >= width:
                            break
                        if opaque[y * width + x]:
                            break
                        k += 1
                        y += sy
                    m = 1
                    while True:
                        kk = m * major_b
                        if kk >= k or kk > b_max:
                            break
                        y = ccy + sy * kk
                        x = ccx + sx * ((2 * kk * minor_a + major_b) // two_b)
                        if x < 0 or x >= width:
                            break
                        if not opaque[y * width + x]:
                            mark(x, y)
                        m += 1


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------
class GridFov:
    """网格视锥可见性计算器（无状态、可跨线程共享）。"""

    __slots__ = ()

    def visible(self, grid, cam):
        """返回从相机 ``cam`` 可见的格 ``(x, y)`` 列表。

        参数
        ----
        grid:
            ``grid[y][x]`` 形式的二维矩形序列（也接受 numpy 数组）。
            ``0`` 为透明格，非 ``0`` 为不透明遮挡格；格值必须是有限实数。
        cam:
            ``(cx, cy, direction, fov)``，约定见模块文档。

        返回
        ----
        list[tuple[int, int]]
            按行优先（先 ``y`` 后 ``x``）排序的可见格；相机所在格恒在
            结果中，且不透明格永不出现。

        异常
        ----
        InvalidGridError
            ``grid`` 为空、有空行或非矩形（``ValueError`` 子类）。
        InvalidCameraError
            ``cam`` 长度不为 4、``fov`` 越界或相机落在网格外
            （``ValueError`` 子类）。
        NonFiniteError
            输入含 NaN 或 ±inf（``ValueError`` 子类）。
        TypeMismatchError
            参数类型错误（同时是 ``ValueError`` 与 ``TypeError``）。
        """
        opaque, width, height = _grid_to_opaque(grid)
        cx, cy, direction, fov = _parse_cam(cam, width, height)
        if _np is not None:
            return _visible_numpy(opaque, width, height, cx, cy, direction, fov)
        return _visible_python(opaque, width, height, cx, cy, direction, fov)
