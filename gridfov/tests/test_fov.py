"""gridfov 题面约束的验收用例（``-k fov`` 选中本文件）。

覆盖：可见集合与「逐格投影 + Bresenham」独立参照逐格一致、相机在格内
不漏、遮挡体不透明、512×512 性能上界、非法参数分类报错、非有限输入单独
分类、并发安全、结果确定且为独立新列表。

参照片段刻意不复用被测实现：投影用独立的 ``atan2`` 环绕判定，线段用
独立的经典整数误差式 Bresenham，二者都与 ``core.py`` 的闭式捷径不同。
"""

import math
import random
import threading
import time

import pytest

import gridfov
from gridfov import (
    GridFov,
    InvalidCameraError,
    InvalidGridError,
    NonFiniteError,
)

G = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
CAM = (0.0, 0.0, 2.0, 2.0)


# ---------------------------------------------------------------------------
# 独立参照实现（逐格投影 + 经典 Bresenham）
# ---------------------------------------------------------------------------
def reference_bresenham(x0, y0, x1, y1):
    """经典整数误差式 Bresenham，返回含两端的完整格序列。"""
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    cells = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return cells
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _angle_inside(vx, vy, direction, half):
    """夹角 |wrap(atan2(v) - direction)| ≤ half，闭区间。"""
    delta = math.atan2(vy, vx) - direction
    delta = math.atan2(math.sin(delta), math.cos(delta))
    return abs(delta) <= half


def reference_visible(grid, cam):
    """逐格投影 + Bresenham 暴力参照，返回行优先列表。"""
    height = len(grid)
    width = len(grid[0])
    cx, cy, direction, fov = cam
    ccx = min(int(math.floor(cx)), width - 1)
    ccy = min(int(math.floor(cy)), height - 1)
    half = fov * 0.5
    out = []
    for y in range(height):
        for x in range(width):
            if x == ccx and y == ccy:
                out.append((x, y))
                continue
            if grid[y][x] != 0:
                continue
            if not _angle_inside(x + 0.5 - cx, y + 0.5 - cy, direction, half):
                continue
            path = reference_bresenham(ccx, ccy, x, y)
            if all(grid[gy][gx] == 0 for gx, gy in path[1:-1]):
                out.append((x, y))
    return out


# ---------------------------------------------------------------------------
# 可见集合与参照一致
# ---------------------------------------------------------------------------
def test_fov_nonempty():
    assert len(GridFov().visible(G, CAM)) > 0


def test_fov_wall_hidden():
    assert (1, 1) not in GridFov().visible(G, CAM)


def test_fov_inside_cell_not_missed():
    grid = [[0, 0], [0, 0]]
    assert len(GridFov().visible(grid, (0.5, 0.5, 0.5, 0.5))) >= 1


def test_fov_bounds():
    assert all(0 <= x < 3 and 0 <= y < 3 for x, y in GridFov().visible(G, CAM))


@pytest.mark.parametrize("seed", range(32))
def test_fov_matches_bresenham_reference(seed):
    rng = random.Random(seed)
    width = rng.randint(1, 18)
    height = rng.randint(1, 18)
    density = rng.choice([0.0, 0.05, 0.12, 0.25, 0.4])
    grid = [
        [1 if rng.random() < density else 0 for _ in range(width)]
        for _ in range(height)
    ]
    cx = rng.uniform(0.0, width - 1e-9) if width > 1 else 0.0
    cy = rng.uniform(0.0, height - 1e-9) if height > 1 else 0.0
    direction = rng.uniform(-2.0 * math.pi, 2.0 * math.pi)
    fov = rng.choice([rng.uniform(0.1, 2.0 * math.pi), 2.0 * math.pi])
    cam = (cx, cy, direction, fov)

    assert GridFov().visible(grid, cam) == reference_visible(grid, cam)


def test_fov_matches_reference_with_wall_grid():
    grid = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
    for cam in [
        (0.0, 0.0, 0.0, 2.0 * math.pi),
        (0.5, 0.5, 0.7, 1.2),
        (3.9, 3.1, -2.4, 3.0),
        (1.5, 0.2, math.pi / 2.0, math.pi),
    ]:
        assert GridFov().visible(grid, cam) == reference_visible(grid, cam)


# ---------------------------------------------------------------------------
# 相机在格内不漏
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("offset", [(0.0, 0.0), (0.25, 0.75), (0.5, 0.5), (0.999, 0.001)])
def test_fov_camera_inside_cell_not_missed_offsets(offset):
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    ox, oy = offset
    cam = (1.0 + ox, 1.0 + oy, 0.0, 0.2)
    got = GridFov().visible(grid, cam)
    assert (1, 1) in got
    assert (1, 1) in reference_visible(grid, cam)


def test_fov_camera_in_walled_cell_still_sees_own_cell():
    # 相机恰好落在墙格里：该格仍可见，但墙后不可见。
    cam = (1.4, 1.6, 0.0, 2.0 * math.pi)
    got = GridFov().visible(G, cam)
    assert (1, 1) in got
    assert got == reference_visible(G, cam)


def test_fov_camera_on_grid_corner():
    grid = [[0, 0], [0, 0]]
    cam = (0.0, 0.0, math.pi / 4.0, math.pi)
    got = GridFov().visible(grid, cam)
    assert (0, 0) in got
    assert got == reference_visible(grid, cam)


# ---------------------------------------------------------------------------
# 遮挡体不透明
# ---------------------------------------------------------------------------
def test_fov_occluders_are_opaque():
    got = GridFov().visible(G, (0.0, 0.0, 2.0, 2.0))
    assert (1, 1) not in got


def test_fov_wall_shadows_cells_behind_it():
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    got = GridFov().visible(grid, (0.5, 0.5, 0.0, math.pi))
    assert (2, 2) not in got  # 墙本身
    assert (3, 2) not in got  # 墙正后方
    assert got == reference_visible(grid, (0.5, 0.5, 0.0, math.pi))


def test_fov_fully_walled_room_only_own_cell():
    grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    assert GridFov().visible(grid, (1.5, 1.5, 0.0, 2.0 * math.pi)) == [(1, 1)]


def test_fov_narrow_cone_excludes_behind():
    grid = [[0] * 9 for _ in range(9)]
    # 视角朝 +x、张角很小：相机左边的格必须被锥体裁掉。
    got = set(GridFov().visible(grid, (4.5, 4.5, 0.0, 0.2)))
    assert (0, 4) not in got
    assert (8, 4) in got


# ---------------------------------------------------------------------------
# 性能上界
# ---------------------------------------------------------------------------
def _perf_grid(size, density, seed):
    rng = random.Random(seed)
    return [
        [1 if rng.random() < density else 0 for _ in range(size)]
        for _ in range(size)
    ]


@pytest.mark.parametrize(
    "density,cam",
    [
        (0.0, (5.5, 7.5, 0.4, 2.0 * math.pi)),
        (0.02, (5.5, 7.5, 0.4, 2.0 * math.pi)),
        (0.02, (256.0, 256.0, 0.4, 1.0)),
        (0.5, (256.0, 256.0, 0.4, 2.0 * math.pi)),
    ],
)
def test_fov_performance_512_under_500ms(density, cam):
    size = 512
    grid = _perf_grid(size, density, seed=2024)
    culler = GridFov()
    start = time.perf_counter()
    got = culler.visible(grid, cam)
    elapsed = time.perf_counter() - start
    assert len(got) >= 1
    assert elapsed < 0.5, "512x512 density=%s took %.0f ms" % (density, elapsed * 1000)


def test_fov_performance_512_open_grid():
    size = 512
    grid = [[0] * size for _ in range(size)]
    start = time.perf_counter()
    got = GridFov().visible(grid, (0.5, 0.5, 0.0, 2.0 * math.pi))
    elapsed = time.perf_counter() - start
    assert len(got) == size * size
    assert elapsed < 0.5, "512x512 open grid took %.0f ms" % (elapsed * 1000)


# ---------------------------------------------------------------------------
# 非法参数：分类报错
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cam",
    [
        (0.0, 0.0, 0.0, 0.0),  # fov = 0
        (0.0, 0.0, 0.0, -1.0),  # fov < 0
        (0.0, 0.0, 0.0, 2.0 * math.pi + 0.1),  # fov > 2π
        (0.0, 0.0, 0.0),  # 长度不足
        (0.0, 0.0, 0.0, 1.0, 1.0),  # 长度过多
        (9.0, 9.0, 0.0, 1.0),  # 相机落在网格外
        (-0.1, 0.0, 0.0, 1.0),
        (0.0, 2.0, 0.0, 1.0),
    ],
)
def test_fov_bad_camera_raises_value_error(cam):
    with pytest.raises(ValueError):
        GridFov().visible([[0, 0], [0, 0]], cam)


@pytest.mark.parametrize(
    "grid",
    [
        [],
        [[]],
        [[0, 0], [0]],  # 非矩形
        [[0, 0], [0, 0], [0]],  # 非矩形（尾行较短）
    ],
)
def test_fov_bad_grid_raises_value_error(grid):
    with pytest.raises(ValueError):
        GridFov().visible(grid, (0.0, 0.0, 0.0, 1.0))


@pytest.mark.parametrize(
    "grid",
    [
        [[0, 0], [0, float("nan")]],
        [[0, float("inf")], [0, 0]],
        [[float("-inf"), 0], [0, 0]],
    ],
)
def test_fov_nonfinite_grid_raises(grid):
    with pytest.raises(ValueError):
        GridFov().visible(grid, (0.0, 0.0, 0.0, 1.0))


@pytest.mark.parametrize(
    "cam",
    [
        (float("nan"), 0.0, 0.0, 1.0),
        (0.0, float("inf"), 0.0, 1.0),
        (0.0, 0.0, float("-inf"), 1.0),
        (0.0, 0.0, 0.0, float("nan")),
        (0.0, 0.0, 0.0, float("inf")),
    ],
)
def test_fov_nonfinite_camera_raises(cam):
    with pytest.raises(ValueError):
        GridFov().visible([[0, 0], [0, 0]], cam)


def test_fov_error_types_are_distinguishable():
    grid = [[0, 0], [0, 0]]
    with pytest.raises(InvalidGridError):
        GridFov().visible([], (0.0, 0.0, 0.0, 1.0))
    with pytest.raises(InvalidCameraError):
        GridFov().visible(grid, (0.0, 0.0, 0.0, 0.0))
    with pytest.raises(NonFiniteError):
        GridFov().visible([[0, float("nan")]], (0.0, 0.0, 0.0, 1.0))
    for exc in (InvalidGridError, InvalidCameraError, NonFiniteError):
        assert issubclass(exc, ValueError)


@pytest.mark.parametrize(
    "cam",
    [None, 3.0, "abcd", (0.0, 0.0, 0.0, "x"), (0.0, 0.0, None, 1.0)],
)
def test_fov_bad_camera_type_raises_type_error(cam):
    with pytest.raises(TypeError):
        GridFov().visible([[0, 0], [0, 0]], cam)


@pytest.mark.parametrize(
    "grid",
    [None, 5, [[0, "x"], [0, 0]], [[0, 1j], [0, 0]], "abcd"],
)
def test_fov_bad_grid_type_raises_type_error(grid):
    with pytest.raises(TypeError):
        GridFov().visible(grid, (0.0, 0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 结果形状：类型、确定性、独立性
# ---------------------------------------------------------------------------
def test_fov_result_is_list_of_int_tuples():
    got = GridFov().visible([[0, 0], [0, 0]], (0.5, 0.5, 0.0, math.pi))
    assert isinstance(got, list)
    assert all(isinstance(cell, tuple) and len(cell) == 2 for cell in got)
    assert all(isinstance(x, int) and isinstance(y, int) for x, y in got)


def test_fov_deterministic():
    grid = [[0, 1, 0], [0, 0, 0], [1, 0, 0]]
    cam = (0.3, 0.7, 1.1, 2.4)
    culler = GridFov()
    assert culler.visible(grid, cam) == culler.visible(grid, cam)


def test_fov_returned_list_is_independent():
    culler = GridFov()
    grid = [[0, 0], [0, 0]]
    cam = (0.5, 0.5, 0.0, 2.0 * math.pi)
    first = culler.visible(grid, cam)
    first.clear()
    assert len(culler.visible(grid, cam)) == 4


def test_fov_accepts_numpy_grid():
    np = pytest.importorskip("numpy")
    cam = (0.0, 0.0, 2.0, 2.0)
    assert GridFov().visible(np.asarray(G), cam) == GridFov().visible(G, cam)


# ---------------------------------------------------------------------------
# 并发安全
# ---------------------------------------------------------------------------
def test_fov_concurrent_calls_consistent():
    culler = GridFov()
    cases = [
        (G, CAM),
        ([[0, 0], [0, 0]], (0.5, 0.5, 0.5, 0.5)),
        ([[0] * 8 for _ in range(8)], (3.5, 3.5, 0.7, math.pi)),
        ([[0, 1, 0], [0, 0, 0], [1, 0, 0]], (0.3, 0.7, 1.1, 2.4)),
    ]
    expected = [culler.visible(grid, cam) for grid, cam in cases]
    errors = []

    def worker():
        try:
            for _ in range(200):
                for (grid, cam), want in zip(cases, expected):
                    got = culler.visible(grid, cam)
                    if got != want:
                        errors.append((grid, cam, got, want))
                        return
                    got.append((0, 0))  # 修改返回值不得影响后续调用
        except Exception as exc:  # pragma: no cover - 失败路径
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


def test_fov_concurrent_matches_serial():
    size = 128
    grid = [[0] * size for _ in range(size)]
    cam = (1.5, 1.5, 0.0, 2.0 * math.pi)
    culler = GridFov()
    serial = culler.visible(grid, cam)
    results = []
    lock = threading.Lock()

    def worker():
        got = culler.visible(grid, cam)
        with lock:
            results.append(got)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 8
    assert all(result == serial for result in results)
