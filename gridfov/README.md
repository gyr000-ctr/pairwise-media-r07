# gridfov

## 题面（可直接粘贴）

视锥 + 遮挡剔除。约束：可见集合与逐格投影+Bresenham 对照一致；相机在格内不漏；512x512 不超过 500 ms；非法参数报错；并发安全。验收：cd gridfov && python -m pytest tests -q -k fov。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd gridfov && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
