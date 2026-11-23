# shadow2d

## 题面（可直接粘贴）

2D 软阴影（PCF）。约束：采样数可配且随采样数收敛；全遮挡处为 0、全亮处为 1（≤1e-6）；同种子可复现；边界像素正确；1 万像素不超过 400 ms；非法参数报错。验收：cd shadow2d && python -m pytest tests -q -k shadow。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd shadow2d && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
