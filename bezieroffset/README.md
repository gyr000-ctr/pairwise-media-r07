# bezieroffset

## 题面（可直接粘贴）

贝塞尔偏移曲线。约束：偏移后点到原曲线距离与偏移量误差 ≤1e-3；首尾法线方向正确；退化曲线报错；1 万次偏移不超过 500 ms；并发安全。验收：cd bezieroffset && python -m pytest tests -q -k offset。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd bezieroffset && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
