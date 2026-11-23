# limiter

## 题面（可直接粘贴）

前视限幅器。约束：输出真峰值 ≤ 阈值（4 倍过采样近似）；前视窗内预衰减且无过冲；release 单调恢复；跨缓冲连续；100 万样本不超过 400 ms；非法参数可区分报错。验收：cd limiter && python -m pytest tests -q -k limit。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd limiter && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
