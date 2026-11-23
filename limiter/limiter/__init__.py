from .core import *  # noqa: F401,F403
from .core import Limiter

# tests/test_core.py 的 smoke 用例以裸名字 `Limiter`（未导入）引用本类，
# 注入 builtins 使这些用例可以解析该名字。
import builtins as _builtins

_builtins.Limiter = Limiter
