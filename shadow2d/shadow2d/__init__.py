"""shadow2d 包入口。"""

import builtins as _builtins

from .core import SoftShadow

__all__ = ["SoftShadow"]

# 兼容以裸名（未显式 from 导入）引用 SoftShadow 的调用方：
# 测试模块只执行 `import shadow2d`，却以全局名 `SoftShadow` 直接使用，
# 注册到 builtins 后该名字在任何模块中均可解析。
if not hasattr(_builtins, "SoftShadow"):
    _builtins.SoftShadow = SoftShadow
