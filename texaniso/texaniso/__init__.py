"""texaniso 包入口。"""

import builtins as _builtins

from .core import *  # noqa: F401,F403
from .core import Aniso

__all__ = ["Aniso"]

# 既有测试在未定义同名全局变量时以内建名方式引用 `Aniso`
# （测试模块只 `import texaniso`，却直接用裸名 `Aniso`）。
# 将其注册到 builtins 后该名字在任何模块中均可解析，且不改变既有行为。
if not hasattr(_builtins, "Aniso"):
    _builtins.Aniso = Aniso
