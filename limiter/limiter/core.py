"""limiter 核心 API（初始未实现）。"""


class Limiter:
    def __init__(self, ceiling, lookahead, release):
        """构造"""
        raise NotImplementedError

    def process(self, samples):
        """处理"""
        raise NotImplementedError
