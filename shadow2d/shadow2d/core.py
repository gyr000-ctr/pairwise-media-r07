"""shadow2d 核心 API（初始未实现）。"""


class SoftShadow:
    def __init__(self, samples, seed=0):
        """构造"""
        raise NotImplementedError

    def sample(self, occ, x, y):
        """遮蔽值"""
        raise NotImplementedError
