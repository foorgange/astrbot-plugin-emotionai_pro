# tests/run_all.py
"""离线运行全部单测。

用法（在项目根目录）：
    python tests/run_all.py

测试通过 tests/astrbot_stub 注入 astrbot 相关模块桩，
因此无需安装 AstrBot 本体即可运行。
"""
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tests.astrbot_stub  # noqa: F401,E402  (必须先注入桩模块)


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.join(_ROOT, "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
