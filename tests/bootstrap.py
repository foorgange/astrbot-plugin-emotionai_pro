# tests/bootstrap.py
"""把项目根目录注册为合法 Python 包名（目录名带连字符，无法直接 import）。

用法：在测试文件顶部 `import tests.bootstrap`，然后用
`from emotionai_pro.xxx import ...` 导入插件模块。
"""
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

if "emotionai_pro" not in sys.modules:
    pkg = types.ModuleType("emotionai_pro")
    pkg.__path__ = [str(_ROOT)]
    pkg.__package__ = "emotionai_pro"
    pkg.__name__ = "emotionai_pro"
    sys.modules["emotionai_pro"] = pkg

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
