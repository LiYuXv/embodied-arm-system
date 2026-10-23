"""cloud_inference 包初始化：把本目录加入 sys.path，使 protoc 生成的
平级 import（import vla_inference_pb2）在包内可用。本地端与云端通用。"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
