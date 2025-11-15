# lib/__init__.py

import sys
import types

# ---- TensorFlow dummy 주입 ----
# transformers.image_transforms 가 "import tensorflow as tf" 를 할 때
# 실제 TensorFlow 패키지를 로딩하지 않고, 이 더미 모듈을 사용하게 만든다.
if "tensorflow" not in sys.modules:
    dummy_tf = types.ModuleType("tensorflow")
    dummy_tf.__dict__["__version__"] = "0.0.0"
    # 필요하면 나중에 tf.image 같은 서브모듈도 여기에 추가로 붙일 수 있음.
    sys.modules["tensorflow"] = dummy_tf

import numpy as np

# NumPy 2.x / 최신 버전 호환용 레거시 alias 복구
# (옛날 코드에서 np.float, np.int 등을 쓰는 경우 대비)

if not hasattr(np, "float"):
    np.float = float

if not hasattr(np, "int"):
    np.int = int

if not hasattr(np, "bool"):
    np.bool = bool

if not hasattr(np, "object"):
    np.object = object

if not hasattr(np, "long"):
    np.long = int