# lib/__init__.py

import sys
import types

# ---- TensorFlow dummy 주입 ----
# transformers.image_transforms 가 "import tensorflow as tf" 를 할 때
# 실제 TensorFlow 패키지를 로딩하지 않고, 이 더미 모듈을 사용하게 만든다.
if "tensorflow" not in sys.modules:
    dummy_tf = types.ModuleType("tensorflow")
    dummy_tf.__dict__["__version__"] = "0.0.0"

    class DummyTensor:
        pass

    dummy_tf.Tensor = DummyTensor

    # (예전에 tf.io.gfile.join 관련 에러를 봤다면, 아래처럼 io/gfile도 미리 만들어 둘 수 있음)
    dummy_tf.io = types.SimpleNamespace(
        gfile=types.SimpleNamespace(
            join=lambda *args, **kwargs: ""
        )
    )

    dummy_tf.__dict__["__version__"] = "0.0.0"
    # 필요하면 나중에 tf.image 같은 서브모듈도 여기에 추가로 붙일 수 있음.
    sys.modules["tensorflow"] = dummy_tf
# --------------------------------

# ---- torchao dummy 주입 (여기 부분을 새로/수정) ----
class DummyQuantization:
    """torchao.quantization.* 에 접근할 때마다
    요청된 이름으로 비어 있는 더미 클래스를 만들어 주는 객체.
    """
    def __getattr__(self, name):
        Dummy = type(name, (), {})  # 예: class Float8WeightOnlyConfig: pass
        setattr(self, name, Dummy)  # 다음부터는 캐시된 걸 사용
        return Dummy

if "torchao" not in sys.modules:
    dummy_torchao = types.ModuleType("torchao")
    dummy_torchao.__dict__["__version__"] = "0.0.0"

    # quantization, dtypes, ops 같은 네임스페이스도 미리 만들어 둔다
    dummy_torchao.quantization = DummyQuantization()
    dummy_torchao.dtypes = DummyQuantization()
    dummy_torchao.ops = DummyQuantization()

    sys.modules["torchao"] = dummy_torchao
# --------------------------------------------------------------

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