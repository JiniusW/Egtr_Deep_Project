# lib/__init__.py

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