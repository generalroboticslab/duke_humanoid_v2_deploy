"""hardware_bindings.ft_servo — loads the C++ `ft_servo_ext` binding and exports `FtServo`.

Importing this package requires the built `ft_servo_ext.abi3.so` beside this
file; when it is missing `_load_ext` raises FileNotFoundError — there is NO
pure-Python fallback here. `ft_servo_python_only.py` in this directory is a
separate, standalone driver with its own `FtServo` class; it is imported
explicitly by `change_id.py` only, and `control/ft_servo_python_only.py` is
its canonical copy (see ../README.md, "Twin modules").
"""
import importlib.util
import pathlib
from functools import lru_cache


@lru_cache(maxsize=None)
def _load_ext(name):
    path = pathlib.Path(__file__).parent / f"{name}.abi3.so"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run `cmake --build control/build` first")
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FtServo = _load_ext("ft_servo_ext").FtServo
