"""
setup_BR.py — Compilar BR_lens_core.pyx en extensión C.

Uso:
    cd /ruta/a/scripts/
    python setup_BR.py build_ext --inplace

Esto genera BR_lens_core.cpython-3X-*.so en el mismo directorio.
Sage usa Python 3.13; si hay problemas de versión, usa:
    /private/var/tmp/sage-10.7-current/local/bin/python3 setup_BR.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Directorio de este script
HERE = os.path.dirname(os.path.abspath(__file__))

ext = Extension(
    "BR_lens_core",
    sources=[os.path.join(HERE, "BR_lens_core.pyx")],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-march=native", "-ffast-math"],
)

setup(
    name="BR_lens_core",
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
    ),
)
