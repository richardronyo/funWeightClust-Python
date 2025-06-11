from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import subprocess

extensions = [
    Extension(
        "py_mixture",
        sources=["mixture.pyx"],
        include_dirs=[
            np.get_include()],
        libraries=["funclustweight"],  # Keep this name if you didn't rename
        library_dirs=["libs"],
        runtime_library_dirs=["@loader_path"],  # for macOS so it finds .dylib at runtime
 
    )
]

setup(
    name="py_mixture",
    version="0.1",
    ext_modules=cythonize(extensions, annotate=True),
)