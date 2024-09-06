# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "py_mixture",
        sources=["mixture.pyx", "src/mixture_wrapper.c"],
        include_dirs=[np.get_include()],
        libraries=["funclustweight"],
        library_dirs=["src"]
    ),
    Extension(
        "imahalanobis",
        sources=["imahalanobis.pyx", "src/imahalanobis.c", "src/TFunHDDC.c"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="py_mixture",
    ext_modules=cythonize(extensions),
)
