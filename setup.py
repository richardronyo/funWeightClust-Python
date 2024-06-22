# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "py_mixture",
        sources=["mixture.pyx", "src/mixture_wrapper.c"],
        include_dirs=[np.get_include()],
        libraries=["src/funclustweight"],
        library_dirs=["src"]
    ),
    Extension(
        "imahalanobis",
        sources=["imahalanobis.pyx", "src/imahalanobis_wrapper.c"],
        include_dirs=[np.get_include()],
        libraries=["src/TFunHDDC"],
        library_dirs=["src"]
    )
]

setup(
    name="py_mixture",
    ext_modules=cythonize(extensions),
)
