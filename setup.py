from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name="non_neg_ls",
    ext_modules=cythonize(
        Extension(
            "non_neg_ls",
            sources=["src/non_neg_ls/*.pyx"],
            include_dirs=[numpy.get_include()]
        )
    ),
    zip_safe=False
)