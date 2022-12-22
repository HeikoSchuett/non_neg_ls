from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="non_neg_ls",
    ext_modules=cythonize(
        "src/non_neg_ls/*.pyx",
        include_path=[numpy.get_include()]),
    zip_safe=False
)