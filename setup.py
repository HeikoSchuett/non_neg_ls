import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

USE_CYTHON = os.getenv('USE_CYTHON')

if USE_CYTHON:
    print('Using cython now!')
    extensions = cythonize([
        Extension(
            "non_neg_ls.solve_chol",
            sources=["src/non_neg_ls/solve_chol.pyx"],
            include_dirs=[numpy.get_include()]
        ),
        Extension(
            "non_neg_ls.solve_qr",
            sources=["src/non_neg_ls/solve_qr.pyx"],
            include_dirs=[numpy.get_include()]
        )
    ])
else:
    print('NOT using cython now, using existing C code!')
    extensions = [
        Extension(
            "non_neg_ls.solve_chol",
            sources=["src/non_neg_ls/solve_chol.c"],
            include_dirs=[numpy.get_include()]
        ),
        Extension(
            "non_neg_ls.solve_qr",
            sources=["src/non_neg_ls/solve_qr.c"],
            include_dirs=[numpy.get_include()]
        )
    ]


setup(
    name="non_neg_ls",
    ext_modules=extensions,
    packages=['non_neg_ls'],
    package_dir = {'': 'src'},
    zip_safe=False
)
