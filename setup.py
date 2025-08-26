from Cython.Build import cythonize
from setuptools import Extension, setup

exts = [
    Extension("solitier_game_lw_fast", ["solitier_game_lw_fast.pyx"]),
]

setup(ext_modules=cythonize(exts, language_level=3, annotate=True))
