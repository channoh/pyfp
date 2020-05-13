#!/usr/bin/python3

from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='custom_fp',
      ext_modules=[cpp_extension.CppExtension('custom_fp',  [
          'custom_fp.cpp'
      ])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
