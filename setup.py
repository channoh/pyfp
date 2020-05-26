#!/usr/bin/python3

from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="custom_fp",
      ext_modules=[cpp_extension.CppExtension(
          name="custom_fp", 
          sources=["src/custom_fp.cc", "src/fp32.cc"],
          extra_compile_args=['-g', '-fopenmp']
          )],
      cmdclass={"build_ext": cpp_extension.BuildExtension})
