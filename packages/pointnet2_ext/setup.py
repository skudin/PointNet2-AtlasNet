import glob

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_src_root = "."
ext_sources = glob.glob("{}/src/*.cpp".format(ext_src_root)) + glob.glob("{}/src/*.cu".format(ext_src_root))

setup(
    name="pointnet2_ext",
    ext_modules=[
        CUDAExtension(
            name="pointnet2_ext",
            sources=ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(ext_src_root))],
            }
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    })
