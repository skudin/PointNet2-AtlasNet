from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer_loss',
    ext_modules=[
        CUDAExtension('chamfer_loss', [
            'chamfer_cuda.cpp',
            'chamfer.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
