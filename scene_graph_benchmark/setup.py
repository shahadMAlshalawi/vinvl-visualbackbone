# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]
from setuptools.command.develop import develop

class CustomDevelopCommand(develop):
    def run(self):
        os.system("python setup.py build_ext --inplace")
        super().run()

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "maskrcnn_benchmark", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "maskrcnn_benchmark._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="maskrcnn_benchmark",
    version="0.1",
    author="Mahmood Anaam",
    url="https://github.com/Mahmood-Anaam/vinvl-visualbackbone.git",
    description="object detection in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),
    license="MIT",
    install_requires=[
        "ipython",
        "h5py",
        "nltk",
        "joblib",
        "jupyter",
        "pandas",
        "ninja",
        "timm",
        "einops",
        "cython",
        "matplotlib",
        "tqdm",
        "opencv-python",
        "yacs>=0.1.8",
        "pycocotools",
        "cityscapesScripts>=2.2.4",
        "clint>=0.5.1",
        "torch==1.13.1",
        "torchvision==0.14.1",
        "numpy==1.23.5",
    ],

    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": torch.utils.cpp_extension.BuildExtension,
        "develop": CustomDevelopCommand, 
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
