from setuptools import setup, Extension
import pybind11
from setuptools.command.build_ext import build_ext
import sys
import os


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        python_executable = sys.executable
        python_root = os.path.dirname(os.path.dirname(python_executable))
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../..")
        )
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={project_root}",
            f"-DPython_ROOT={python_root}",
            f"-DPYTHON_EXECUTABLE={python_executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        build_args = ["--config", "Release"]

        self.build_temp = os.path.join(self.build_temp, "build")
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Change to build directory before running cmake
        os.chdir(self.build_temp)
        self.spawn(["cmake", ext.sourcedir] + cmake_args)
        self.spawn(["cmake", "--build", "."] + build_args)


setup(
    name="vamanalib",
    version="0.1",
    author="Liziyi",
    description="Python bindings for Vamana graph-based index",
    ext_modules=[CMakeExtension("vamanallib")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
)
