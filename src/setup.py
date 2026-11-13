# Native build entry point rewritten for Futurism.
# Replaces the Python packaging helper inherited through Kubism from KMCUDA.
# Pinned sources and file comparisons: docs/SOURCE_AUDIT.md.
# Source attribution and license scope: docs/PROVENANCE.md.

"""Build libfuturism.so with the existing `python3 setup.py build` command.

This is a native library build entry point, not a Python module distribution.
"""

import os
from pathlib import Path
import subprocess

from setuptools import Command, setup


class BuildNative(Command):
    description = "build the Futurism native shared library in src/"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        source_dir = Path(__file__).resolve().parent
        try:
            jobs = int(os.environ.get("BUILD_JOBS", "2"))
        except ValueError as exc:
            raise SystemExit("BUILD_JOBS must be a positive integer") from exc
        if jobs < 1:
            raise SystemExit("BUILD_JOBS must be a positive integer")

        configure = [
            "cmake", "-S", str(source_dir), "-B", str(source_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCUDA_ARCH=" + os.environ.get("CUDA_ARCH", "87"),
        ]
        if os.environ.get("CUDA_TOOLKIT_ROOT_DIR"):
            configure.append("-DCUDA_TOOLKIT_ROOT_DIR=" + os.environ["CUDA_TOOLKIT_ROOT_DIR"])
        subprocess.check_call(configure)
        subprocess.check_call([
            "cmake", "--build", str(source_dir), "--parallel", str(jobs),
        ])


setup(
    name="futurism-native-build",
    version="0.0.0",
    description="Local build entry point for the Futurism native library",
    packages=[],
    py_modules=[],
    cmdclass={"build": BuildNative},
)
