"""
Copyright (C) 2026 Laurent G. Courty

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1
of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
"""

import os
import platform
from pathlib import Path

from Cython.Build import cythonize
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension

BUILD_JOBS = 5


def parse_env_bool(name: str) -> bool | None:
    """Return a boolean override from an environment variable."""
    value = os.getenv(name)
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"Environment variable {name!r} must be one of 1/0, true/false, yes/no, on/off"
    )


class BuildConfig:
    """Centralized build configuration management for itzi package.

    This class handles:
    - Detection of build mode (source vs wheel)
    - Platform and architecture detection
    - Compiler flag selection based on build mode and target platform
    """

    def __init__(self):
        self.is_wheel_build = os.getenv("ITZI_BDIST_WHEEL") is not None
        self.platform = self.detect_platform()
        self.architecture = self.detect_architecture()
        self.openmp_override = parse_env_bool("ITZI_USE_OPENMP")
        self.use_openmp = self.detect_openmp_usage()
        self.compiler_type = None  # Set during build
        self.base_compile_args_plain = ["-O3", "-w"]
        self.base_compile_args_unix = ["-O3", "-w", "-fopenmp"]
        self.base_compile_args_macos = ["-O3", "-w", "-Xpreprocessor", "-fopenmp"]
        self.base_compile_args_msvc = ["/openmp", "/Ox"]
        self.base_compile_args_msvc_plain = ["/Ox"]
        self.base_link_args_unix = ["-lgomp"]

    def detect_openmp_usage(self) -> bool:
        """Return whether OpenMP should be enabled for this build."""
        if self.openmp_override is not None:
            return self.openmp_override

        return not (
            self.is_wheel_build and self.platform == "macos" and self.architecture == "arm64"
        )

    def get_macos_openmp_paths(self) -> tuple[list[str], list[str]]:
        """Select OpenMP paths, honoring an explicit runtime installation."""
        prefix = os.getenv("ITZI_OPENMP_PREFIX")
        if prefix is not None:
            root = Path(prefix).expanduser().resolve()
            for required in (root / "include" / "omp.h", root / "lib" / "libomp.dylib"):
                if not required.is_file():
                    raise FileNotFoundError(f"ITZI_OPENMP_PREFIX is missing {required}")
            return [str(root / "include")], [str(root / "lib")]

        return (
            [path for path in macos_includes if os.path.isdir(path)],
            [path for path in macos_libs if os.path.isdir(path)],
        )

    def detect_platform(self):
        """Detect current platform (Linux, Windows, macOS)"""
        system = platform.system()
        if system == "Linux":
            return "linux"
        elif system == "Windows":
            return "windows"
        elif system == "Darwin":
            return "macos"
        else:
            return "unknown"

    def detect_architecture(self):
        """Detect current architecture (x86_64, ARM64, etc.)"""
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            return "x86_64"
        elif machine in ("arm64", "aarch64"):
            return "arm64"
        elif machine.startswith("arm"):
            return "arm"
        else:
            return machine

    def get_optimization_flags(self, compiler_type):
        """Return appropriate compiler flags based on build mode and platform"""
        self.compiler_type = compiler_type

        if self.is_wheel_build:
            return self._get_wheel_optimization_flags()
        else:
            return self._get_source_optimization_flags()

    def _get_source_optimization_flags(self):
        """Get optimization flags for source builds (aggressive, machine-specific)"""
        compile_args = []
        link_args = []

        if self.compiler_type == "msvc":
            # Conservative MSVC flags for source builds
            if self.use_openmp:
                compile_args = self.base_compile_args_msvc.copy()
            else:
                compile_args = self.base_compile_args_msvc_plain.copy()
            link_args = []
        elif self.compiler_type == "mingw32":
            if self.use_openmp:
                compile_args = self.base_compile_args_unix + [
                    "-lgomp",
                    "-lpthread",
                    "-march=native",
                ]
                link_args = ["-lgomp", "-lpthread"]
            else:
                compile_args = self.base_compile_args_plain + ["-lpthread", "-march=native"]
                link_args = ["-lpthread"]
        elif self.compiler_type == "unix":
            if self.platform == "macos":
                # macOS specific handling
                if self.use_openmp:
                    compile_args = self.base_compile_args_macos + ["-march=native"]
                    link_args = ["-lomp"]
                else:
                    compile_args = self.base_compile_args_plain + ["-march=native"]
                    link_args = []
            else:
                # Linux and other Unix systems
                if self.use_openmp:
                    compile_args = self.base_compile_args_unix + ["-march=native"]
                    link_args = self.base_link_args_unix
                else:
                    compile_args = self.base_compile_args_plain + ["-march=native"]
                    link_args = []

        return compile_args, link_args

    def _get_wheel_optimization_flags(self):
        """Get optimization flags for wheel builds (conservative, architecture-specific)"""
        compile_args = []
        link_args = []

        if self.compiler_type == "msvc":
            if self.use_openmp:
                compile_args = self.base_compile_args_msvc.copy()
            else:
                compile_args = self.base_compile_args_msvc_plain.copy()
            if self.architecture == "x86_64":
                compile_args.append("/arch:AVX2")
            elif self.architecture == "arm64":
                compile_args.append("/arch:armv8.2")
            link_args = []

        elif self.compiler_type == "mingw32":
            if self.architecture == "x86_64":
                if self.use_openmp:
                    compile_args = self.base_compile_args_unix + [
                        "-lgomp",
                        "-lpthread",
                        "-march=x86-64-v3",
                    ]
                else:
                    compile_args = self.base_compile_args_plain + ["-lpthread", "-march=x86-64-v3"]
            elif self.architecture == "arm64":
                if self.use_openmp:
                    compile_args = self.base_compile_args_unix + ["-march=armv8-a+simd"]
                else:
                    compile_args = self.base_compile_args_plain + ["-march=armv8-a+simd"]
            else:
                if self.use_openmp:
                    compile_args = self.base_compile_args_unix + ["-lgomp", "-lpthread"]
                else:
                    compile_args = self.base_compile_args_plain + ["-lpthread"]
            if self.use_openmp:
                link_args = ["-lgomp", "-lpthread"]
            else:
                link_args = ["-lpthread"]

        elif self.compiler_type == "unix":
            if self.platform == "macos":
                if self.architecture == "arm64":
                    if self.use_openmp:
                        compile_args = self.base_compile_args_macos + ["-march=armv8-a+simd"]
                    else:
                        compile_args = self.base_compile_args_plain + ["-march=armv8-a+simd"]
                else:
                    if self.use_openmp:
                        compile_args = self.base_compile_args_macos
                    else:
                        compile_args = self.base_compile_args_plain.copy()
                link_args = ["-lomp"] if self.use_openmp else []
            else:
                # Linux and other Unix systems
                if self.architecture == "x86_64":
                    if self.use_openmp:
                        compile_args = self.base_compile_args_unix + ["-march=x86-64-v3"]
                    else:
                        compile_args = self.base_compile_args_plain + ["-march=x86-64-v3"]
                elif self.architecture == "arm64":
                    if self.use_openmp:
                        compile_args = self.base_compile_args_unix + ["-march=armv8-a+simd"]
                    else:
                        compile_args = self.base_compile_args_plain + ["-march=armv8-a+simd"]
                else:
                    if self.use_openmp:
                        compile_args = self.base_compile_args_unix.copy()
                    else:
                        compile_args = self.base_compile_args_plain.copy()
                link_args = self.base_link_args_unix if self.use_openmp else []

        return compile_args, link_args


macos_includes = [
    "/opt/homebrew/include",
    "/usr/local/include",
    "/opt/homebrew/opt/llvm/include",
    "/opt/homebrew/opt/libomp/include",
]
macos_libs = [
    "/opt/homebrew/lib",
    "/usr/local/lib",
    "/opt/homebrew/opt/llvm/lib",
    "/opt/homebrew/opt/libomp/lib",
]


class build_ext_compiler_check(build_ext):
    def finalize_options(self) -> None:
        super().finalize_options()
        if self.parallel is None:
            self.parallel = BUILD_JOBS

    def build_extensions(self) -> None:
        build_config = BuildConfig()
        compiler = self.compiler.compiler_type
        print(f"Compiler detected: {compiler}")
        print(f"Build mode: {'wheel' if build_config.is_wheel_build else 'source'}")
        print(f"Platform: {build_config.platform}")
        print(f"Architecture: {build_config.architecture}")
        print(f"OpenMP enabled: {build_config.use_openmp}")
        print(f"Parallel build jobs: {self.parallel}")

        # Validate an explicit runtime before the legacy fallback can catch errors.
        include_dirs, library_dirs = [], []
        if compiler == "unix" and build_config.platform == "macos" and build_config.use_openmp:
            include_dirs, library_dirs = build_config.get_macos_openmp_paths()

        # Get optimization flags from BuildConfig
        try:
            compile_args, link_args = build_config.get_optimization_flags(compiler)
            print("Using optimized build configuration")
            print(f"Compile args: {compile_args}")
            print(f"Link args: {link_args}")

            for ext in self.extensions:
                # Apply optimized flags
                ext.extra_compile_args = compile_args
                ext.extra_link_args = link_args

        except (AttributeError, TypeError, ValueError) as e:
            print(
                f"Warning: Failed to get optimized flags ({e}), falling back to legacy configuration"
            )
            # Fallback to legacy system
            for ext in self.extensions:
                if compiler == "msvc":
                    if build_config.use_openmp:
                        ext.extra_compile_args = build_config.base_compile_args_msvc.copy()
                    else:
                        ext.extra_compile_args = build_config.base_compile_args_msvc_plain.copy()
                    ext.extra_link_args = []
                elif compiler == "mingw32":
                    if build_config.use_openmp:
                        ext.extra_compile_args = build_config.base_compile_args_unix + [
                            "-lgomp",
                            "-lpthread",
                        ]
                        ext.extra_link_args = ["-lgomp", "-lpthread"]
                    else:
                        ext.extra_compile_args = build_config.base_compile_args_plain + [
                            "-lpthread"
                        ]
                        ext.extra_link_args = ["-lpthread"]
                elif compiler == "unix":
                    if platform.system() == "Darwin":
                        if build_config.use_openmp:
                            ext.extra_compile_args = build_config.base_compile_args_macos.copy()
                            ext.extra_link_args = ["-lomp"]
                        else:
                            ext.extra_compile_args = build_config.base_compile_args_plain.copy()
                            ext.extra_link_args = []
                    else:
                        if build_config.use_openmp:
                            ext.extra_compile_args = build_config.base_compile_args_unix.copy()
                            ext.extra_link_args = build_config.base_link_args_unix
                        else:
                            ext.extra_compile_args = build_config.base_compile_args_plain.copy()
                            ext.extra_link_args = []

        for ext in self.extensions:
            ext.include_dirs.extend(include_dirs)
            ext.library_dirs.extend(library_dirs)
            if library_dirs and os.getenv("ITZI_OPENMP_PREFIX") is not None:
                # The conda runtime uses @rpath/libomp.dylib; delocate needs to resolve it.
                ext.extra_link_args = [*ext.extra_link_args, f"-Wl,-rpath,{library_dirs[0]}"]

        build_ext.build_extensions(self)


extensions = [
    Extension(
        "itzi_core.compute.partial_inertia_h",
        sources=["src/itzi_core/compute/partial_inertia_h.pyx"],
    ),
    Extension(
        "itzi_core.compute.partial_inertia_q",
        sources=["src/itzi_core/compute/partial_inertia_q.pyx"],
    ),
    Extension(
        "itzi_core.compute.rastermetrics", sources=["src/itzi_core/compute/rastermetrics.pyx"]
    ),
    Extension("itzi_core.compute.hydrology", sources=["src/itzi_core/compute/hydrology.pyx"]),
    Extension("itzi_core.compute.snippets", sources=["src/itzi_core/compute/snippets.pyx"]),
]
setup(
    ext_modules=cythonize(extensions, nthreads=BUILD_JOBS),
    cmdclass={"build_ext": build_ext_compiler_check},
)
