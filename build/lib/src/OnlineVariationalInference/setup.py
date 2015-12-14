from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


ext_module = Extension(
    "lda_vi_cython",
    ["lda_vi_cython.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
    name = 'Hello world app',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
)
        