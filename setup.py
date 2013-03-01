#coding:utf8
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='Topics',
    version="0.1.0",
    author= 'Flávio C. Coelho',
    author_email= "fccoelho@gmail.com",
    license= "GPLv3",
    description="Set of tools for topic modeling and visualization",
    ext_modules=cythonize("visualization/*.pyx"),
    packages = ['onlineldavb','visualization','onlinehdp','tests'],
    package_data={'tests': ['data/*']},
)
