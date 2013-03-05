#coding:utf8
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Topics',
    version="0.2.3",
    author='Flávio C. Coelho',
    author_email="fccoelho@gmail.com",
    license="GPLv3",
    url="https://github.com/NAMD/topicmodeling",
    description="Set of tools for topic modeling and visualization",
    ext_modules=cythonize("Topics/visualization/*.pyx"),
    packages=['Topics', 'Topics.onlineldavb', 'Topics.onlinehdp', 'Topics.visualization'],
    install_requires=['numpy>=1.6.2', 'PIL>=1.1.7', 'cython>=0.18', 'regex'],
    #    package_data={'tests': ['data/*']},
)
