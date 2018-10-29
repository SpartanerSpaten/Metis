from setuptools import setup
import sys

REQUIRED = ["numpy","marshal",]
PACKAGES = ['Metis_CPU',]


if "--only_pycuda" in sys.argv:
    PACKAGES = []

if "--pycuda" in sys.argv or "--only_pycuda" in sys.argv:
    REQUIRED.append("pycuda","skcuda")
    PACKAGES.append("Metis_Pycuda")

setup(
    name='metis',
    version='0.0.4',
    packages=PACKAGES,
    install_requires=REQUIRED,
    url='https://github.com/SpartanerSpaten/Metis',
    license='MIT',
    author='dre',
    author_email='espriworkemail@gmail.com',
    description=' Metis is a light deep learning framework. ',
    long_description=open('README.md').read()
)
