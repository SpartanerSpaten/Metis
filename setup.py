from setuptools import setup
import sys

REQUIRED = ["numpy",]
PACKAGES = ["Metis_CPU","Additional",]


if "--only_pycuda" in sys.argv:
    PACKAGES = ["Additional"]

if "--pycuda" in sys.argv or "--only_pycuda" in sys.argv:
    REQUIRED.append("pycuda","skcuda")
    PACKAGES.append("Metis_Pycuda")

setup(
    name='metis',
    version='0.0.5',
    packages=PACKAGES,
    install_requires=REQUIRED,
    #include_package_data=True,
    #data_files=[("",["./Additional/_Internal.py","./Additional/etc.py","./Additional/Functions.py","./Additional/Image.py"])],
    url='https://github.com/SpartanerSpaten/Metis',
    license='MIT',
    author='dre',
    author_email='espriworkemail@gmail.com',
    description=' Metis is a light deep learning framework. ',
    long_description=open('README.md').read()
)
