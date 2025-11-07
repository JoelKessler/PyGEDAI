from setuptools import setup, find_packages

setup(
    name="PyGEDAI",
    version="0.1.0",
    author="Joel Kessler",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
    ]
)