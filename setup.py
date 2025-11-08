from setuptools import setup, find_packages

setup(
    name="pygedai",
    version="0.1.0",
    author="Joel Kessler",
    packages=find_packages(),
    install_requires=[
        # pure-Python deps here, but NOT torch
    ],
    extras_require={
        # do: pip install pygedai[torch]
        "torch": ["torch>=2.2.0"], # no CUDA channel here, choose at install time
    },
)