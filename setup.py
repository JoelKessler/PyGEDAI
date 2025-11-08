from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pygedai",
    description="PyGEDAI: Generalized Eigenvalue De-Artifacting Instrument in PyTorch",
    version="0.1.0",
    author="Joel Kessler",
    license="PolyForm Noncommercial License 1.0.0",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license_files=("LICENSE",),
    classifiers=[
        "License :: Other/Proprietary License",
        "Intended Audience :: Science/Research"
    ],
    url="https://github.com/JoelKessler/PyGEDAI",
    project_urls={
        "Source": "https://github.com/JoelKessler/PyGEDAI",
        "License": "https://github.com/JoelKessler/PyGEDAI/blob/main/LICENSE",
        "Bug Tracker": "https://github.com/JoelKessler/PyGEDAI/issues",
        "Documentation": "https://github.com/JoelKessler/PyGEDAI#readme",
    },
    install_requires=[
        # pure-Python deps here, but NOT torch
    ],
    extras_require={
        # do: pip install pygedai[torch]
        "torch": ["torch>=2.2.0"], # no CUDA channel here, choose at install time
    },
)