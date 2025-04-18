from setuptools import setup, find_packages

setup(
    name="pyscsopt",
    version="0.1.0",
    description="A Python library for self-concordant smooth optimization (Python port of SelfConcordantSmoothOptimization.jl)",
    author="Adeyemi Damilare Adeoye",
    author_email="",
    url="https://github.com/adeyemiadeoye/pyscsopt",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "jax",
    ],
    python_requires=">=3.8",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)