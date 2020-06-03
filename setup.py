import setuptools
import sys

extra_install_requires = []
if sys.version_info < (3, 7):
    # Use `dataclasses` package as a backport for Python 3.6.
    extra_install_requires += ["dataclasses"]

setuptools.setup(
    name="spins-meep",
    version="0.0.1",
    python_requires=">=3.6",
    install_requires=[
        "contours[shapely]",
        "dill",
        "flatdict",
        "h5py",
        "jsonschema",
        "matplotlib",
        "numpy",
        "requests",
        "schematics",
        "scipy",
        "typing-inspect",
    ] + extra_install_requires,
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-xdist",
        ],
        "dev": [
            "pylint",
            "pytype",
            "yapf",
        ],
    },
    packages=["spins.goos_sim.meep"],
)
