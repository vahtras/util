from setuptools import setup
from util import __version__

setup(
    name="blocked-matrix-utils",
    author="Olav Vahtras",
    author_email="olav.vahtras@gmail.com",
    version=__version__,
    url="https://github.com/vahtras/util",
    packages=["util"],
    install_requires=["numpy", "scipy"],
)
