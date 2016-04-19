from setuptools import setup

setup(
    name="util",
    author="Olav Vahtras",
    author_email="olav.vahtras@gmail.com",
    version="1.0",
    url="https://github.com/vahtras/util",
    packages=["util"],
    scripts=["scripts/fb.py"],
    install_requires=["numpy", "scipy"],
)
