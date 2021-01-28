from invoke import task
import re

PKG = "blocked-matrix-utils"
PYVERSIONS = "3.6 3.7 3.8 3.9".split()


@task
def skeleton(c):
    c.run(f'conda skeleton pypi {PKG}')


@task
def build(c):
    for pyversion in PYVERSIONS:
        c.run(f"conda build --py {pyversion} {PKG} -c conda-forge")


@task
def convert(c):
    for pyversion in PYVERSIONS:
        c.run(f"conda convert --platform all linux-64/{PKG}-{version()}-py*.tar.bz2")


@task
def show_version(c):
    c.run(f'echo {version()}')


@task
def show_builds(c):
    c.run(f'ls */{PKG}-{version()}-*.tar.bz2')


@task
def upload(c):
    c.run(f'anaconda upload */{PKG}-{version()}-*.tar.bz2 --force')


def version():
    with open(f"{PKG}/meta.yaml") as meta:
        _version = re.search(
            r'.*% set version = "(.*)" %.*',
            meta.read()
        ).group(1)
    return _version
