import setuptools


setuptools.setup(
    name="polyiv",
    version="0.1",
    author="Charles Li",
    author_email="charlesli@ucsb.edu",
    url="https://github.com/charles9li/polyiv",
    install_requires=['numpy', 'mdtraj'],
    packages=setuptools.find_packages()
)
