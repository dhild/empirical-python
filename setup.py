from setuptools import setup, find_packages

README = open('README.md').read()

requires = [
    'numpy >= 1.7',
    'scipy >= 0.11',
    'matplotlib >= 1.2.1',
    'sympy >= 0.7.2'
]

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.4",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development",
]

setup(name='empirical',
      version='0.3-dev',
      description='Empirical Method of Fundamental Solutions solver',
      long_description=README,
      classifiers=classifiers,
      author='D. Ryan Hild',
      author_email='d.ryan.hild@gmail.com',
      url='http://pypi.python.org/pypi/empirical/',
      packages=find_packages(),
      license='MIT',
      include_package_data=True,
      install_requires=requires,
      )
